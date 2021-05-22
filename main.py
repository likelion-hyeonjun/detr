# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from datasets.imsitu import collater
import util.misc as utils
from datasets import build_dataset
from engine import evaluate_swig, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_backbone', default=5e-5, type=float)
    parser.add_argument('--optimizer', default="Adamax", type=str,
                        choices=["Adam", "AdamW", "Adamax", "SGD", "RmsProp"])
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--clip_max_norm', default=0.25, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_verb_queries', type=int, default=1,
                        help="Number of verb query slots")
    parser.add_argument('--num_role_queries', type=int, default=190,
                        help="Number of role query slots")
    parser.add_argument('--use_role_adj_attn_mask', action="store_true",
                        help="Use role adjacency matrix as attention mask")
    parser.add_argument('--pre_norm', action='store_true')

    # * Loss coefficients
    parser.add_argument('--verb_loss_coef', default=1, type=float)
    parser.add_argument('--noun_loss_coef', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='imsitu')
    parser.add_argument('--imsitu_path', type=str, default="imSitu")

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    elif args.optimizer == 'Adamw':
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr)
    elif args.optimizer == 'Adamax':
        optimizer = torch.optim.Adamax(param_dicts, lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr)
    elif args.optimizer == 'RmsProp':
        optimizer = torch.optim.RmsProp(param_dicts, lr=args.lr)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    batch_sampler_val = torch.utils.data.BatchSampler(sampler_val, args.batch_size, drop_last=False)
    data_loader_train = DataLoader(dataset_train, num_workers=args.num_workers,
                                   collate_fn=collater, batch_sampler=batch_sampler_train)
    data_loader_val = DataLoader(dataset_val, num_workers=args.num_workers,
                                 drop_last=False, collate_fn=collater, batch_sampler=batch_sampler_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats = evaluate_swig(model, criterion, data_loader_val, device, args.output_dir)
        return

    min_test_loss = np.inf
    max_test_acc_mean = -np.inf
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)

        test_stats = evaluate_swig(model, criterion,
                                   data_loader_val, device, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint for every new min loss or new max acc
            if log_stats['test_loss'] < min_test_loss or log_stats['test_mean_acc_unscaled'] > max_test_acc_mean:
                min_test_loss = log_stats['test_loss'] if log_stats['test_loss'] < min_test_loss else min_test_loss
                max_test_acc_mean = log_stats['test_mean_acc_unscaled'] if log_stats['test_mean_acc_unscaled'] > max_test_acc_mean else max_test_acc_mean
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
