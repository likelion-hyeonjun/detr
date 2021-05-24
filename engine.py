# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import numpy as np
import pandas as pd
from typing import Iterable

import torch

import util.misc as utils
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if type(v) is not str else v for k, v in t.items()} for t in targets]

        outputs, attn_weight, img_attn_weight = model(samples, need_weights =False)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_swig(model, criterion, data_loader, device, output_dir, csv_folder, need_weights):
    # TODO
    # Need check
    model.eval()
    criterion.eval()

    role_attn_list = []
    img_attn_list = []
    gt_and_pred_noun_list =[]

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if type(v) is not str else v for k, v in t.items()} for t in targets]

        outputs, attn_weight, img_attn_weight = model(samples, need_weights)

        #for role1-6
        if need_weights: 
            assert attn_weight is not None and img_attn_weight is not None

            image_name = targets[0]["image_name"].split('/')[2]
            #TODO index -> noun 
            gt_roles = targets[0]['roles'].cpu().numpy()
            gt_roles_w_pad = np.append(gt_roles, [191 for i in range(6-len(gt_roles))])
            gt_and_pred_noun_list.append((image_name +" GTroles w pad", pd.DataFrame(gt_roles_w_pad.reshape(-1,6))))
            noun_predict = outputs['pred_logits'].squeeze()[:len(gt_roles)]
            top5_pred_noun_indices = torch.transpose(torch.topk(noun_predict,5)[1],0,1).cpu().numpy()
            gt_nouns = torch.transpose(targets[0]['labels'][:len(gt_roles),:],0,1).cpu().numpy()
            for i in range(5):
                gt_and_pred_noun_list.append((image_name+" top {}".format(i+1),pd.DataFrame(top5_pred_noun_indices[i].reshape(-1, len(gt_roles))))) #
            for i in range(3): #num frame
                gt_and_pred_noun_list.append((image_name+" frame {}".format(i+1),pd.DataFrame(gt_nouns[i].reshape(-1, len(gt_roles))))) # 3 (# frame) * len(gt roles)
            
            role_attn = pd.DataFrame(attn_weight.cpu().numpy())
            img_attn = pd.DataFrame(img_attn_weight.cpu().numpy()) #should reshape to  6 * 49 * 190
            role_attn_list.append((image_name,role_attn))
            img_attn_list.append((image_name,img_attn))

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

    if need_weights:
        assert csv_folder is not None, "There is no indicateded csv_folder name"
        role_attn = pd.concat(dict(role_attn_list))
        role_attn.to_csv(csv_folder+"/role_attn.csv", mode="w")
        img_attn = pd.concat(dict(img_attn_list))
        img_attn.to_csv(csv_folder+"/img_attn.csv", mode="w")
        result = pd.concat(dict(gt_and_pred_noun_list))
        result.to_csv(csv_folder+"/result.csv", mode="w")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats
