# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
from codecs import ignore_errors

import torch
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       masked_sum, masked_mean, masked_any, masked_all)

from .backbone import build_backbone
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_verbs, num_nouns, num_verb_queries, num_role_queries, verb_role_tgt_mask):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_nouns: number of object classes
            num_role_queries: number of role queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image.
        """
        super().__init__()
        self.num_verb_queries = num_verb_queries
        self.num_role_queries = num_role_queries
        self.verb_role_tgt_mask = verb_role_tgt_mask
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.nhead = transformer.nhead
        self.query_embed = nn.Embedding(num_verb_queries + num_role_queries, hidden_dim)
        # self.verb_linear = nn.Linear(hidden_dim, num_verbs)
        from torchvision.models import vgg16_bn
        vgg = vgg16_bn()
        num_features = vgg.classifier[6].in_features
        features = list(vgg.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, num_verbs)]) # Add our layer with 4 outputs
        features.insert(0, nn.Flatten(-3)) # Add our layer with 4 outputs
        self.verb_linear = nn.Sequential(*features)
        self.noun_linear = nn.Linear(hidden_dim, num_nouns)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
               - verbs: batched gt verbs of sample images [batch_size x 1]
               - roles: bathced roles according to gt verbs of sample iamges [batch_size x (max role:6)]

            It returns a dict with the following elements:
               - "pred_logits": the classification logits for role queries.
                                Shape= [batch_size x num_roles x num_nouns]
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        out = {}

        decoder_tgt_mask = self.verb_role_tgt_mask
        decoder_memory_mask = None

        # hs: num_layers x batch_size x len(query_embed) x hidden_dim
        hs = self.transformer(
            self.input_proj(src), mask, self.query_embed.weight, pos[-1], decoder_tgt_mask, decoder_memory_mask)[0]
        verb_hs, role_hs = hs.split([self.num_verb_queries, self.num_role_queries], dim=2)
        # import pdb
        # pdb.set_trace()
        outputs_verb = self.verb_linear(src)[None,:,None,:]
        out.update({'pred_verb': outputs_verb[-1]})

        outputs_class = self.noun_linear(role_hs)
        out.update({'pred_logits': outputs_class[-1]})

        return out


class imSituCriterion(nn.Module):
    """ This class computes the loss for DETR with imSitu dataset.
    """

    def __init__(self, num_roles, pad_noun, weight_dict):
        """ Create the criterion.
        """
        super().__init__()
        self.num_roles = num_roles
        self.pad_noun = pad_noun
        self.weight_dict = weight_dict
        self.loss_function = nn.CrossEntropyLoss(ignore_index=pad_noun, reduction='none')
        self.loss_function_for_noun = nn.CrossEntropyLoss(ignore_index=pad_noun)
        self.loss_function_for_verb = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        topk = 5
        assert 'pred_verb' in outputs
        # batch_size x num_verbs
        verb_pred_logits = outputs['pred_verb'].squeeze(dim=1)
        # batch_size
        gt_verbs = torch.stack([t['verb'] for t in targets])
        # ()
        verb_loss = self.loss_function_for_verb(verb_pred_logits, gt_verbs)
        # batch_size x topk -> topk x batch_size
        verb_correct = verb_pred_logits.topk(topk)[1].eq(gt_verbs.unsqueeze(-1)).t().cumsum(0).bool()
        # import pdb
        # pdb.set_trace()

        assert 'pred_logits' in outputs
        # batch_size x num_roles x num_nouns
        pred_logits = outputs['pred_logits']
        device = pred_logits.device

        # batch_size x num_roles x num_targets
        labels = torch.stack([t['labels'] for t in targets]).long()
        lmask = (labels != self.pad_noun)

        # batch_size x num_roles x num_targets
        noun_loss = torch.stack([self.loss_function(
            pred_logits.transpose(1, 2),  # batch_size x num_nouns x num_roles
            labels[:, :, n]  # batch_size x num_roles
        ) for n in range(3)], axis=2)
        # batch_size x num_roles
        noun_loss = masked_sum(noun_loss, lmask, dim=-1)
        # batch_size
        noun_loss = masked_mean(noun_loss, lmask.any(-1), dim=-1)
        # ()
        noun_loss = masked_mean(noun_loss, lmask.any(-1).any(-1), dim=-1)

        # batch_size x num_roles x num_targets
        noun_correct = pred_logits.argmax(-1).unsqueeze(-1).eq(labels)
        # topk x batch_size x num_roles x num_targets
        noun_correct_verb = noun_correct & verb_correct.unsqueeze(-1).unsqueeze(-1)
        # topk x batch_size x num_roles
        noun_correct_verb = masked_any(noun_correct_verb, lmask, dim=-1)
        # batch_size x num_roles
        noun_correct = masked_any(noun_correct, lmask, dim=-1)
        # topk x batch_size
        noun_correct_verb_all = masked_all(noun_correct_verb, lmask.any(-1), dim=-1)
        noun_acc_verb = masked_mean(noun_correct_verb, lmask.any(-1), dim=-1)
        # batch_size
        noun_correct_all = masked_all(noun_correct, lmask.any(-1), dim=-1)
        noun_acc = masked_mean(noun_correct, lmask.any(-1), dim=-1)

        stat = {'loss_vce': verb_loss,
                'loss_nce': noun_loss,
                'verb_top1_acc': verb_correct[0].float().mean()*100,
                'noun_top1_acc': noun_acc_verb[0].mean()*100,
                'noun_top1_acc_all': noun_correct_verb_all[0].float().mean()*100,
                'verb_top5_acc': verb_correct[4].float().mean()*100,
                'noun_top5_acc': noun_acc_verb[4].mean()*100,
                'noun_top5_acc_all': noun_correct_verb_all[4].float().mean()*100,
                'noun_gt_acc': noun_acc.mean()*100,
                'noun_gt_acc_all': noun_correct_all.float().mean()*100,
                'class_error': torch.tensor(0.).to(device)}
        stat.update({'mean_acc': torch.stack([v for k, v in stat.items() if 'acc' in k]).mean()})

        return stat


def build(args):
    if args.use_role_adj_attn_mask:
        verb_role_tgt_mask = torch.tensor(~args.vr_adj_mat.any(0))
    else:
        verb_role_tgt_mask = None

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_verbs=args.num_verbs,
        num_nouns=args.num_nouns,
        num_verb_queries=args.num_verb_queries,
        num_role_queries=args.num_role_queries,
        verb_role_tgt_mask=verb_role_tgt_mask,
    )
    weight_dict = {'loss_vce': args.verb_loss_coef, 'loss_nce': args.noun_loss_coef}
    criterion = imSituCriterion(args.num_roles, args.pad_noun, weight_dict=weight_dict)

    return model, criterion
