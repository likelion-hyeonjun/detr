# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
from codecs import ignore_errors

import torch
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, accuracy_swig)

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
        self.verb_linear = nn.Linear(hidden_dim, num_verbs)
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

        outputs_verb = self.verb_linear(verb_hs)
        out.update({'pred_verb': outputs_verb[-1]})

        outputs_class = self.noun_linear(role_hs)
        out.update({'pred_logits': outputs_class[-1]})

        return out


def masked_sum(input, mask, dim):
    return (input / mask).nansum(dim)


def masked_mean(input, mask, dim):
    return ((input / mask).nansum(dim) / mask.sum(dim)).nan_to_num(0)


def masked_all(input, mask, dim):
    return (input / mask).nansum(dim).eq(mask.sum(dim))


def masked_any(input, mask, dim):
    return (input / mask).nansum(dim).bool()


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
        # top1_verb = verb_correct[:1].any(0).float().mean()
        # top5_verb = verb_correct[:5].any(0).float().mean()

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

        batch_noun_loss = []
        batch_noun_acc = []
        for p, t in zip(pred_logits, targets):
            roles = t['roles']
            num_roles = len(roles)
            role_pred = p[roles]
            role_targ = t['labels'][roles]
            role_targ = role_targ.long()
            batch_noun_acc += accuracy_swig(role_pred, role_targ)

            role_noun_loss = []
            for n in range(3):
                role_noun_loss.append(self.loss_function_for_noun(role_pred, role_targ[:, n]))
            batch_noun_loss.append(sum(role_noun_loss))
        if not noun_loss.allclose(torch.stack(batch_noun_loss).mean()):
            import pdb
            pdb.set_trace()
            noun_loss - torch.stack(batch_noun_loss).mean()
        # assert (noun_loss == torch.stack(batch_noun_loss).mean()).all()
        if not torch.stack(batch_noun_acc).mean().allclose(noun_acc.mean()*100):
            import pdb
            pdb.set_trace()
            torch.stack(batch_noun_acc).mean() - (noun_acc*100).mean()
        # assert (noun_acc.mean() == noun_correct_gt[:1].any(0).float().mean()).all()

        batch_noun_acc_topk = []
        for verbs in verb_pred_logits.topk(5)[1].transpose(0, 1):
            batch_noun_acc = []
            for v, p, t in zip(verbs, pred_logits, targets):
                if v == t['verb']:
                    roles = t['roles']
                    num_roles = len(roles)
                    role_targ = t['labels'][roles]
                    role_targ = role_targ.long().cuda()
                    role_pred = p[roles]
                    batch_noun_acc += accuracy_swig(role_pred, role_targ)
                else:
                    batch_noun_acc += [torch.tensor(0., device=device)]
            batch_noun_acc_topk.append(torch.stack(batch_noun_acc))
        noun_acc_topk = torch.stack(batch_noun_acc_topk)
        assert noun_acc_topk[0].mean().allclose(noun_acc_verb[0].mean()*100)
        if not noun_acc_topk.sum(0).mean().allclose(noun_acc_verb[4].mean()*100):
            import pdb
            pdb.set_trace()

        if noun_correct_all.float().mean():
            import pdb
            pdb.set_trace()

        stat = {'loss_vce': verb_loss,
                'loss_nce': noun_loss,
                'verb_top1_acc': verb_correct[0].float().mean(),
                'noun_top1_acc': noun_acc_verb[0].mean(),
                'noun_top1_acc_all': noun_correct_verb_all[0].float().mean(),
                'verb_top5_acc': verb_correct[4].float().mean(),
                'noun_top5_acc': noun_acc_verb[4].mean(),
                'noun_top5_acc_all': noun_correct_verb_all[4].float().mean(),
                'noun_gt_acc': noun_acc.mean(),
                'noun_gt_acc_all': noun_correct_all.float().mean(),
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
