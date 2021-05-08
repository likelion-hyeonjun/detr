# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_verb_emb_1_w_role import build

#change .py file to use appropriate transformer model.
def build_model(args):
    return build(args)
