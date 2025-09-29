# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.train_cfg.warmup_epochs:
        lr = args.train_cfg.lr * epoch / args.train_cfg.warmup_epochs 
    else:
        lr = args.train_cfg.min_lr + (args.train_cfg.lr - args.train_cfg.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.train_cfg.warmup_epochs) / (args.train_cfg.epochs - args.train_cfg.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
