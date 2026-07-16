#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2026 SpecLoc Authors
# Modified by SpecLoc Authors from MMDetection's training utility.
# SPDX-License-Identifier: Apache-2.0

import argparse

from cliUtils import (
    add_cfg_options_arg,
    add_launcher_args,
    apply_amp,
    apply_auto_scale_lr,
    apply_train_resume,
    build_runner,
    disable_apple_mps,
    load_config,
    resolve_work_dir,
    setup_local_rank,
)
from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='训练 SpecLoc 数据集配置指定的检测基线')
    parser.add_argument('config', help='Path to the training config file')
    parser.add_argument('--work-dir', help='Directory to save logs and checkpoints')
    parser.add_argument('--amp', action='store_true', help='Enable AMP training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='Enable automatic LR scaling by batch size',
    )
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='Resume training. Omit value for auto-resume.',
    )
    parser.add_argument(
        '--disable-mps',
        action='store_true',
        help='Disable Apple MPS backend (recommended on macOS)',
    )
    add_cfg_options_arg(parser)
    add_launcher_args(parser)
    args = parser.parse_args()
    setup_local_rank(args)
    return args


def main():
    args = parse_args()

    if args.disable_mps:
        disable_apple_mps()

    setup_cache_size_limit_of_dynamo()

    cfg = load_config(args.config, args.cfg_options)
    resolve_work_dir(cfg, args.config, args.work_dir)

    if args.amp:
        apply_amp(cfg)
    if args.auto_scale_lr:
        apply_auto_scale_lr(cfg)
    apply_train_resume(cfg, args.resume)

    cfg.launcher = args.launcher
    build_runner(cfg).train()


if __name__ == '__main__':
    main()
