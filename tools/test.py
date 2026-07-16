#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2026 SpecLoc Authors
# Modified by SpecLoc Authors from MMDetection's testing utility.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os.path as osp
from typing import Optional

from cliUtils import (
    add_cfg_options_arg,
    add_launcher_args,
    build_runner,
    load_config,
    resolve_work_dir,
    setup_local_rank,
)


def parse_args():
    parser = argparse.ArgumentParser(description='评估 SpecLoc 检测配置并导出预测')
    parser.add_argument('config', help='Path to the test config file')
    parser.add_argument('checkpoint', help='Path to the checkpoint file')
    parser.add_argument('--work-dir', help='Directory to save evaluation results')
    parser.add_argument(
        '--out',
        help='COCO 预测 JSON 路径；推荐以 .bbox.json 结尾',
    )
    parser.add_argument('--show', action='store_true', help='Show prediction results')
    parser.add_argument('--show-dir', help='Directory to save visualization images')
    add_cfg_options_arg(parser)
    add_launcher_args(parser)
    args = parser.parse_args()
    setup_local_rank(args)
    return args


def _enable_visualization(cfg, show: bool, show_dir: Optional[str]) -> None:
    if not (show or show_dir):
        return
    vis_hook = cfg.default_hooks.get('visualization')
    if vis_hook is None:
        cfg.default_hooks['visualization'] = dict(
            type='DetVisualizationHook',
            enable=True,
            show=show,
        )
    else:
        vis_hook['enable'] = True
        vis_hook['show'] = show
    if show_dir:
        cfg.default_hooks.visualization['test_out_dir'] = show_dir


def main():
    args = parse_args()

    cfg = load_config(args.config, args.cfg_options)
    resolve_work_dir(cfg, args.config, args.work_dir)

    cfg.load_from = args.checkpoint
    cfg.launcher = args.launcher

    if args.out is not None:
        cfg.test_evaluator['format_only'] = False
        if args.out.endswith('.bbox.json'):
            cfg.test_evaluator['outfile_prefix'] = args.out[:-len('.bbox.json')]
        else:
            cfg.test_evaluator['outfile_prefix'] = osp.splitext(args.out)[0]

    _enable_visualization(cfg, args.show, args.show_dir)
    build_runner(cfg).test()


if __name__ == '__main__':
    main()
