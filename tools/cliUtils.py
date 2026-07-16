# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2026 SpecLoc Authors
# Modified by SpecLoc Authors from MMDetection's CLI utilities.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for SpecLoc CLI entrypoints."""

from __future__ import annotations

import os
import os.path as osp
from argparse import ArgumentParser, Namespace
from typing import Optional

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner


def add_cfg_options_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config settings, e.g. KEY=VAL',
    )


def add_launcher_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher',
    )
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)


def setup_local_rank(args: Namespace) -> None:
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)


def load_config(config_path: str, cfg_options: Optional[dict] = None) -> Config:
    cfg = Config.fromfile(config_path)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    return cfg


def resolve_work_dir(
    cfg: Config,
    config_path: str,
    work_dir: Optional[str] = None,
) -> None:
    if work_dir is not None:
        cfg.work_dir = work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs',
            osp.splitext(osp.basename(config_path))[0],
        )


def build_runner(cfg: Config) -> Runner:
    if 'runner_type' not in cfg:
        return Runner.from_cfg(cfg)
    return RUNNERS.build(cfg)


def apply_train_resume(cfg: Config, resume: Optional[str]) -> None:
    if resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif resume is not None:
        cfg.resume = True
        cfg.load_from = resume


def apply_amp(cfg: Config) -> None:
    optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
    if optim_wrapper != 'OptimWrapper':
        raise ValueError('`--amp` requires optim_wrapper.type=OptimWrapper')
    cfg.optim_wrapper.type = 'AmpOptimWrapper'
    cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')


def apply_auto_scale_lr(cfg: Config) -> None:
    cfg.setdefault('auto_scale_lr', dict(enable=True))
    cfg.auto_scale_lr.enable = True


def disable_apple_mps() -> None:
    """Prevent MMCV ops from routing to Apple MPS on macOS."""
    import torch
    from mmengine.device import utils as device_utils

    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False
    # MMEngine caches the selected device when it is imported. Updating the
    # PyTorch probes alone is therefore too late for CLI entrypoints.
    device_utils.DEVICE = 'cpu'
