"""Write a machine-readable provenance record before a training run."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import mmcv
import mmdet
import mmengine
import torch
from mmengine.fileio import dump
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from specloc import __version__


def _git_revision(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return 'UNVERSIONED'


def _git_dirty(repo: Path) -> bool | None:
    try:
        output = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def _data_manifest(config) -> dict:
    path_value = config.get('data_manifest', None)
    if path_value is None:
        return {'status': 'UNVERIFIED', 'path': None}
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        return {'status': 'UNVERIFIED', 'path': str(path.resolve())}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            'status': 'INVALID_MANIFEST',
            'path': str(path.resolve()),
            'error': str(exc),
        }


@HOOKS.register_module()
class ExperimentMetadataHook(Hook):
    """在训练开始前保存代码、环境、数据、模型和随机种子信息。"""

    priority = 'VERY_HIGH'

    def before_run(self, runner) -> None:
        work_dir = Path(runner.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        repo = Path(__file__).resolve().parents[4]
        cuda_available = torch.cuda.is_available()
        data_manifest = _data_manifest(runner.cfg)
        git_revision = _git_revision(repo)
        git_dirty = _git_dirty(repo)
        metadata = {
            'project': 'SpecLoc',
            'project_version': __version__,
            'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'config': getattr(runner.cfg, 'filename', None),
            'work_dir': str(work_dir.resolve()),
            'git_revision': git_revision,
            'git_dirty': git_dirty,
            'data_version': runner.cfg.get('data_version', 'UNVERIFIED'),
            'data_manifest': data_manifest,
            'randomness': dict(runner.cfg.get('randomness', {})),
            'model_type': runner.model.__class__.__name__,
            'command': sys.argv,
            'environment': {
                'python': sys.version.split()[0],
                'platform': platform.platform(),
                'conda_environment': os.environ.get('CONDA_DEFAULT_ENV', 'UNKNOWN'),
                'torch': torch.__version__,
                'mmcv': mmcv.__version__,
                'mmdet': mmdet.__version__,
                'mmengine': mmengine.__version__,
                'cuda_available': cuda_available,
                'cuda_runtime': torch.version.cuda,
                'devices': [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ] if cuda_available else ['cpu'],
            },
        }
        dump(metadata, work_dir / 'experiment_metadata.json', indent=2)
        if (
            runner.cfg.get('require_verified_data', False)
            and data_manifest.get('status') != 'verified'
        ):
            raise RuntimeError(
                '正式训练要求已验证的数据 manifest。请先运行 '
                f'`{runner.cfg.get("data_validator", "对应的数据校验脚本")}`。'
            )
        if runner.cfg.get('require_clean_git', False) and (
            git_revision == 'UNVERSIONED' or git_dirty is not False
        ):
            raise RuntimeError(
                '正式训练要求有版本记录且工作树干净。请提交改动后重试。'
            )
