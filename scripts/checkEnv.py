#!/usr/bin/env python3
"""检查 SpecLoc 云端环境、配置、模型和选定数据集。"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = {
    'rsod': {
        'config': REPO / 'configs' / 'rsodCenternetR50.py',
        'root': REPO / 'data' / 'RSOD',
        'manifest_dataset': 'RSOD',
        'targets': (
            ('图像目录', 'images'),
            ('训练标注', 'annotations/instances_train.json'),
            ('验证标注', 'annotations/instances_val.json'),
            ('测试标注', 'annotations/instances_test.json'),
            ('图像校验和', 'images.sha256'),
        ),
    },
    'aitod': {
        'config': REPO / 'configs' / 'aitodCenternetR50.py',
        'root': REPO / 'data' / 'AI-TOD',
        'manifest_dataset': 'AI-TOD-v2',
        'targets': (
            ('训练图像目录', 'train'),
            ('验证图像目录', 'val'),
            ('训练标注', 'annotations/aitod_train_v2.json'),
            ('验证标注', 'annotations/aitod_val_v2.json'),
        ),
    },
}

VERSION_CONTRACT = {
    'setuptools': '80.10.2',
    'torch': '2.1.2',
    'torchvision': '0.16.2',
    'mmcv': '2.1.0',
    'mmdet': '3.3.0',
    'numpy': '1.26.4',
}


@dataclass
class Result:
    name: str
    passed: bool
    detail: str = ''
    required: bool = True


def check_imports() -> list[Result]:
    results = []
    for module in (
        'setuptools', 'torch', 'torchvision', 'mmcv', 'mmengine', 'mmdet',
        'numpy', 'scipy', 'pandas', 'PIL', 'yaml', 'specloc',
    ):
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, '__version__', 'available')
            results.append(Result(f'依赖 {module}', True, str(version)))
        except Exception as exc:
            results.append(Result(f'依赖 {module}', False, str(exc)))
    return results


def check_version_contract() -> list[Result]:
    """核对云端复现实验要求的精确核心版本。"""
    results = []
    for module, expected in VERSION_CONTRACT.items():
        try:
            imported = importlib.import_module(module)
            actual = str(imported.__version__)
            # PyTorch CPU/CUDA wheel 常附带 +cpu、+cu118 等本地版本标记。
            normalized = actual.split('+', 1)[0]
            results.append(Result(
                f'版本合同 {module}',
                normalized == expected,
                f'expected={expected}, actual={actual}',
            ))
        except Exception as exc:
            results.append(Result(f'版本合同 {module}', False, str(exc)))
    return results


def check_files(dataset: str) -> list[Result]:
    dataset_files = {
        'rsod': ('configs/rsodCenternetR50.py', 'scripts/validateRsod.py'),
        'aitod': ('configs/aitodCenternetR50.py', 'scripts/validateAitod.py'),
    }
    required = (
        *dataset_files[dataset],
        'scripts/runCloudExperiment.sh',
        'requirements.txt',
        'src/specloc/cli.py',
        'tools/train.py',
        'tools/test.py',
        'tools/distTrain.sh',
        'tools/analysisTools/extractSpectrumObjectTable.py',
        'tools/analysisTools/spectrumInformationGate.py',
    )
    return [
        Result(f'文件 {relative}', (REPO / relative).is_file())
        for relative in required
    ]


def check_git() -> Result:
    try:
        revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if status:
            return Result('Git 版本', False, f'{revision}; 工作树有未提交改动')
        return Result('Git 版本', True, f'{revision}; 工作树干净')
    except (OSError, subprocess.SubprocessError) as exc:
        return Result('Git 版本', False, f'请保留 .git：{exc}')


def check_config_and_model(config_path: Path) -> list[Result]:
    try:
        from mmdet.registry import MODELS
        from mmdet.utils import register_all_modules
        from mmengine.config import Config

        import specloc.registry  # noqa: F401

        register_all_modules(init_default_scope=True)
        cfg = Config.fromfile(str(config_path))
        result = [Result('配置加载', True, str(config_path.relative_to(REPO)))]
        model = MODELS.build(cfg.model)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        result.append(Result('模型构建', True, f'{parameters:,} parameters'))
        expected_classes = len(cfg.get('classes', ()))
        actual_classes = cfg.model.bbox_head.num_classes
        result.append(Result(
            '类别数量',
            expected_classes == actual_classes,
            f'config={expected_classes}, model={actual_classes}',
        ))
        return result
    except Exception as exc:
        return [Result('配置与模型', False, str(exc))]


def check_data(dataset: str, skip_data: bool) -> list[Result]:
    spec = DATASETS[dataset]
    root = spec['root']
    results = [
        Result(name, (root / relative).exists(), str(root / relative), required=not skip_data)
        for name, relative in spec['targets']
    ]
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        validator = 'validateRsod.py' if dataset == 'rsod' else 'validateAitod.py'
        results.append(Result(
            '数据 manifest',
            False,
            f'先运行 scripts/{validator}',
            required=not skip_data,
        ))
        return results
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        status_valid = manifest.get('status') == 'verified'
        dataset_valid = manifest.get('dataset') == spec['manifest_dataset']
        results.append(Result(
            '数据 manifest',
            status_valid and dataset_valid,
            f"dataset={manifest.get('dataset')}, status={manifest.get('status')}",
            required=not skip_data,
        ))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        results.append(Result('数据 manifest', False, str(exc), required=not skip_data))
    return results


def check_gpu(require_gpu: bool) -> Result:
    try:
        import torch

        if torch.cuda.is_available():
            names = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            return Result(
                'CUDA GPU',
                True,
                f"CUDA {torch.version.cuda}; {', '.join(names)}",
                required=require_gpu,
            )
        return Result('CUDA GPU', False, 'torch.cuda.is_available() == False', require_gpu)
    except Exception as exc:
        return Result('CUDA GPU', False, str(exc), require_gpu)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(DATASETS), default='rsod')
    parser.add_argument(
        '--skip-data', action='store_true', help='CI 或代码检查时不要求数据'
    )
    parser.add_argument('--require-gpu', action='store_true', help='将 CUDA GPU 作为必需项')
    parser.add_argument(
        '--strict-versions',
        action='store_true',
        help='强制核对 requirements/ 中约定的核心依赖版本',
    )
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    results = []
    results.extend(check_imports())
    if args.strict_versions:
        results.extend(check_version_contract())
    results.extend(check_files(args.dataset))
    results.append(check_git())
    results.extend(check_config_and_model(spec['config']))
    results.extend(check_data(args.dataset, args.skip_data))
    results.append(check_gpu(args.require_gpu))

    print('=' * 72)
    print(f'SpecLoc 最小项目检查（dataset={args.dataset}）')
    print('=' * 72)
    for item in results:
        if item.passed:
            marker = '通过'
        elif item.required:
            marker = '失败'
        else:
            marker = '提示'
        suffix = f' — {item.detail}' if item.detail else ''
        print(f'[{marker}] {item.name}{suffix}')

    required = [item for item in results if item.required]
    failures = [item for item in required if not item.passed]
    print('-' * 72)
    print(f'必需检查：{len(required) - len(failures)}/{len(required)} 通过')
    if failures:
        print('不能开始正式训练。')
        return 1
    print('可以开始正式训练。' if not args.skip_data else '代码侧检查通过。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
