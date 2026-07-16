"""Stable command-line interface for SpecLoc research workflows."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from specloc import __version__

DATASETS = {
    'rsod': {
        'config': 'configs/rsodCenternetR50.py',
        'data_root': 'data/RSOD',
        'validator': 'scripts/validateRsod.py',
        'work_dir': 'work_dirs/rsod_centernet_r50',
    },
    'aitod': {
        'config': 'configs/aitodCenternetR50.py',
        'data_root': 'data/AI-TOD',
        'validator': 'scripts/validateAitod.py',
        'work_dir': 'work_dirs/aitod_centernet_r50',
    },
}


def find_repo_root(start: Path | None = None) -> Path:
    """Find a SpecLoc checkout without depending on the shell's directory."""
    override = os.environ.get('SPECLOC_ROOT')
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    current = (start or Path.cwd()).resolve()
    candidates.extend((current, *current.parents))
    candidates.append(Path(__file__).resolve().parents[2])
    for candidate in candidates:
        if (
            (candidate / 'pyproject.toml').is_file()
            and (candidate / 'configs' / 'rsodCenternetR50.py').is_file()
        ):
            return candidate.resolve()
    raise RuntimeError(
        '未找到 SpecLoc 仓库根目录。请在仓库内运行，或设置 SPECLOC_ROOT。'
    )


def run_command(command: list[str], repo: Path) -> None:
    """Show and execute a child command with consistent failure handling."""
    print(f'$ {shlex.join(command)}', flush=True)
    subprocess.run(command, cwd=repo, check=True)


def validator_command(args: argparse.Namespace, repo: Path) -> list[str]:
    spec = DATASETS[args.dataset]
    command = [
        sys.executable,
        str(repo / spec['validator']),
        '--root',
        str(repo / spec['data_root']),
    ]
    if args.dataset == 'aitod':
        source = args.source or os.environ.get('AITOD_SOURCE')
        license_note = args.license_note or os.environ.get('AITOD_LICENSE_NOTE')
        if source:
            command.extend(['--source', source])
        if not license_note:
            raise ValueError(
                'AI-TOD-v2 校验需要 --license-note 或 AITOD_LICENSE_NOTE。'
            )
        command.extend(['--license-note', license_note])
    return command


def validate(args: argparse.Namespace) -> None:
    repo = find_repo_root()
    run_command(validator_command(args, repo), repo)


def doctor(args: argparse.Namespace) -> None:
    repo = find_repo_root()
    if not args.skip_data:
        run_command(validator_command(args, repo), repo)
    command = [
        sys.executable,
        str(repo / 'scripts' / 'checkEnv.py'),
        '--dataset',
        args.dataset,
        '--strict-versions',
    ]
    if args.skip_data:
        command.append('--skip-data')
    if not args.allow_cpu:
        command.append('--require-gpu')
    run_command(command, repo)


def train(args: argparse.Namespace) -> None:
    repo = find_repo_root()
    spec = DATASETS[args.dataset]
    work_dir = args.work_dir or spec['work_dir']
    accumulation = args.accumulate or max(1, 4 // args.batch_size)
    command = [
        sys.executable,
        str(repo / 'tools' / 'train.py'),
        spec['config'],
        '--work-dir',
        work_dir,
    ]
    if args.amp:
        command.append('--amp')
    if args.resume:
        command.append('--resume')
    command.extend([
        '--cfg-options',
        f'train_dataloader.batch_size={args.batch_size}',
        f'train_dataloader.num_workers={args.workers}',
        f'train_dataloader.persistent_workers={args.workers > 0}',
        f'optim_wrapper.accumulative_counts={accumulation}',
    ])
    command.extend(args.cfg_option)
    run_command(command, repo)


def find_checkpoint(work_dir: Path) -> Path:
    candidates = [
        *work_dir.glob('best_*.pth'),
        *work_dir.glob('epoch_*.pth'),
    ]
    if not candidates:
        raise FileNotFoundError(f'未在 {work_dir} 找到 checkpoint')
    best = list(work_dir.glob('best_*.pth'))
    pool = best or candidates
    return max(pool, key=lambda path: (path.stat().st_mtime_ns, path.name))


def evaluate(args: argparse.Namespace) -> None:
    repo = find_repo_root()
    spec = DATASETS[args.dataset]
    train_work_dir = Path(args.work_dir or spec['work_dir'])
    if not train_work_dir.is_absolute():
        train_work_dir = repo / train_work_dir
    checkpoint = (
        find_checkpoint(train_work_dir)
        if args.checkpoint == 'auto'
        else Path(args.checkpoint).expanduser().resolve()
    )
    output_dir = args.output_dir or f'{train_work_dir}_test'
    predictions = args.out or str(train_work_dir / 'predictions.bbox.json')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint 不存在：{checkpoint}')
    command = [
        sys.executable,
        str(repo / 'tools' / 'test.py'),
        spec['config'],
        str(checkpoint),
        '--work-dir',
        output_dir,
        '--out',
        predictions,
    ]
    run_command(command, repo)


def info(_: argparse.Namespace) -> None:
    import mmcv
    import mmdet
    import mmengine
    import torch
    import torchvision

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'
    print(f'SpecLoc {__version__}')
    print(f'Python {sys.version.split()[0]}')
    print(f'PyTorch {torch.__version__}; TorchVision {torchvision.__version__}')
    print(f'MMCV {mmcv.__version__}; MMEngine {mmengine.__version__}; MMDet {mmdet.__version__}')
    print(f'CUDA runtime {torch.version.cuda}; GPU {gpu}')


def add_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('dataset', choices=tuple(DATASETS), nargs='?', default='rsod')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='specloc',
        description='SpecLoc 统一安装后命令：校验、检查、训练与评估。',
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    subparsers = parser.add_subparsers(dest='command', required=True)

    info_parser = subparsers.add_parser('info', help='显示依赖、CUDA 和 GPU 信息')
    info_parser.set_defaults(handler=info)

    validate_parser = subparsers.add_parser('validate', help='严格校验数据集')
    add_dataset_argument(validate_parser)
    validate_parser.add_argument('--source', help='AI-TOD-v2 授权来源')
    validate_parser.add_argument('--license-note', help='AI-TOD-v2 许可说明')
    validate_parser.set_defaults(handler=validate)

    doctor_parser = subparsers.add_parser('doctor', help='训练前完整验收')
    add_dataset_argument(doctor_parser)
    doctor_parser.add_argument('--skip-data', action='store_true', help='仅检查代码环境')
    doctor_parser.add_argument('--allow-cpu', action='store_true', help='不强制要求 CUDA')
    doctor_parser.add_argument('--source', help='AI-TOD-v2 授权来源')
    doctor_parser.add_argument('--license-note', help='AI-TOD-v2 许可说明')
    doctor_parser.set_defaults(handler=doctor)

    train_parser = subparsers.add_parser('train', help='训练标准基线')
    add_dataset_argument(train_parser)
    train_parser.add_argument('--batch-size', type=int, choices=(1, 2, 4), default=2)
    train_parser.add_argument('--workers', type=int, default=2)
    train_parser.add_argument('--accumulate', type=int, help='梯度累积步数')
    train_parser.add_argument('--work-dir', help='训练输出目录')
    train_parser.add_argument('--resume', action='store_true', help='自动断点续训')
    train_parser.add_argument(
        '--amp', action=argparse.BooleanOptionalAction, default=True,
        help='启用或关闭自动混合精度',
    )
    train_parser.add_argument(
        '--cfg-option', action='append', default=[], metavar='KEY=VALUE',
        help='追加 MMEngine 配置覆盖项，可重复传入',
    )
    train_parser.set_defaults(handler=train)

    evaluate_parser = subparsers.add_parser('evaluate', help='评估最佳或指定模型')
    add_dataset_argument(evaluate_parser)
    evaluate_parser.add_argument('--checkpoint', default='auto', help='路径或 auto')
    evaluate_parser.add_argument('--work-dir', help='训练工作目录')
    evaluate_parser.add_argument('--output-dir', help='评估输出目录')
    evaluate_parser.add_argument('--out', help='COCO 预测 JSON 路径')
    evaluate_parser.set_defaults(handler=evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
