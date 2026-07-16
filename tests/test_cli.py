import os
from pathlib import Path

from specloc import cli

REPO = Path(__file__).resolve().parents[1]


def _arguments(*values):
    return cli.build_parser().parse_args(values)


def test_train_defaults_hide_mmengine_override_details(monkeypatch):
    captured = []
    monkeypatch.setattr(cli, 'find_repo_root', lambda: REPO)
    monkeypatch.setattr(cli, 'run_command', lambda command, repo: captured.append(command))

    cli.train(_arguments('train', 'rsod'))

    command = captured[0]
    assert command[1].endswith('tools/train.py')
    assert 'configs/rsodCenternetR50.py' in command
    assert '--amp' in command
    assert 'train_dataloader.batch_size=2' in command
    assert 'train_dataloader.num_workers=2' in command
    assert 'optim_wrapper.accumulative_counts=2' in command


def test_doctor_runs_data_validator_before_environment_check(monkeypatch):
    captured = []
    monkeypatch.setattr(cli, 'find_repo_root', lambda: REPO)
    monkeypatch.setattr(cli, 'run_command', lambda command, repo: captured.append(command))

    cli.doctor(_arguments('doctor', 'rsod'))

    assert captured[0][1].endswith('scripts/validateRsod.py')
    assert captured[1][1].endswith('scripts/checkEnv.py')
    assert '--strict-versions' in captured[1]
    assert '--require-gpu' in captured[1]


def test_find_checkpoint_prefers_latest_best_checkpoint(tmp_path):
    old_best = tmp_path / 'best_old.pth'
    new_best = tmp_path / 'best_new.pth'
    epoch = tmp_path / 'epoch_12.pth'
    old_best.write_bytes(b'old')
    new_best.write_bytes(b'new')
    epoch.write_bytes(b'epoch')
    os.utime(old_best, ns=(1_000_000_000, 1_000_000_000))
    os.utime(new_best, ns=(2_000_000_000, 2_000_000_000))
    os.utime(epoch, ns=(3_000_000_000, 3_000_000_000))

    assert cli.find_checkpoint(tmp_path) == new_best


def test_requirements_pin_pkg_resources_compatible_setuptools():
    base = (REPO / 'requirements' / 'base.txt').read_text(encoding='utf-8')
    build = (REPO / 'pyproject.toml').read_text(encoding='utf-8')

    assert 'setuptools==80.10.2' in base
    assert 'setuptools>=69,<82' in build
