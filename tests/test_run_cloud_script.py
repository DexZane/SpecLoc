import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts' / 'runCloudExperiment.sh'


def _dry_run(dataset=None):
    command = ['bash', str(SCRIPT)]
    if dataset is not None:
        command.extend(['--dataset', dataset])
    command.append('--dry-run')
    return subprocess.run(
        command,
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    ).stdout


def test_rsod_is_cloud_script_default_and_uses_specloc_environment():
    output = _dry_run()

    assert '数据集        : rsod' in output
    assert 'configs/rsodCenternetR50.py' in output
    assert 'data/RSOD' in output
    assert 'Conda 环境    : specloc' in output


def test_aitod_remains_selectable():
    output = _dry_run('aitod')

    assert '数据集        : aitod' in output
    assert 'configs/aitodCenternetR50.py' in output
    assert 'data/AI-TOD' in output


def test_cloud_script_enforces_the_pinned_version_contract():
    source = SCRIPT.read_text(encoding='utf-8')

    assert '--strict-versions' in source
