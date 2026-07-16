import json
from types import SimpleNamespace

import pytest
from mmengine.config import Config
from torch import nn

from specloc.engine.hooks.experimentMetadataHook import ExperimentMetadataHook


def test_metadata_hook_writes_model_seed_and_verified_manifest(tmp_path):
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(dict(
        dataset='AI-TOD-v2',
        status='verified',
    )))
    runner = SimpleNamespace(
        work_dir=str(tmp_path),
        model=nn.Linear(4, 2),
        cfg=Config(dict(
            data_version='AI-TOD-v2',
            data_manifest=str(manifest_path),
            require_verified_data=True,
            randomness=dict(seed=42),
        )),
    )
    ExperimentMetadataHook().before_run(runner)
    metadata = json.loads((tmp_path / 'experiment_metadata.json').read_text())
    assert metadata['project'] == 'SpecLoc'
    assert metadata['project_version'] == '0.8.0'
    assert metadata['model_type'] == 'Linear'
    assert metadata['randomness']['seed'] == 42
    assert metadata['data_manifest']['status'] == 'verified'


def test_formal_run_rejects_missing_verified_manifest(tmp_path):
    runner = SimpleNamespace(
        work_dir=str(tmp_path),
        model=nn.Identity(),
        cfg=Config(dict(
            require_verified_data=True,
            data_manifest=str(tmp_path / 'missing.json'),
        )),
    )
    with pytest.raises(RuntimeError, match='manifest'):
        ExperimentMetadataHook().before_run(runner)
    metadata = json.loads((tmp_path / 'experiment_metadata.json').read_text())
    assert metadata['data_manifest']['status'] == 'UNVERIFIED'
