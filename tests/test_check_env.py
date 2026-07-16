from scripts import checkEnv


def test_cloud_version_contract_matches_environment_file():
    assert checkEnv.VERSION_CONTRACT == {
        'setuptools': '80.10.2',
        'torch': '2.1.2',
        'torchvision': '0.16.2',
        'mmcv': '2.1.0',
        'mmdet': '3.3.0',
        'numpy': '1.26.4',
    }
