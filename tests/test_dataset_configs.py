from mmengine.config import Config


def test_rsod_config_uses_four_classes_and_distinct_validation_and_test_sets():
    cfg = Config.fromfile('configs/rsodCenternetR50.py')

    assert cfg.custom_imports.imports == ['specloc.registry']
    assert cfg.model.bbox_head.num_classes == 4
    assert cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt is False
    assert cfg.train_dataloader.dataset.ann_file == 'annotations/instances_train.json'
    assert cfg.val_dataloader.dataset.ann_file == 'annotations/instances_val.json'
    assert cfg.test_dataloader.dataset.ann_file == 'annotations/instances_test.json'
    assert cfg.val_evaluator.ann_file.endswith('instances_val.json')
    assert cfg.test_evaluator.ann_file.endswith('instances_test.json')
    assert cfg.data_manifest == 'data/RSOD/manifest.json'


def test_aitod_config_imports_renamed_package():
    cfg = Config.fromfile('configs/aitodCenternetR50.py')

    assert cfg.custom_imports.imports == ['specloc.registry']
    assert cfg.model.bbox_head.num_classes == 8
