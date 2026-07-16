import json

from scripts import validateAitod


def _categories():
    return [
        dict(id=index, name=name)
        for index, name in enumerate(sorted(validateAitod.EXPECTED_CLASSES), start=1)
    ]


def _write_split(root, split, image_name='sample.png', include_image=True):
    image_dir = root / split
    image_dir.mkdir(parents=True, exist_ok=True)
    if include_image:
        (image_dir / image_name).write_bytes(b'image-placeholder')
    annotation_dir = root / 'annotations'
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (annotation_dir / f'aitod_{split}_v2.json').write_text(json.dumps(dict(
        images=[dict(id=1, file_name=image_name)],
        annotations=[],
        categories=_categories(),
    )))


def test_build_manifest_accepts_complete_authorized_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(validateAitod, 'SPLITS', {
        'train': ('annotations/aitod_train_v2.json', 'train', 1),
        'val': ('annotations/aitod_val_v2.json', 'val', 1),
    })
    _write_split(tmp_path, 'train')
    _write_split(tmp_path, 'val')
    manifest = validateAitod.build_manifest(
        tmp_path,
        source='authorized-source',
        license_note='terms verified',
    )
    assert manifest['status'] == 'verified'
    assert manifest['splits']['train']['annotation_sha256']
    assert manifest['splits']['val']['missing_image_count'] == 0


def test_build_manifest_rejects_missing_image(tmp_path, monkeypatch):
    monkeypatch.setattr(validateAitod, 'SPLITS', {
        'val': ('annotations/aitod_val_v2.json', 'val', 1),
    })
    _write_split(tmp_path, 'val', include_image=False)
    manifest = validateAitod.build_manifest(
        tmp_path,
        source='authorized-source',
        license_note='terms verified',
    )
    assert manifest['status'] == 'invalid'
    assert manifest['splits']['val']['missing_image_count'] == 1
