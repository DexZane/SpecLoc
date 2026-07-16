import hashlib
import json

from PIL import Image

from scripts import validateRsod


def _write_dataset(root):
    image_dir = root / 'images'
    annotation_dir = root / 'annotations'
    split_dir = root / 'splits'
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    split_dir.mkdir()

    categories = [
        dict(id=index, name=name)
        for index, name in enumerate(validateRsod.CATEGORY_NAMES, start=1)
    ]
    all_images = []
    all_annotations = []
    checksums = []
    for index, split in enumerate(validateRsod.SPLIT_NAMES, start=1):
        name = f'{split}.jpg'
        image_path = image_dir / name
        Image.new('RGB', (10, 8), color=(index * 30, 0, 0)).save(image_path)
        image = dict(id=index, file_name=name, width=10, height=8)
        annotation = dict(
            id=index,
            image_id=index,
            category_id=index,
            bbox=[1, 1, 5, 4],
            area=20,
            iscrowd=0,
        )
        all_images.append(image)
        all_annotations.append(annotation)
        payload = dict(images=[image], annotations=[annotation], categories=categories)
        (annotation_dir / f'instances_{split}.json').write_text(json.dumps(payload))
        (split_dir / f'{split}.txt').write_text(f'{name}\n')
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        checksums.append(f'{digest}  images/{name}\n')

    (annotation_dir / 'instances_all.json').write_text(json.dumps(dict(
        images=all_images,
        annotations=all_annotations,
        categories=categories,
    )))
    (root / 'images.sha256').write_text(''.join(checksums))
    (root / 'manifest.json').write_text(json.dumps(dict(
        dataset='RSOD',
        status='verified',
        totals=dict(images=3, annotations=3),
        quality=dict(exact_duplicate_group_members={}),
    )))


def test_validate_accepts_complete_merged_dataset(tmp_path):
    _write_dataset(tmp_path)

    report = validateRsod.validate(tmp_path)

    assert report['status'] == 'verified'
    assert report['images'] == 3
    assert report['annotations'] == 3
    assert report['checksums_verified'] == 3
    assert report['errors'] == []


def test_validate_rejects_checksum_corruption(tmp_path):
    _write_dataset(tmp_path)
    (tmp_path / 'images' / 'train.jpg').write_bytes(b'corrupted')

    report = validateRsod.validate(tmp_path)

    assert report['status'] == 'invalid'
    assert any('checksum mismatch' in error for error in report['errors'])
