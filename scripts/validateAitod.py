#!/usr/bin/env python3
"""Validate an AI-TOD-v2 installation and write a provenance manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

EXPECTED_CLASSES = {
    'airplane', 'bridge', 'storage-tank', 'ship', 'swimming-pool',
    'vehicle', 'person', 'wind-mill',
}
SPLITS = {
    'train': ('annotations/aitod_train_v2.json', 'train', 11214),
    'val': ('annotations/aitod_val_v2.json', 'val', 5607),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split(root: Path, split: str) -> dict:
    annotation_rel, image_rel, expected_images = SPLITS[split]
    annotation_path = root / annotation_rel
    image_root = root / image_rel
    record = {
        'annotation_file': annotation_rel,
        'image_directory': image_rel,
        'expected_images': expected_images,
        'errors': [],
    }
    if not annotation_path.is_file():
        record['errors'].append(f'missing annotation: {annotation_path}')
        return record
    if not image_root.is_dir():
        record['errors'].append(f'missing image directory: {image_root}')

    try:
        payload = json.loads(annotation_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        record['errors'].append(f'invalid JSON: {exc}')
        return record

    images = payload.get('images', [])
    annotations = payload.get('annotations', [])
    categories = payload.get('categories', [])
    class_names = {category.get('name') for category in categories}
    record.update({
        'annotation_sha256': sha256(annotation_path),
        'images_in_json': len(images),
        'annotations_in_json': len(annotations),
        'categories': sorted(name for name in class_names if name),
    })
    if len(images) != expected_images:
        record['errors'].append(
            f'expected {expected_images} images in {split}, found {len(images)}'
        )
    if class_names != EXPECTED_CLASSES:
        record['errors'].append(
            'category set differs from the official eight AI-TOD-v2 classes'
        )

    if image_root.is_dir():
        missing = [
            image.get('file_name', '') for image in images
            if image.get('file_name') and not (image_root / image['file_name']).is_file()
        ]
        record['missing_image_count'] = len(missing)
        record['missing_image_examples'] = missing[:10]
        if missing:
            record['errors'].append(
                f'{len(missing)} annotation image paths do not exist'
            )
    return record


def build_manifest(root: Path, source: str, license_note: str) -> dict:
    splits = {name: validate_split(root, name) for name in SPLITS}
    valid = all(not record['errors'] for record in splits.values())
    return {
        'dataset': 'AI-TOD-v2',
        'status': 'verified' if valid else 'invalid',
        'validated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'root': str(root.resolve()),
        'source': source,
        'license_note': license_note,
        'splits': splits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/AI-TOD'))
    parser.add_argument(
        '--source',
        default='https://github.com/Chasel-Tsui/mmdet-aitod',
        help='Exact authorized download/source URL used for this copy',
    )
    parser.add_argument(
        '--license-note',
        required=True,
        help='Locally verified dataset license or access terms',
    )
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    manifest = build_manifest(args.root, args.source, args.license_note)
    output = args.output or args.root / 'manifest.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest['status'] == 'verified' else 2


if __name__ == '__main__':
    raise SystemExit(main())
