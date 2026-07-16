#!/usr/bin/env python3
"""Validate a merged RSOD copy before a cloud training run."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from PIL import Image

CATEGORY_NAMES = ('aircraft', 'oiltank', 'overpass', 'playground')
SPLIT_NAMES = ('train', 'val', 'test')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, errors: list[str]) -> dict | None:
    if not path.is_file():
        errors.append(f'missing JSON: {path}')
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f'invalid JSON {path}: {exc}')
        return None
    if not isinstance(payload, dict):
        errors.append(f'JSON root must be an object: {path}')
        return None
    return payload


def _category_names(payload: dict) -> tuple[str, ...]:
    categories = sorted(payload.get('categories', []), key=lambda item: item.get('id', -1))
    return tuple(item.get('name') for item in categories)


def _validate_payload(
    root: Path,
    label: str,
    payload: dict,
    errors: list[str],
    inspect_files: bool,
) -> dict:
    images = payload.get('images', [])
    annotations = payload.get('annotations', [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        errors.append(f'{label}: images and annotations must be lists')
        return {'images': {}, 'annotations': {}}
    if _category_names(payload) != CATEGORY_NAMES:
        errors.append(
            f'{label}: expected categories {CATEGORY_NAMES}, found {_category_names(payload)}'
        )

    image_by_id = {}
    image_by_name = {}
    for image in images:
        image_id = image.get('id')
        file_name = image.get('file_name')
        if image_id in image_by_id:
            errors.append(f'{label}: duplicate image id {image_id}')
            continue
        if not isinstance(file_name, str) or Path(file_name).name != file_name:
            errors.append(f'{label}: unsafe or invalid image name {file_name!r}')
            continue
        if file_name in image_by_name:
            errors.append(f'{label}: duplicate image name {file_name}')
            continue
        image_by_id[image_id] = image
        image_by_name[file_name] = image
        if inspect_files:
            image_path = root / 'images' / file_name
            if not image_path.is_file():
                errors.append(f'{label}: missing image {image_path}')
                continue
            try:
                with Image.open(image_path) as opened:
                    opened.load()
                    actual_size = opened.size
            except Exception as exc:
                errors.append(f'{label}: unreadable image {image_path}: {exc}')
                continue
            declared_size = (image.get('width'), image.get('height'))
            if declared_size != actual_size:
                errors.append(
                    f'{label}: {file_name} declares {declared_size}, actual {actual_size}'
                )

    annotation_by_id = {}
    valid_category_ids = set(range(1, len(CATEGORY_NAMES) + 1))
    for annotation in annotations:
        annotation_id = annotation.get('id')
        if annotation_id in annotation_by_id:
            errors.append(f'{label}: duplicate annotation id {annotation_id}')
            continue
        annotation_by_id[annotation_id] = annotation
        image = image_by_id.get(annotation.get('image_id'))
        if image is None:
            errors.append(
                f'{label}: annotation {annotation_id} references unknown image '
                f'{annotation.get("image_id")}'
            )
            continue
        if annotation.get('category_id') not in valid_category_ids:
            errors.append(
                f'{label}: annotation {annotation_id} has invalid category '
                f'{annotation.get("category_id")}'
            )
        bbox = annotation.get('bbox')
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f'{label}: annotation {annotation_id} has invalid bbox {bbox!r}')
            continue
        x, y, width, height = bbox
        if not all(isinstance(value, (int, float)) for value in bbox):
            errors.append(f'{label}: annotation {annotation_id} bbox is not numeric')
            continue
        if (
            x < 0 or y < 0 or width <= 0 or height <= 0
            or x + width > image.get('width', -1)
            or y + height > image.get('height', -1)
        ):
            errors.append(
                f'{label}: annotation {annotation_id} bbox {bbox} exceeds '
                f'{image.get("file_name")} bounds'
            )
    return {'images': image_by_name, 'annotations': annotation_by_id}


def _validate_checksums(root: Path, expected_names: set[str], errors: list[str]) -> int:
    checksum_path = root / 'images.sha256'
    if not checksum_path.is_file():
        errors.append(f'missing checksum file: {checksum_path}')
        return 0
    declared = {}
    for line_number, line in enumerate(checksum_path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            errors.append(f'{checksum_path}:{line_number}: malformed checksum line')
            continue
        digest, relative = fields
        if relative.startswith('*'):
            relative = relative[1:]
        if relative.startswith('images/'):
            relative = relative[len('images/'):]
        if Path(relative).name != relative:
            errors.append(f'{checksum_path}:{line_number}: unsafe path {relative!r}')
            continue
        if relative in declared:
            errors.append(f'{checksum_path}:{line_number}: duplicate name {relative}')
            continue
        declared[relative] = digest
    if set(declared) != expected_names:
        missing = sorted(expected_names - set(declared))
        extra = sorted(set(declared) - expected_names)
        errors.append(
            f'checksum/image mismatch: missing={missing[:5]}, extra={extra[:5]}'
        )
    checked = 0
    for name in sorted(expected_names & set(declared)):
        image_path = root / 'images' / name
        if image_path.is_file():
            actual = sha256(image_path)
            checked += 1
            if actual != declared[name]:
                errors.append(f'checksum mismatch: {image_path}')
    return checked


def validate(root: Path) -> dict:
    root = root.resolve()
    errors = []
    manifest = _load_json(root / 'manifest.json', errors)
    payloads = {
        label: _load_json(root / 'annotations' / f'instances_{label}.json', errors)
        for label in ('all', *SPLIT_NAMES)
    }
    if manifest is None or any(payload is None for payload in payloads.values()):
        return {
            'dataset': 'RSOD',
            'status': 'invalid',
            'validated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'root': str(root),
            'errors': errors,
        }

    if manifest.get('dataset') != 'RSOD' or manifest.get('status') != 'verified':
        errors.append(
            f'manifest must identify verified RSOD, found '
            f'{manifest.get("dataset")!r}/{manifest.get("status")!r}'
        )

    validated = {
        'all': _validate_payload(root, 'all', payloads['all'], errors, inspect_files=True)
    }
    all_images = validated['all']['images']
    all_annotations = validated['all']['annotations']
    split_names = {}
    for split in SPLIT_NAMES:
        validated[split] = _validate_payload(
            root, split, payloads[split], errors, inspect_files=False
        )
        split_names[split] = set(validated[split]['images'])
        if not split_names[split] <= set(all_images):
            errors.append(f'{split}: contains images absent from instances_all.json')
        for name, image in validated[split]['images'].items():
            reference = all_images.get(name)
            if reference is not None and image != reference:
                errors.append(f'{split}: image metadata differs from all: {name}')
        for annotation_id, annotation in validated[split]['annotations'].items():
            reference = all_annotations.get(annotation_id)
            if reference is None:
                errors.append(f'{split}: annotation {annotation_id} absent from all')
            elif annotation != reference:
                errors.append(f'{split}: annotation {annotation_id} differs from all')

    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1:]:
            overlap = split_names[left] & split_names[right]
            if overlap:
                errors.append(f'{left}/{right}: {len(overlap)} overlapping images')
    split_union = set().union(*(split_names[name] for name in SPLIT_NAMES))
    if split_union != set(all_images):
        errors.append(
            f'split union has {len(split_union)} images, all has {len(all_images)}'
        )

    checksum_count = _validate_checksums(root, set(all_images), errors)
    totals = manifest.get('totals', {})
    if totals.get('images') != len(all_images):
        errors.append('manifest image total differs from instances_all.json')
    if totals.get('annotations') != len(all_annotations):
        errors.append('manifest annotation total differs from instances_all.json')

    duplicate_groups = (
        manifest.get('quality', {}).get('exact_duplicate_group_members', {})
    )
    for members in duplicate_groups.values():
        occupied = {
            split for split, names in split_names.items() if set(members) & names
        }
        if len(occupied) != 1:
            errors.append(f'duplicate group crosses splits: {members}')

    return {
        'dataset': 'RSOD',
        'status': 'verified' if not errors else 'invalid',
        'validated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'root': str(root),
        'images': len(all_images),
        'annotations': len(all_annotations),
        'checksums_verified': checksum_count,
        'split_images': {name: len(values) for name, values in split_names.items()},
        'duplicate_groups_checked': len(duplicate_groups),
        'errors': errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/RSOD'))
    args = parser.parse_args()
    report = validate(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'verified' else 2


if __name__ == '__main__':
    raise SystemExit(main())
