#!/usr/bin/env python3
"""Merge the four RSOD class folders and export audited COCO annotations."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

CATEGORY_NAMES = ('aircraft', 'oiltank', 'overpass', 'playground')
CATEGORY_IDS = {name: index for index, name in enumerate(CATEGORY_NAMES, start=1)}
SPLIT_NAMES = ('train', 'val', 'test')
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
SOURCE_URL = 'https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_label_file(path: Path) -> list[tuple[str, int, int, int, int]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f'{path}:{line_number}: expected 6 fields, found {len(fields)}')
        _, category, xmin, ymin, xmax, ymax = fields
        records.append((category, int(xmin), int(ymin), int(xmax), int(ymax)))
    return records


def _read_xml(
    path: Path,
    expected_category: str,
    image_size: tuple[int, int],
) -> tuple[list[dict], int]:
    root = ET.parse(path).getroot()
    width, height = image_size
    size = root.find('size')
    if size is None:
        raise ValueError(f'{path}: missing <size>')
    xml_size = (int(size.findtext('width', '-1')), int(size.findtext('height', '-1')))
    if xml_size != image_size:
        raise ValueError(f'{path}: XML size {xml_size} differs from image size {image_size}')

    annotations = []
    clipped_count = 0
    for object_index, obj in enumerate(root.findall('object'), 1):
        category = obj.findtext('name')
        if category != expected_category:
            raise ValueError(
                f'{path}: object {object_index} is {category!r}, expected '
                f'{expected_category!r}'
            )
        box = obj.find('bndbox')
        if box is None:
            raise ValueError(f'{path}: object {object_index} has no <bndbox>')
        raw = tuple(int(box.findtext(key, '')) for key in ('xmin', 'ymin', 'xmax', 'ymax'))
        xmin, ymin, xmax, ymax = raw
        clipped = (
            min(max(xmin, 0), width),
            min(max(ymin, 0), height),
            min(max(xmax, 0), width),
            min(max(ymax, 0), height),
        )
        xmin, ymin, xmax, ymax = clipped
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f'{path}: object {object_index} has invalid box {raw}')
        was_clipped = clipped != raw
        clipped_count += int(was_clipped)
        box_width = xmax - xmin
        box_height = ymax - ymin
        annotations.append({
            'category_name': category,
            'bbox': [xmin, ymin, box_width, box_height],
            'area': box_width * box_height,
            'raw_bbox_xyxy': list(raw),
            'bbox_clipped': was_clipped,
            'difficult': int(obj.findtext('difficult', '0')),
            'truncated': int(obj.findtext('truncated', '0')),
            'pose': obj.findtext('pose', 'Unspecified'),
        })
    return annotations, clipped_count


def load_records(source: Path) -> tuple[list[dict], dict]:
    source = source.resolve()
    records = []
    seen_names = set()
    missing_xml = []
    clipped_count = 0
    by_class = {}

    for category in CATEGORY_NAMES:
        image_dir = source / category / 'JPEGImages'
        xml_dir = source / category / 'Annotation' / 'xml'
        label_dir = source / category / 'Annotation' / 'labels'
        for required in (image_dir, xml_dir, label_dir):
            if not required.is_dir():
                raise FileNotFoundError(f'missing RSOD directory: {required}')

        image_paths = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        )
        class_annotations = 0
        class_missing_xml = 0
        for image_path in image_paths:
            if image_path.name in seen_names:
                raise ValueError(f'duplicate image filename across categories: {image_path.name}')
            seen_names.add(image_path.name)
            with Image.open(image_path) as image:
                image.load()
                image_size = image.size

            xml_path = xml_dir / f'{image_path.stem}.xml'
            label_path = label_dir / f'{image_path.stem}.txt'
            if xml_path.is_file():
                annotations, clipped = _read_xml(xml_path, category, image_size)
                if not label_path.is_file():
                    raise FileNotFoundError(f'XML exists but label file is missing: {label_path}')
                label_records = _read_label_file(label_path)
                xml_records = [
                    (item['category_name'], *item['raw_bbox_xyxy']) for item in annotations
                ]
                if label_records != xml_records:
                    raise ValueError(f'XML and TXT annotations differ: {image_path.name}')
                clipped_count += clipped
            else:
                if label_path.exists():
                    raise ValueError(f'label exists without XML: {label_path}')
                annotations = []
                missing_xml.append(image_path.name)
                class_missing_xml += 1

            class_annotations += len(annotations)
            records.append({
                'source_class': category,
                'source_path': image_path,
                'file_name': image_path.name,
                'width': image_size[0],
                'height': image_size[1],
                'sha256': sha256(image_path),
                'annotations': annotations,
                'missing_xml': not xml_path.is_file(),
            })
        by_class[category] = {
            'images': len(image_paths),
            'annotations': class_annotations,
            'images_without_xml': class_missing_xml,
        }

    digest_groups = defaultdict(list)
    for record in records:
        digest_groups[record['sha256']].append(record)
    duplicate_groups = {
        digest: sorted(item['file_name'] for item in group)
        for digest, group in digest_groups.items()
        if len(group) > 1
    }
    for digest, group in digest_groups.items():
        categories = {item['source_class'] for item in group}
        if len(categories) > 1:
            raise ValueError(
                f'identical image {digest} occurs under multiple source classes: '
                f'{sorted(categories)}'
            )

    audit = {
        'by_class': by_class,
        'missing_xml_images': sorted(missing_xml),
        'bbox_clipped_count': clipped_count,
        'duplicate_groups': duplicate_groups,
    }
    return records, audit


def _target_counts(total: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remaining]:
        counts[index] += 1
    return dict(zip(SPLIT_NAMES, counts))


def split_records(
    records: list[dict],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 2026,
) -> dict[str, list[dict]]:
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios):
        raise ValueError('split ratios must contain three positive values')
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f'split ratios must sum to 1.0, found {sum(ratios):.12f}')

    output = {name: [] for name in SPLIT_NAMES}
    for category in CATEGORY_NAMES:
        class_records = [item for item in records if item['source_class'] == category]
        digest_groups = defaultdict(list)
        for record in class_records:
            digest_groups[record['sha256']].append(record)
        groups = [sorted(group, key=lambda item: item['file_name']) for group in digest_groups.values()]
        groups.sort(key=lambda group: group[0]['sha256'])
        random.Random(f'{seed}:{category}').shuffle(groups)

        targets = _target_counts(len(class_records), ratios)
        assigned = Counter()
        for group in groups:
            size = len(group)
            fitting = [name for name in SPLIT_NAMES if targets[name] - assigned[name] >= size]
            candidates = fitting or list(SPLIT_NAMES)
            selected = max(
                candidates,
                key=lambda name: (targets[name] - assigned[name], -SPLIT_NAMES.index(name)),
            )
            output[selected].extend(group)
            assigned[selected] += size

    digest_to_split = {}
    for split, items in output.items():
        for item in items:
            previous = digest_to_split.setdefault(item['sha256'], split)
            if previous != split:
                raise AssertionError(f'image digest crosses splits: {item["sha256"]}')
        items.sort(key=lambda item: item['file_name'])
    return output


def _coco_payload(records: list[dict], image_ids: dict[str, int], annotation_ids: dict) -> dict:
    images = []
    annotations = []
    for record in records:
        image_id = image_ids[record['file_name']]
        images.append({
            'id': image_id,
            'file_name': record['file_name'],
            'width': record['width'],
            'height': record['height'],
            'source_class': record['source_class'],
            'sha256': record['sha256'],
            'has_annotation': bool(record['annotations']),
        })
        for object_index, item in enumerate(record['annotations']):
            annotation = {
                'id': annotation_ids[(record['file_name'], object_index)],
                'image_id': image_id,
                'category_id': CATEGORY_IDS[item['category_name']],
                'bbox': item['bbox'],
                'area': item['area'],
                'iscrowd': 0,
                'segmentation': [],
                'raw_bbox_xyxy': item['raw_bbox_xyxy'],
                'bbox_clipped': item['bbox_clipped'],
                'difficult': item['difficult'],
                'truncated': item['truncated'],
                'pose': item['pose'],
            }
            annotations.append(annotation)
    return {
        'info': {
            'description': 'RSOD merged and converted from the four official class folders',
            'source': SOURCE_URL,
            'coordinate_conversion': (
                'Raw coordinates are treated as zero-based XYXY border coordinates; '
                'out-of-image values are clipped before COCO XYWH conversion.'
            ),
        },
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': [
            {'id': CATEGORY_IDS[name], 'name': name, 'supercategory': 'remote-sensing-object'}
            for name in CATEGORY_NAMES
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _copy_image(source: Path, destination: Path, mode: str) -> None:
    if mode == 'copy':
        shutil.copy2(source, destination)
    elif mode == 'hardlink':
        os.link(source, destination)
    else:
        raise ValueError(f'unknown copy mode: {mode}')


def _readme_text(manifest: dict) -> str:
    totals = manifest['totals']
    split_lines = '\n'.join(
        f"- `{name}`：{item['images']} 张图像，{item['annotations']} 个标注"
        for name, item in manifest['splits'].items()
    )
    return f"""# RSOD 合并数据集

本目录由 `scripts/prepareRsod.py` 从 RSOD 四个类别目录可重复生成。

- 图像总数：{totals['images']}
- 标注总数：{totals['annotations']}
- 类别：{', '.join(CATEGORY_NAMES)}
- 固定随机种子：{manifest['split_policy']['seed']}
- 划分比例：{manifest['split_policy']['ratios']}

{split_lines}

## 目录

```text
RSOD_merged/
├── images/
├── annotations/
│   ├── instances_all.json
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
├── splits/
├── images.sha256
├── manifest.json
└── README.md
```

没有 XML 的图像按无目标负样本保留。完全相同的图像被锁定在同一个划分中，
避免训练集与验证/测试集发生逐字节重复泄漏。原始越界框仅在 COCO 导出结果中裁剪，
原坐标保存在每条标注的 `raw_bbox_xyxy` 字段中。

上游仓库未明确附带数据集许可文件。上传或重新分发前，请自行核实使用条款：
{SOURCE_URL}
"""


def prepare_dataset(
    source: Path,
    output: Path,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 2026,
    copy_mode: str = 'copy',
    overwrite: bool = False,
    license_note: str = '上游仓库未附明确许可文件；重新分发前须自行核实使用条款。',
) -> dict:
    source = source.resolve()
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f'output already exists: {output}; pass --overwrite to replace it')

    records, audit = load_records(source)
    split_map = split_records(records, ratios=ratios, seed=seed)
    ordered = sorted(records, key=lambda item: item['file_name'])
    image_ids = {item['file_name']: index for index, item in enumerate(ordered, start=1)}
    annotation_ids = {}
    next_annotation_id = 1
    for record in ordered:
        for object_index in range(len(record['annotations'])):
            annotation_ids[(record['file_name'], object_index)] = next_annotation_id
            next_annotation_id += 1

    temporary = output.parent / f'.{output.name}.tmp-{os.getpid()}'
    if temporary.exists():
        shutil.rmtree(temporary)
    image_dir = temporary / 'images'
    annotation_dir = temporary / 'annotations'
    split_dir = temporary / 'splits'
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    split_dir.mkdir()

    try:
        for record in ordered:
            _copy_image(record['source_path'], image_dir / record['file_name'], copy_mode)

        payloads = {'all': _coco_payload(ordered, image_ids, annotation_ids)}
        payloads.update({
            split: _coco_payload(items, image_ids, annotation_ids)
            for split, items in split_map.items()
        })
        for split, payload in payloads.items():
            _write_json(annotation_dir / f'instances_{split}.json', payload)
        for split, items in split_map.items():
            names = ''.join(f'{item["file_name"]}\n' for item in items)
            (split_dir / f'{split}.txt').write_text(names, encoding='utf-8')

        checksum_lines = ''.join(
            f'{record["sha256"]}  images/{record["file_name"]}\n' for record in ordered
        )
        (temporary / 'images.sha256').write_text(checksum_lines, encoding='utf-8')

        manifest = {
            'dataset': 'RSOD',
            'status': 'verified',
            'prepared_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'source_directory': str(source),
            'source_url': SOURCE_URL,
            'license_note': license_note,
            'copy_mode': copy_mode,
            'categories': list(CATEGORY_NAMES),
            'coordinate_policy': {
                'raw': 'zero-based XYXY border coordinates',
                'export': 'COCO XYWH',
                'out_of_bounds': 'clipped to image bounds; raw_bbox_xyxy retained',
            },
            'split_policy': {
                'seed': seed,
                'ratios': dict(zip(SPLIT_NAMES, ratios)),
                'stratified_by': 'source class',
                'duplicate_policy': 'byte-identical images remain in the same split',
            },
            'totals': {
                'images': len(ordered),
                'annotations': sum(len(item['annotations']) for item in ordered),
            },
            'by_class': audit['by_class'],
            'quality': {
                'images_without_xml': len(audit['missing_xml_images']),
                'images_without_xml_names': audit['missing_xml_images'],
                'bbox_clipped_count': audit['bbox_clipped_count'],
                'exact_duplicate_groups': len(audit['duplicate_groups']),
                'exact_duplicate_images': sum(len(v) for v in audit['duplicate_groups'].values()),
                'exact_duplicate_group_members': audit['duplicate_groups'],
            },
            'splits': {
                split: {
                    'images': len(items),
                    'annotations': sum(len(item['annotations']) for item in items),
                    'by_source_class': dict(sorted(Counter(
                        item['source_class'] for item in items
                    ).items())),
                }
                for split, items in split_map.items()
            },
        }
        _write_json(temporary / 'manifest.json', manifest)
        (temporary / 'README.md').write_text(_readme_text(manifest), encoding='utf-8')

        if output.exists():
            shutil.rmtree(output)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='RSOD four-folder source')
    parser.add_argument('--output', type=Path, required=True, help='merged output directory')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument(
        '--ratios',
        type=float,
        nargs=3,
        metavar=('TRAIN', 'VAL', 'TEST'),
        default=DEFAULT_RATIOS,
    )
    parser.add_argument('--copy-mode', choices=('copy', 'hardlink'), default='copy')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--license-note',
        default='上游仓库未附明确许可文件；重新分发前须自行核实使用条款。',
    )
    args = parser.parse_args()
    manifest = prepare_dataset(
        source=args.source,
        output=args.output,
        ratios=tuple(args.ratios),
        seed=args.seed,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
        license_note=args.license_note,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
