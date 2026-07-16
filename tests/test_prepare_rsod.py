import json

import pytest
from PIL import Image

from scripts import prepareRsod


def _write_xml(path, image_name, category, size, boxes):
    width, height = size
    objects = ''.join(
        f'''<object><name>{category}</name><pose>Left</pose><truncated>0</truncated>
        <difficult>0</difficult><bndbox><xmin>{box[0]}</xmin><ymin>{box[1]}</ymin>
        <xmax>{box[2]}</xmax><ymax>{box[3]}</ymax></bndbox></object>'''
        for box in boxes
    )
    path.write_text(
        f'''<annotation><filename>{image_name}</filename><size><width>{width}</width>
        <height>{height}</height><depth>3</depth></size>{objects}</annotation>'''
    )


def _add_image(root, category, name, color, boxes=None):
    image_dir = root / category / 'JPEGImages'
    xml_dir = root / category / 'Annotation' / 'xml'
    label_dir = root / category / 'Annotation' / 'labels'
    image_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / name
    Image.new('RGB', (10, 8), color=color).save(image_path)
    if boxes is None:
        return
    _write_xml(xml_dir / f'{image_path.stem}.xml', name, category, (10, 8), boxes)
    lines = ''.join(
        f'{name}\t{category}\t{box[0]}\t{box[1]}\t{box[2]}\t{box[3]}\n'
        for box in boxes
    )
    (label_dir / f'{image_path.stem}.txt').write_text(lines)


def _sample_source(root):
    _add_image(root, 'aircraft', 'aircraft_1.jpg', 'red', [(-1, 1, 5, 7)])
    _add_image(root, 'aircraft', 'aircraft_2.jpg', 'red', [(0, 1, 5, 7)])
    _add_image(root, 'oiltank', 'oiltank_1.jpg', 'green', [(1, 1, 6, 6)])
    _add_image(root, 'overpass', 'overpass_1.jpg', 'blue', [(2, 2, 9, 7)])
    _add_image(root, 'playground', 'playground_1.jpg', 'yellow', None)


def test_prepare_dataset_merges_and_prevents_duplicate_split_leakage(tmp_path):
    source = tmp_path / 'source'
    output = tmp_path / 'merged'
    _sample_source(source)

    manifest = prepareRsod.prepare_dataset(
        source,
        output,
        ratios=(0.50, 0.25, 0.25),
        seed=7,
    )

    assert manifest['status'] == 'verified'
    assert manifest['totals'] == {'images': 5, 'annotations': 4}
    assert manifest['quality']['images_without_xml'] == 1
    assert manifest['quality']['bbox_clipped_count'] == 1
    assert manifest['quality']['exact_duplicate_groups'] == 1
    assert len(list((output / 'images').glob('*.jpg'))) == 5

    memberships = {}
    for split in prepareRsod.SPLIT_NAMES:
        names = set((output / 'splits' / f'{split}.txt').read_text().splitlines())
        for name in names:
            memberships[name] = split
    assert memberships['aircraft_1.jpg'] == memberships['aircraft_2.jpg']

    all_payload = json.loads((output / 'annotations' / 'instances_all.json').read_text())
    assert len(all_payload['images']) == 5
    assert len(all_payload['annotations']) == 4
    clipped = [item for item in all_payload['annotations'] if item['bbox_clipped']]
    assert clipped[0]['bbox'] == [0, 1, 5, 6]
    assert clipped[0]['raw_bbox_xyxy'] == [-1, 1, 5, 7]


def test_prepare_dataset_rejects_xml_label_disagreement(tmp_path):
    source = tmp_path / 'source'
    _sample_source(source)
    label = source / 'oiltank' / 'Annotation' / 'labels' / 'oiltank_1.txt'
    label.write_text('oiltank_1.jpg\toiltank\t1\t1\t7\t6\n')

    with pytest.raises(ValueError, match='XML and TXT annotations differ'):
        prepareRsod.load_records(source)
