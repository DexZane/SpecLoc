import json

import numpy as np
import pytest

pytest.importorskip('mmdet')

from specloc.evaluation import AITODMetric


def test_official_area_partitions_and_perfect_predictions(tmp_path):
    boxes_xywh = [
        [5, 5, 4, 4],
        [20, 5, 12, 12],
        [40, 5, 24, 24],
        [70, 5, 40, 40],
    ]
    annotations = []
    predictions = []
    for index, (x, y, width, height) in enumerate(boxes_xywh, start=1):
        annotations.append(dict(
            id=index,
            image_id=1,
            category_id=1,
            bbox=[x, y, width, height],
            area=width * height,
            iscrowd=0,
        ))
        predictions.append([x, y, x + width, y + height])

    ann_file = tmp_path / 'annotations.json'
    ann_file.write_text(json.dumps(dict(
        images=[dict(id=1, width=128, height=64, file_name='sample.png')],
        annotations=annotations,
        categories=[dict(id=1, name='object')],
    )))

    metric = AITODMetric(ann_file=str(ann_file), metric='bbox')
    metric.dataset_meta = {'classes': ('object',)}
    result = metric.compute_metrics([({}, dict(
        img_id=1,
        bboxes=np.asarray(predictions, dtype=np.float32),
        scores=np.ones(4, dtype=np.float32),
        labels=np.zeros(4, dtype=np.int64),
    ))])

    for key in (
        'bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75',
        'bbox_mAP_vt', 'bbox_mAP_t', 'bbox_mAP_s', 'bbox_mAP_m',
    ):
        assert result[key] == pytest.approx(1.0), key


def test_ap25_only_protocol_is_rejected():
    with pytest.raises(ValueError, match='standard COCO IoU thresholds'):
        AITODMetric(iou_thrs=[0.25, 0.5])
