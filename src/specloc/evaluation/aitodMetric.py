# Copyright (c) 2026 SpecLoc Authors
# AI-TOD area partitions are derived from the official cocoapi-aitod toolkit.
# See docs/THIRD_PARTY_NOTICES.md for the retained upstream notice.
# SPDX-License-Identifier: Apache-2.0
"""AI-TOD-v2 evaluation with the official object-size partitions."""

from __future__ import annotations

import os.path as osp
import tempfile
from typing import Dict

import numpy as np
from mmdet.datasets.api_wrappers import COCOeval
from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS
from mmengine.fileio import load


@METRICS.register_module()
class AITODMetric(CocoMetric):
    """COCO AP plus the official AI-TOD-v2 scale-specific AP metrics.

    The overall AP, AP50 and AP75 use the standard IoU range 0.50:0.95.
    Scale partitions follow the official AI-TOD COCO API:

    - very tiny: 0--8 pixels;
    - tiny: 8--16 pixels;
    - small: 16--32 pixels;
    - medium: at least 32 pixels.

    ``proposal_nums`` defaults to ``(100, 300, 1500)`` because dense aerial
    scenes require the official 1500-detection evaluation cap.
    """

    default_prefix = 'aitod'
    AREA_RANGES = (
        (0**2, 1e5**2),
        (0**2, 8**2),
        (8**2, 16**2),
        (16**2, 32**2),
        (32**2, 1e5**2),
    )
    AREA_LABELS = ('all', 'verytiny', 'tiny', 'small', 'medium')

    def __init__(self, *args, proposal_nums=(100, 300, 1500), **kwargs):
        super().__init__(*args, proposal_nums=proposal_nums, **kwargs)
        if len(self.iou_thrs) != 10 or not np.allclose(
            self.iou_thrs, np.linspace(0.5, 0.95, 10)
        ):
            raise ValueError(
                'AITODMetric requires the standard COCO IoU thresholds '
                '0.50:0.05:0.95. Do not use AP25 as the primary endpoint.'
            )

    @staticmethod
    def _mean_precision(coco_eval: COCOeval, area_index: int) -> float:
        precision = coco_eval.eval['precision'][:, :, :, area_index, -1]
        precision = precision[precision > -1]
        return float(np.mean(precision)) if precision.size else -1.0

    def compute_metrics(self, results: list) -> Dict[str, float]:
        # Keep the prediction JSON alive long enough to run the official size
        # partitions after MMDetection computes standard COCO metrics.
        owned_tmp_dir = None
        original_prefix = self.outfile_prefix
        if original_prefix is None:
            owned_tmp_dir = tempfile.TemporaryDirectory()
            self.outfile_prefix = osp.join(owned_tmp_dir.name, 'results')

        try:
            eval_results = super().compute_metrics(results)
            if self.format_only or 'bbox' not in self.metrics:
                return eval_results

            predictions = load(f'{self.outfile_prefix}.bbox.json')
            if not predictions:
                return eval_results
            coco_dt = self._coco_api.loadRes(predictions)
            coco_eval = COCOeval(self._coco_api, coco_dt, 'bbox')
            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs
            coco_eval.params.areaRng = [list(item) for item in self.AREA_RANGES]
            coco_eval.params.areaRngLbl = list(self.AREA_LABELS)
            coco_eval.evaluate()
            coco_eval.accumulate()

            for area_index, suffix in enumerate(('all', 'vt', 't', 's', 'm')):
                if suffix == 'all':
                    continue
                value = self._mean_precision(coco_eval, area_index)
                eval_results[f'bbox_mAP_{suffix}'] = float(f'{value:.3f}')

            # COCO's default large partition is not part of the AI-TOD-v2
            # reporting contract.  APs/APm above now refer to 16--32/>=32.
            eval_results.pop('bbox_mAP_l', None)
            return eval_results
        finally:
            self.outfile_prefix = original_prefix
            if owned_tmp_dir is not None:
                owned_tmp_dir.cleanup()
