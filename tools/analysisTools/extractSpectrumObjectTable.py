#!/usr/bin/env python3
"""Create a scene-aware per-object spectral table from frozen predictions."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment


def xywh_to_center(boxes: np.ndarray) -> np.ndarray:
    output = np.asarray(boxes, dtype=float).copy()
    output[:, 0] += output[:, 2] / 2
    output[:, 1] += output[:, 3] / 2
    return output


def nwd_error(gt_boxes: np.ndarray, pred_boxes: np.ndarray, constant: float) -> np.ndarray:
    """Pairwise 1-NWD for axis-aligned boxes represented as COCO xywh."""
    if constant <= 0:
        raise ValueError('NWD normalization constant must be positive.')
    gt = xywh_to_center(gt_boxes)
    pred = xywh_to_center(pred_boxes)
    gt_gaussian = np.column_stack([gt[:, :2], gt[:, 2:] / 2])
    pred_gaussian = np.column_stack([pred[:, :2], pred[:, 2:] / 2])
    distance = np.sqrt(
        ((gt_gaussian[:, None, :] - pred_gaussian[None, :, :]) ** 2).sum(axis=2)
    )
    return 1.0 - np.exp(-distance / constant)


def assign_predictions(
    annotations: list[dict],
    predictions: list[dict],
    constant: float,
) -> dict[int, tuple[float, float | None]]:
    """One-to-one class-aware matching; unmatched ground truths receive error 1."""
    assignments = {int(ann['id']): (1.0, None) for ann in annotations}
    gt_by_class: dict[int, list[dict]] = defaultdict(list)
    pred_by_class: dict[int, list[dict]] = defaultdict(list)
    for annotation in annotations:
        gt_by_class[int(annotation['category_id'])].append(annotation)
    for prediction in predictions:
        pred_by_class[int(prediction['category_id'])].append(prediction)

    for category_id, ground_truth in gt_by_class.items():
        candidates = pred_by_class.get(category_id, [])
        if not candidates:
            continue
        cost = nwd_error(
            np.asarray([item['bbox'] for item in ground_truth], dtype=float),
            np.asarray([item['bbox'] for item in candidates], dtype=float),
            constant,
        )
        gt_indices, pred_indices = linear_sum_assignment(cost)
        for gt_index, pred_index in zip(gt_indices, pred_indices):
            annotation_id = int(ground_truth[gt_index]['id'])
            assignments[annotation_id] = (
                float(cost[gt_index, pred_index]),
                float(candidates[pred_index].get('score', np.nan)),
            )
    return assignments


def crop_patch(
    image: np.ndarray,
    bbox: list[float],
    window_scale: float,
    min_window: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    x, y, width, height = map(float, bbox)
    side = max(min_window, int(np.ceil(window_scale * max(width, height))))
    side += side % 2
    center_x, center_y = x + width / 2, y + height / 2
    x0 = int(np.floor(center_x - side / 2))
    y0 = int(np.floor(center_y - side / 2))
    x1, y1 = x0 + side, y0 + side
    src_x0, src_y0 = max(x0, 0), max(y0, 0)
    src_x1, src_y1 = min(x1, image.shape[1]), min(y1, image.shape[0])
    patch = image[src_y0:src_y1, src_x0:src_x1]
    pads = (
        (src_y0 - y0, y1 - src_y1),
        (src_x0 - x0, x1 - src_x1),
    )
    patch = np.pad(patch, pads, mode='reflect')
    local_bbox = (x - x0, y - y0, width, height)
    return patch, local_bbox


def patch_descriptors(
    patch: np.ndarray,
    local_bbox: tuple[float, float, float, float],
) -> dict[str, float]:
    patch = np.asarray(patch, dtype=np.float64) / 255.0
    height, width = patch.shape
    x, y, box_width, box_height = local_bbox
    x0, y0 = max(int(np.floor(x)), 0), max(int(np.floor(y)), 0)
    x1 = min(int(np.ceil(x + box_width)), width)
    y1 = min(int(np.ceil(y + box_height)), height)
    target_mask = np.zeros_like(patch, dtype=bool)
    target_mask[y0:y1, x0:x1] = True
    background_mask = ~target_mask
    target_mean = patch[target_mask].mean() if target_mask.any() else patch.mean()
    background_mean = (
        patch[background_mask].mean() if background_mask.any() else patch.mean()
    )
    contrast = abs(target_mean - background_mean)
    gradient_y, gradient_x = np.gradient(patch)
    gradient = np.hypot(gradient_x, gradient_y)
    background_complexity = (
        gradient[background_mask].mean() if background_mask.any() else gradient.mean()
    )

    centered = patch - patch.mean()
    centered = centered[: height - height % 2, : width - width % 2]
    x00 = centered[0::2, 0::2]
    x01 = centered[0::2, 1::2]
    x10 = centered[1::2, 0::2]
    x11 = centered[1::2, 1::2]
    subbands = (
        (x00 + x01 + x10 + x11) * 0.25,
        (x00 - x01 + x10 - x11) * 0.25,
        (x00 + x01 - x10 - x11) * 0.25,
        (x00 - x01 - x10 + x11) * 0.25,
    )
    energies = [float(np.mean(subband**2)) for subband in subbands]
    return dict(
        contrast=float(contrast),
        background_complexity=float(background_complexity),
        ll_energy=energies[0],
        lh_energy=energies[1],
        hl_energy=energies[2],
        hh_energy=energies[3],
    )


def resolve_scene_id(image: dict, scene_field: str, scene_pattern: str | None) -> str:
    if scene_field in image and image[scene_field] not in (None, ''):
        return str(image[scene_field])
    if scene_pattern:
        match = re.search(scene_pattern, image.get('file_name', ''))
        if match:
            return str(match.group(1) if match.groups() else match.group(0))
    raise ValueError(
        f"Image {image.get('id')} has no {scene_field!r}. Provide --scene-regex "
        'that extracts the original scene from file_name; tile IDs are not valid groups.'
    )


def extract_table(
    annotation_payload: dict,
    predictions: list[dict],
    image_root: Path,
    score_threshold: float = 0.05,
    nwd_constant: float = 12.8,
    window_scale: float = 4.0,
    min_window: int = 32,
    scene_field: str = 'scene_id',
    scene_pattern: str | None = None,
) -> pd.DataFrame:
    images = {int(image['id']): image for image in annotation_payload['images']}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    predictions_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in annotation_payload['annotations']:
        if annotation.get('iscrowd', 0):
            continue
        annotations_by_image[int(annotation['image_id'])].append(annotation)
    for prediction in predictions:
        if float(prediction.get('score', 0.0)) >= score_threshold:
            predictions_by_image[int(prediction['image_id'])].append(prediction)

    rows = []
    for image_id, image_annotations in sorted(annotations_by_image.items()):
        image_info = images[image_id]
        scene_id = resolve_scene_id(image_info, scene_field, scene_pattern)
        image_path = image_root / image_info['file_name']
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image = np.asarray(Image.open(image_path).convert('L'))
        matches = assign_predictions(
            image_annotations, predictions_by_image.get(image_id, []), nwd_constant
        )
        for annotation in image_annotations:
            x, y, width, height = map(float, annotation['bbox'])
            patch, local_bbox = crop_patch(
                image, annotation['bbox'], window_scale, min_window
            )
            descriptor = patch_descriptors(patch, local_bbox)
            target_error, match_score = matches[int(annotation['id'])]
            rows.append(dict(
                scene_id=scene_id,
                image_id=image_id,
                file_name=image_info['file_name'],
                annotation_id=int(annotation['id']),
                class_id=int(annotation['category_id']),
                target_error=target_error,
                matched_score=match_score,
                width=width,
                height=height,
                window_scale=window_scale,
                min_window=min_window,
                nwd_constant=nwd_constant,
                score_threshold=score_threshold,
                **descriptor,
            ))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations', type=Path)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('image_root', type=Path)
    parser.add_argument('output_csv', type=Path)
    parser.add_argument('--score-threshold', type=float, default=0.05)
    parser.add_argument('--nwd-constant', type=float, default=12.8)
    parser.add_argument('--window-scale', type=float, default=4.0)
    parser.add_argument('--min-window', type=int, default=32)
    parser.add_argument('--scene-field', default='scene_id')
    parser.add_argument('--scene-regex')
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error('--score-threshold must be in [0, 1]')
    if args.window_scale <= 0 or args.min_window < 2:
        parser.error('window parameters must be positive')

    annotation_payload = json.loads(args.annotations.read_text(encoding='utf-8'))
    predictions = json.loads(args.predictions.read_text(encoding='utf-8'))
    table = extract_table(
        annotation_payload,
        predictions,
        args.image_root,
        score_threshold=args.score_threshold,
        nwd_constant=args.nwd_constant,
        window_scale=args.window_scale,
        min_window=args.min_window,
        scene_field=args.scene_field,
        scene_pattern=args.scene_regex,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    print(f'Wrote {len(table)} objects from {table.scene_id.nunique()} scenes.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
