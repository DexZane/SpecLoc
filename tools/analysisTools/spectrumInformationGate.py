#!/usr/bin/env python3
"""Scene-grouped Size-control versus Size+Spectrum mechanism gate.

The input is one row per ground-truth object from a frozen detector.  This
script deliberately does not train or modify a detector.  It tests whether
predefined local spectral descriptors predict localization error after
controlling for object geometry, class, contrast and background complexity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    'scene_id',
    'class_id',
    'target_error',
    'width',
    'height',
    'contrast',
    'background_complexity',
    'll_energy',
    'lh_energy',
    'hl_energy',
    'hh_energy',
)
CONTROL_FEATURES = (
    'log_width',
    'log_height',
    'log_area',
    'log_aspect_ratio',
    'contrast',
    'background_complexity',
)
SPECTRUM_FEATURES = (
    'log_ll_energy',
    'log_lh_energy',
    'log_hl_energy',
    'log_hh_energy',
    'log_ll_hf_ratio',
    'hf_fraction',
)


def validate_table(table: pd.DataFrame, folds: int) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in table]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if table.empty:
        raise ValueError('The object table is empty.')
    numeric = [column for column in REQUIRED_COLUMNS if column not in {'scene_id', 'class_id'}]
    values = table[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Numeric columns must contain only finite values.')
    if (table[['width', 'height']].to_numpy(dtype=float) <= 0).any():
        raise ValueError('Object width and height must be positive.')
    if (table[['ll_energy', 'lh_energy', 'hl_energy', 'hh_energy']].to_numpy() < 0).any():
        raise ValueError('Spectral energies must be non-negative.')
    if table['scene_id'].nunique() < folds:
        raise ValueError(
            f'Need at least {folds} distinct scenes for {folds}-fold grouped CV.'
        )


def engineer_features(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    eps = np.finfo(float).eps
    width = output['width'].to_numpy(dtype=float)
    height = output['height'].to_numpy(dtype=float)
    output['log_width'] = np.log(width)
    output['log_height'] = np.log(height)
    output['log_area'] = np.log(width * height)
    output['log_aspect_ratio'] = np.log(width / height)

    energies = output[
        ['ll_energy', 'lh_energy', 'hl_energy', 'hh_energy']
    ].to_numpy(dtype=float)
    for index, name in enumerate(('ll', 'lh', 'hl', 'hh')):
        output[f'log_{name}_energy'] = np.log1p(energies[:, index])
    hf_energy = energies[:, 1:].sum(axis=1)
    total_energy = energies.sum(axis=1)
    output['log_ll_hf_ratio'] = np.log((energies[:, 0] + eps) / (hf_energy + eps))
    output['hf_fraction'] = hf_energy / (total_energy + eps)
    return output


def grouped_folds(groups: Iterable, n_splits: int, seed: int) -> list[np.ndarray]:
    groups = np.asarray(list(groups), dtype=object)
    unique, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    tie_break = rng.random(len(unique))
    order = np.lexsort((tie_break, -counts))
    fold_sizes = np.zeros(n_splits, dtype=int)
    fold_groups: list[list[object]] = [[] for _ in range(n_splits)]
    for index in order:
        fold_index = int(np.argmin(fold_sizes))
        fold_groups[fold_index].append(unique[index])
        fold_sizes[fold_index] += counts[index]
    return [np.isin(groups, group_set) for group_set in fold_groups]


def _design_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    train_numeric = train[list(numeric_features)].to_numpy(dtype=float)
    test_numeric = test[list(numeric_features)].to_numpy(dtype=float)
    mean = train_numeric.mean(axis=0)
    std = train_numeric.std(axis=0)
    std[std < 1e-12] = 1.0
    train_numeric = (train_numeric - mean) / std
    test_numeric = (test_numeric - mean) / std

    categories = sorted(train['class_id'].astype(str).unique())
    train_class = train['class_id'].astype(str).to_numpy()
    test_class = test['class_id'].astype(str).to_numpy()
    train_one_hot = np.column_stack([train_class == item for item in categories]).astype(float)
    test_one_hot = np.column_stack([test_class == item for item in categories]).astype(float)
    return (
        np.column_stack([train_numeric, train_one_hot]),
        np.column_stack([test_numeric, test_one_hot]),
    )


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    ridge: float,
) -> np.ndarray:
    train_design = np.column_stack([np.ones(len(train_x)), train_x])
    test_design = np.column_stack([np.ones(len(test_x)), test_x])
    penalty = np.eye(train_design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(
        train_design.T @ train_design + penalty
    ) @ train_design.T @ train_y
    return test_design @ coefficients


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = target - prediction
    denominator = np.sum((target - target.mean()) ** 2)
    r2 = 1.0 - np.sum(residual**2) / denominator if denominator > 0 else np.nan
    return {
        'mae': float(np.mean(np.abs(residual))),
        'rmse': float(np.sqrt(np.mean(residual**2))),
        'r2': float(r2),
    }


def _calibration(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if np.std(prediction) < 1e-12:
        return {'intercept': float(target.mean()), 'slope': 0.0}
    slope, intercept = np.polyfit(prediction, target, deg=1)
    return {'intercept': float(intercept), 'slope': float(slope)}


def _scene_bootstrap_ci(
    values: np.ndarray,
    scenes: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    unique = np.unique(scenes)
    scene_values = np.asarray([values[scenes == scene].mean() for scene in unique])
    rng = np.random.default_rng(seed)
    draws = rng.choice(scene_values, size=(samples, len(scene_values)), replace=True)
    means = draws.mean(axis=1)
    return tuple(float(item) for item in np.quantile(means, [0.025, 0.975]))


def analyze_table(
    table: pd.DataFrame,
    folds: int = 5,
    seed: int = 42,
    ridge: float = 1.0,
    bootstrap_samples: int = 5000,
    min_relative_improvement: float = 0.02,
) -> tuple[dict, pd.DataFrame]:
    validate_table(table, folds)
    data = engineer_features(table).reset_index(drop=True)
    masks = grouped_folds(data['scene_id'], folds, seed)
    target = data['target_error'].to_numpy(dtype=float)
    control_oof = np.full(len(data), np.nan)
    spectrum_oof = np.full(len(data), np.nan)
    fold_records = []

    for fold_index, test_mask in enumerate(masks):
        train_mask = ~test_mask
        train = data.loc[train_mask]
        test = data.loc[test_mask]
        control_train_x, control_test_x = _design_matrices(
            train, test, CONTROL_FEATURES
        )
        spectrum_train_x, spectrum_test_x = _design_matrices(
            train, test, CONTROL_FEATURES + SPECTRUM_FEATURES
        )
        control_prediction = _ridge_predict(
            control_train_x, target[train_mask], control_test_x, ridge
        )
        spectrum_prediction = _ridge_predict(
            spectrum_train_x, target[train_mask], spectrum_test_x, ridge
        )
        control_oof[test_mask] = control_prediction
        spectrum_oof[test_mask] = spectrum_prediction
        control_metrics = _metrics(target[test_mask], control_prediction)
        spectrum_metrics = _metrics(target[test_mask], spectrum_prediction)
        train_scenes = set(data.loc[train_mask, 'scene_id'])
        test_scenes = set(data.loc[test_mask, 'scene_id'])
        if train_scenes & test_scenes:
            raise RuntimeError('Scene leakage detected in grouped folds.')
        fold_records.append({
            'fold': fold_index,
            'train_scenes': len(train_scenes),
            'test_scenes': len(test_scenes),
            'test_objects': int(test_mask.sum()),
            'control': control_metrics,
            'size_plus_spectrum': spectrum_metrics,
            'mae_improvement': control_metrics['mae'] - spectrum_metrics['mae'],
        })

    if not np.isfinite(control_oof).all() or not np.isfinite(spectrum_oof).all():
        raise RuntimeError('Out-of-fold predictions are incomplete.')
    absolute_error_improvement = np.abs(target - control_oof) - np.abs(
        target - spectrum_oof
    )
    ci_low, ci_high = _scene_bootstrap_ci(
        absolute_error_improvement,
        data['scene_id'].to_numpy(),
        bootstrap_samples,
        seed + 1,
    )
    control_metrics = _metrics(target, control_oof)
    spectrum_metrics = _metrics(target, spectrum_oof)
    mean_improvement = control_metrics['mae'] - spectrum_metrics['mae']
    relative_improvement = mean_improvement / max(control_metrics['mae'], 1e-12)
    improving_folds = sum(record['mae_improvement'] > 0 for record in fold_records)
    stable_fraction = improving_folds / folds
    passed = (
        ci_low > 0
        and relative_improvement >= min_relative_improvement
        and stable_fraction >= 0.8
    )

    report = {
        'gate': 'Size-control versus Size+Spectrum',
        'independent_unit': 'scene_id',
        'objects': len(data),
        'scenes': int(data['scene_id'].nunique()),
        'folds': folds,
        'seed': seed,
        'ridge': ridge,
        'control_features': list(CONTROL_FEATURES) + ['class_id_one_hot'],
        'spectrum_features': list(SPECTRUM_FEATURES),
        'control_oof': control_metrics,
        'size_plus_spectrum_oof': spectrum_metrics,
        'control_calibration': _calibration(target, control_oof),
        'size_plus_spectrum_calibration': _calibration(target, spectrum_oof),
        'mae_improvement': float(mean_improvement),
        'relative_mae_improvement': float(relative_improvement),
        'scene_bootstrap_95ci': [ci_low, ci_high],
        'improving_fold_fraction': stable_fraction,
        'predefined_pass_rule': {
            'ci_lower_bound_gt_zero': True,
            'minimum_relative_improvement': min_relative_improvement,
            'minimum_improving_fold_fraction': 0.8,
        },
        'decision': 'pass' if passed else 'stop',
        'fold_results': fold_records,
    }
    predictions = data[['scene_id', 'class_id', 'target_error']].copy()
    predictions['control_oof_prediction'] = control_oof
    predictions['spectrum_oof_prediction'] = spectrum_oof
    predictions['absolute_error_improvement'] = absolute_error_improvement
    return report, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv', type=Path)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--predictions-csv', type=Path)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ridge', type=float, default=1.0)
    parser.add_argument('--bootstrap-samples', type=int, default=5000)
    parser.add_argument('--min-relative-improvement', type=float, default=0.02)
    args = parser.parse_args()
    if args.ridge < 0:
        parser.error('--ridge must be non-negative')
    if args.bootstrap_samples < 100:
        parser.error('--bootstrap-samples must be at least 100')

    table = pd.read_csv(args.input_csv)
    report, predictions = analyze_table(
        table,
        folds=args.folds,
        seed=args.seed,
        ridge=args.ridge,
        bootstrap_samples=args.bootstrap_samples,
        min_relative_improvement=args.min_relative_improvement,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    if args.predictions_csv:
        args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.predictions_csv, index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['decision'] == 'pass' else 3


if __name__ == '__main__':
    raise SystemExit(main())
