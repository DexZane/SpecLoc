import numpy as np
import pandas as pd

from tools.analysisTools.spectrumInformationGate import (
    analyze_table,
    grouped_folds,
)


def _synthetic_table(seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for scene in range(30):
        for _ in range(8):
            width = rng.uniform(3, 24)
            height = rng.uniform(3, 24)
            ll = rng.gamma(2, 1)
            lh, hl, hh = rng.gamma(2, 1, size=3)
            hf_fraction = (lh + hl + hh) / (ll + lh + hl + hh)
            error = (
                3.0 / np.sqrt(width * height)
                + 1.5 * hf_fraction
                + rng.normal(0, 0.04)
            )
            rows.append(dict(
                scene_id=f'scene_{scene}',
                class_id=scene % 3,
                target_error=error,
                width=width,
                height=height,
                contrast=rng.normal(),
                background_complexity=rng.uniform(),
                ll_energy=ll,
                lh_energy=lh,
                hl_energy=hl,
                hh_energy=hh,
            ))
    return pd.DataFrame(rows)


def test_grouped_folds_never_split_a_scene():
    groups = np.repeat(np.arange(12), [1, 2, 3, 4] * 3)
    folds = grouped_folds(groups, n_splits=4, seed=3)
    assert np.all(np.sum(np.column_stack(folds), axis=1) == 1)
    for group in np.unique(groups):
        memberships = [mask[groups == group].any() for mask in folds]
        assert sum(memberships) == 1


def test_spectral_signal_passes_the_predefined_gate():
    report, predictions = analyze_table(
        _synthetic_table(),
        folds=5,
        seed=42,
        bootstrap_samples=500,
        min_relative_improvement=0.02,
    )
    assert report['decision'] == 'pass'
    assert report['scene_bootstrap_95ci'][0] > 0
    assert report['improving_fold_fraction'] >= 0.8
    assert predictions['spectrum_oof_prediction'].notna().all()
