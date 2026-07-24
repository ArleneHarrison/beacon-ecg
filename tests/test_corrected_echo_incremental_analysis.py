from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "42_corrected_echo_incremental_analysis.py"


def load_module():
    spec = importlib.util.spec_from_file_location("corrected_echo_incremental", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fold_preprocessing_uses_training_rows_only():
    module = load_module()
    n = 40
    frame = pd.DataFrame(
        {
            "feature": np.arange(n, dtype=float),
            "label": np.tile([0, 1], n // 2),
        }
    )
    extreme_index = 39
    frame.loc[extreme_index, "feature"] = 10000.0
    _, audits = module.cross_validated_predictions(
        frame,
        label_col="label",
        feature_cols=["feature"],
        folds=5,
        seed=20260723,
    )
    held_out_fold = next(audit for audit in audits if extreme_index in audit["test_indices"])
    expected = float(frame.drop(index=held_out_fold["test_indices"])["feature"].median())
    assert held_out_fold["imputer_statistics"][0] == expected


def test_cross_validated_predictions_are_deterministic():
    module = load_module()
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"x": rng.normal(size=100), "label": np.tile([0, 1], 50)})
    first, _ = module.cross_validated_predictions(
        frame, "label", ["x"], folds=5, seed=20260723
    )
    second, _ = module.cross_validated_predictions(
        frame, "label", ["x"], folds=5, seed=20260723
    )
    np.testing.assert_array_equal(first, second)


def test_informative_ecg_signature_improves_paired_auroc():
    module = load_module()
    rng = np.random.default_rng(11)
    n = 300
    y = rng.integers(0, 2, size=n)
    frame = pd.DataFrame(
        {
            "subject_id": np.arange(n),
            "dead_365d": y,
            "echo": rng.normal(size=n),
            "age_at_ecg": rng.normal(65, 10, size=n),
            "sex_male": rng.integers(0, 2, size=n),
            "signature": np.clip(0.1 + 0.8 * y + rng.normal(0, 0.04, size=n), 0, 1),
        }
    )
    result = module.evaluate_incremental_models(
        frame,
        label_col="dead_365d",
        base_features=["echo", "age_at_ecg", "sex_male"],
        added_features=["signature"],
        folds=5,
        bootstrap=100,
        seed=20260723,
    )
    assert result["comparison"]["delta_auroc"] > 0.20
    assert result["comparison"]["delta_auroc_ci"][0] > 0.0
    assert 0.0 <= result["comparison"]["delong_p"] <= 1.0
    assert result["post_hoc_exploratory"] is True


def test_paired_bootstrap_is_deterministic():
    module = load_module()
    y = np.array([0, 0, 0, 1, 1, 1])
    base = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    added = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
    first = module.paired_stratified_bootstrap(y, added, base, 100, 20260723)
    second = module.paired_stratified_bootstrap(y, added, base, 100, 20260723)
    assert first == second
    assert first["n_valid"] == 100
