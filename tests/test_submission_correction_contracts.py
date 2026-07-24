from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "40_submission_correction_metrics.py"


def load_module():
    spec = importlib.util.spec_from_file_location("submission_correction_metrics", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def toy_predictions() -> pd.DataFrame:
    path = Path(__file__).parent / "fixtures" / "submission_correction_toy.csv"
    return pd.read_csv(path)


def test_validate_predictions_accepts_unique_binary_finite_probabilities(toy_predictions):
    metrics = load_module()
    checked = metrics.validate_predictions(toy_predictions)
    assert checked.shape == toy_predictions.shape
    assert checked["subject_id"].is_unique


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True), "duplicate patient"),
        (lambda d: d.assign(label=d["label"].mask(d.index == 0)), "missing label"),
        (lambda d: d.assign(label=2), "binary"),
        (lambda d: d.assign(probability=np.inf), "finite"),
        (lambda d: d.assign(probability=1.01), r"\[0, 1\]"),
    ],
)
def test_validate_predictions_rejects_invalid_inputs(toy_predictions, mutator, message):
    metrics = load_module()
    with pytest.raises(ValueError, match=message):
        metrics.validate_predictions(mutator(toy_predictions.copy()))


def test_binary_metrics_has_submission_contract(toy_predictions):
    metrics = load_module()
    result = metrics.binary_metrics(
        toy_predictions["label"].to_numpy(), toy_predictions["probability"].to_numpy()
    )
    assert set(result) == {"auroc", "auprc", "brier"}
    assert result["auroc"] == pytest.approx(1.0)
    assert 0.0 <= result["brier"] <= 1.0


def test_stratified_bootstrap_is_deterministic_and_keeps_all_replicates(toy_predictions):
    metrics = load_module()
    kwargs = dict(
        y_true=toy_predictions["label"].to_numpy(),
        y_prob=toy_predictions["probability"].to_numpy(),
        n_boot=200,
        seed=20260723,
    )
    first = metrics.stratified_patient_bootstrap(**kwargs)
    second = metrics.stratified_patient_bootstrap(**kwargs)
    assert first == second
    assert first["n_valid"] == 200
    for name in ("auroc", "auprc", "brier"):
        assert set(first[name]) == {"estimate", "ci_lower", "ci_upper"}
        assert first[name]["ci_lower"] <= first[name]["estimate"] <= first[name]["ci_upper"]


def test_bootstrap_rejects_single_class_labels(toy_predictions):
    metrics = load_module()
    with pytest.raises(ValueError, match="both classes"):
        metrics.stratified_patient_bootstrap(
            np.zeros(len(toy_predictions), dtype=int),
            toy_predictions["probability"].to_numpy(),
            n_boot=20,
            seed=20260723,
        )
