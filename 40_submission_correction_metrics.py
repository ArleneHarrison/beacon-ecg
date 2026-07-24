"""Validated, deterministic metrics for the BEACON-ECG submission rerun."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def validate_predictions(
    data: pd.DataFrame,
    patient_col: str = "subject_id",
    label_col: str = "label",
    probability_col: str = "probability",
) -> pd.DataFrame:
    """Validate one patient-level binary prediction table and return a copy."""
    required = [patient_col, label_col, probability_col]
    missing_columns = [column for column in required if column not in data]
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    checked = data.copy()
    if checked[patient_col].isna().any():
        raise ValueError("missing patient identifier")
    if checked[patient_col].duplicated().any():
        raise ValueError("duplicate patient identifier")
    if checked[label_col].isna().any():
        raise ValueError("missing label")

    labels = checked[label_col].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("label must be binary (0 or 1)")

    probabilities = checked[probability_col].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("probability must be finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("probability must be in [0, 1]")
    return checked


def _validated_arrays(
    y_true: Iterable[float], y_prob: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    probabilities = np.asarray(y_prob, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be one-dimensional and equally sized")
    if len(labels) == 0:
        raise ValueError("labels and probabilities must not be empty")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("probabilities must be in [0, 1]")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    if np.unique(labels).size != 2:
        raise ValueError("labels must contain both classes")
    return labels.astype(int, copy=False), probabilities


def binary_metrics(y_true: Iterable[float], y_prob: Iterable[float]) -> dict[str, float]:
    """Return the three prespecified binary prediction metrics."""
    labels, probabilities = _validated_arrays(y_true, y_prob)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def stratified_patient_bootstrap(
    y_true: Iterable[float],
    y_prob: Iterable[float],
    n_boot: int = 5000,
    seed: int = 20260723,
) -> dict[str, object]:
    """Bootstrap patients within outcome class and return percentile intervals."""
    labels, probabilities = _validated_arrays(y_true, y_prob)
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(n_boot, dtype=float) for name in ("auroc", "auprc", "brier")}

    for index in range(n_boot):
        draw = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        result = binary_metrics(labels[draw], probabilities[draw])
        for name, value in result.items():
            samples[name][index] = value

    estimates = binary_metrics(labels, probabilities)
    output: dict[str, object] = {"n_valid": int(n_boot)}
    for name, values in samples.items():
        lower, upper = np.quantile(values, [0.025, 0.975])
        output[name] = {
            "estimate": estimates[name],
            "ci_lower": float(lower),
            "ci_upper": float(upper),
        }
    return output
