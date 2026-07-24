"""Corrected post-hoc ECG-signature increment beyond echo, age, and sex."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "40_submission_correction_metrics.py"
_SPEC = importlib.util.spec_from_file_location("submission_correction_metrics", METRICS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load metrics module from {METRICS_PATH}")
_METRICS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_METRICS)


ECHO_FEATURES = [
    "lvef_value",
    "av_pk_vel",
    "av_mean_grad",
    "av_area_continuity",
    "septal_thickness",
    "lvedd",
    "lvesd",
    "la_vol",
    "tr_mmhg",
]
DEMOGRAPHIC_FEATURES = ["age_at_ecg", "sex_male"]
ECG_SIGNATURES = ["pred_hfref_le40", "pred_as_severe", "pred_lvh"]


def _pipeline(class_weight: str | None, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight=class_weight,
                    random_state=seed,
                ),
            ),
        ]
    )


def cross_validated_predictions(
    data: pd.DataFrame,
    label_col: str,
    feature_cols: list[str],
    folds: int = 5,
    seed: int = 20260723,
    class_weight: str | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Return OOF probabilities with preprocessing fit separately in each fold."""
    missing = [column for column in [label_col, *feature_cols] if column not in data]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    y = data[label_col].to_numpy(dtype=int)
    if np.unique(y).size != 2:
        raise ValueError("label must contain both classes")
    x = data[feature_cols].astype(float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    predictions = np.full(len(data), np.nan, dtype=float)
    audits: list[dict[str, object]] = []
    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(x, y)):
        pipeline = _pipeline(class_weight=class_weight, seed=seed + fold_index)
        pipeline.fit(x.iloc[train_indices], y[train_indices])
        predictions[test_indices] = pipeline.predict_proba(x.iloc[test_indices])[:, 1]
        audits.append(
            {
                "fold": fold_index,
                "train_n": int(len(train_indices)),
                "test_n": int(len(test_indices)),
                "train_indices": train_indices.tolist(),
                "test_indices": test_indices.tolist(),
                "imputer_statistics": pipeline.named_steps["imputer"].statistics_.tolist(),
            }
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("OOF prediction assignment was incomplete")
    return predictions, audits


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    restored = np.empty(len(values), dtype=float)
    restored[order] = ranks
    return restored


def paired_delong(
    y_true: np.ndarray, first: np.ndarray, second: np.ndarray
) -> dict[str, float]:
    """Paired DeLong comparison for two correlated AUROCs."""
    y = np.asarray(y_true, dtype=int)
    positive = y == 1
    negative = y == 0
    m = int(positive.sum())
    n = int(negative.sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong comparison requires both classes")
    predictions = np.vstack([first, second])
    ordered = np.concatenate([np.flatnonzero(positive), np.flatnonzero(negative)])
    predictions = predictions[:, ordered]
    tx = np.array([_midrank(row[:m]) for row in predictions])
    ty = np.array([_midrank(row[m:]) for row in predictions])
    tz = np.array([_midrank(row) for row in predictions])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n
    contrast = np.array([1.0, -1.0])
    variance = max(float(contrast @ covariance @ contrast), 0.0)
    delta = float(aucs[0] - aucs[1])
    if variance == 0.0:
        z_value = 0.0 if delta == 0.0 else float(np.sign(delta) * np.inf)
        p_value = 1.0 if delta == 0.0 else 0.0
    else:
        z_value = float(delta / np.sqrt(variance))
        p_value = float(2.0 * stats.norm.sf(abs(z_value)))
    return {
        "first_auroc": float(aucs[0]),
        "second_auroc": float(aucs[1]),
        "delta_auroc": delta,
        "z": z_value,
        "p": p_value,
    }


def paired_stratified_bootstrap(
    y_true: np.ndarray,
    added_prediction: np.ndarray,
    base_prediction: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Paired patient bootstrap stratified by outcome."""
    y = np.asarray(y_true, dtype=int)
    added = np.asarray(added_prediction, dtype=float)
    base = np.asarray(base_prediction, dtype=float)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    if len(negative) == 0 or len(positive) == 0:
        raise ValueError("bootstrap requires both classes")
    rng = np.random.default_rng(seed)
    delta_auroc = np.empty(replicates, dtype=float)
    delta_auprc = np.empty(replicates, dtype=float)
    brier_improvement = np.empty(replicates, dtype=float)
    for index in range(replicates):
        draw = np.concatenate(
            [
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            ]
        )
        draw_y = y[draw]
        delta_auroc[index] = roc_auc_score(draw_y, added[draw]) - roc_auc_score(
            draw_y, base[draw]
        )
        delta_auprc[index] = average_precision_score(
            draw_y, added[draw]
        ) - average_precision_score(draw_y, base[draw])
        brier_improvement[index] = brier_score_loss(
            draw_y, base[draw]
        ) - brier_score_loss(draw_y, added[draw])
    return {
        "n_valid": int(replicates),
        "delta_auroc_ci": [float(value) for value in np.quantile(delta_auroc, [0.025, 0.975])],
        "delta_auprc_ci": [float(value) for value in np.quantile(delta_auprc, [0.025, 0.975])],
        "brier_improvement_ci": [
            float(value) for value in np.quantile(brier_improvement, [0.025, 0.975])
        ],
    }


def calibration_summary(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
    model.fit(logit, np.asarray(y_true, dtype=int))
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


def evaluate_incremental_models(
    data: pd.DataFrame,
    label_col: str,
    base_features: list[str],
    added_features: list[str],
    folds: int = 5,
    bootstrap: int = 5000,
    seed: int = 20260723,
    class_weight: str | None = None,
) -> dict[str, object]:
    y = data[label_col].to_numpy(dtype=int)
    base_prediction, base_audits = cross_validated_predictions(
        data, label_col, base_features, folds, seed, class_weight
    )
    added_prediction, added_audits = cross_validated_predictions(
        data, label_col, [*base_features, *added_features], folds, seed, class_weight
    )
    base_metrics = _METRICS.binary_metrics(y, base_prediction)
    added_metrics = _METRICS.binary_metrics(y, added_prediction)
    delong = paired_delong(y, added_prediction, base_prediction)
    interval = paired_stratified_bootstrap(
        y, added_prediction, base_prediction, bootstrap, seed
    )
    return {
        "post_hoc_exploratory": True,
        "n": int(len(y)),
        "events": int(y.sum()),
        "folds": int(folds),
        "seed": int(seed),
        "class_weight": class_weight,
        "base_features": base_features,
        "added_features": added_features,
        "base_model": {
            "metrics": base_metrics,
            "calibration": calibration_summary(y, base_prediction),
        },
        "added_model": {
            "metrics": added_metrics,
            "calibration": calibration_summary(y, added_prediction),
        },
        "comparison": {
            "delta_auroc": float(added_metrics["auroc"] - base_metrics["auroc"]),
            "delta_auprc": float(added_metrics["auprc"] - base_metrics["auprc"]),
            "brier_improvement": float(base_metrics["brier"] - added_metrics["brier"]),
            "delta_auroc_ci": interval["delta_auroc_ci"],
            "delta_auprc_ci": interval["delta_auprc_ci"],
            "brier_improvement_ci": interval["brier_improvement_ci"],
            "delong_z": delong["z"],
            "delong_p": delong["p"],
            "bootstrap_replicates": interval["n_valid"],
        },
        "fold_audit": {
            "base": base_audits,
            "added": added_audits,
        },
        "oof_predictions": {
            "base": base_prediction,
            "added": added_prediction,
        },
    }


def _json_ready(result: dict[str, object]) -> dict[str, object]:
    copy = dict(result)
    copy.pop("oof_predictions", None)
    return copy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--oof-output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.predictions)
    if "ok" in frame:
        frame = frame.loc[frame["ok"].fillna(False).astype(bool)]
    required = ["subject_id", "dead_365d", *DEMOGRAPHIC_FEATURES, *ECG_SIGNATURES]
    frame = frame.dropna(subset=required).reset_index(drop=True)
    available_echo = [column for column in ECHO_FEATURES if column in frame]
    base_features = [*available_echo, *DEMOGRAPHIC_FEATURES]
    primary = evaluate_incremental_models(
        frame,
        "dead_365d",
        base_features,
        ECG_SIGNATURES,
        bootstrap=args.bootstrap,
        seed=args.seed,
        class_weight=None,
    )
    sensitivity = evaluate_incremental_models(
        frame,
        "dead_365d",
        base_features,
        ECG_SIGNATURES,
        bootstrap=args.bootstrap,
        seed=args.seed,
        class_weight="balanced",
    )
    output = {
        "analysis": "corrected_post_hoc_incremental_mortality_prediction",
        "primary_comparison": "echo+age+sex versus echo+age+sex+three frozen structural ECG scores",
        "probability_model": "unweighted L2 logistic regression with fold-local median imputation and standardization",
        "primary": _json_ready(primary),
        "balanced_weight_sensitivity": _json_ready(sensitivity),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.oof_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "subject_id": frame["subject_id"].to_numpy(),
            "dead_365d": frame["dead_365d"].astype(int).to_numpy(),
            "primary_base": primary["oof_predictions"]["base"],
            "primary_added": primary["oof_predictions"]["added"],
            "balanced_base": sensitivity["oof_predictions"]["base"],
            "balanced_added": sensitivity["oof_predictions"]["added"],
        }
    ).to_parquet(args.oof_output, index=False)


if __name__ == "__main__":
    main()
