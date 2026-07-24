"""Reproducible EchoNext flow audit and optional frozen-ensemble inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ENDPOINTS = {
    "hfref_le40": {"task_index": 0, "definition": "LVEF <= 40%"},
    "lvh": {"task_index": 2, "definition": "max(IVS, LVPW) >= 1.5 cm"},
}
MEMBERS = {
    "baseline": {"checkpoint": "outputs/final/best.pt", "weight": 1.0, "kind": "waveform"},
    "ptbxl": {"checkpoint": "outputs/ft/best.pt", "weight": 1.0, "kind": "waveform"},
    "cognition": {"checkpoint": "outputs/cog/best_cog.pt", "weight": 2.0, "kind": "cognition"},
    "vigilance": {"checkpoint": "outputs/vigil/best_cog.pt", "weight": 2.0, "kind": "vigilance"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_external_labels(test_metadata: pd.DataFrame) -> pd.DataFrame:
    """Apply the locked BEACON label thresholds to EchoNext continuous values."""
    required = ["lvef_value", "ivs_measurement", "lvpw_measurement"]
    missing = [column for column in required if column not in test_metadata]
    if missing:
        raise ValueError(f"missing external label columns: {missing}")
    mapped = test_metadata.copy().reset_index(drop=True)
    lvef = pd.to_numeric(mapped["lvef_value"], errors="coerce")
    mapped["hfref_le40"] = (lvef <= 40.0).astype(float)
    mapped.loc[lvef.isna(), "hfref_le40"] = np.nan
    walls = pd.concat(
        [
            pd.to_numeric(mapped["ivs_measurement"], errors="coerce"),
            pd.to_numeric(mapped["lvpw_measurement"], errors="coerce"),
        ],
        axis=1,
    )
    wall_max = walls.max(axis=1, skipna=True)
    mapped["lvh"] = (wall_max >= 1.5).astype(float)
    mapped.loc[wall_max.isna(), "lvh"] = np.nan
    return mapped


def build_flow(
    all_metadata: pd.DataFrame, mapped_test_metadata: pd.DataFrame
) -> dict[str, object]:
    split_counts = {
        str(key): int(value)
        for key, value in all_metadata["split"].value_counts(dropna=False).items()
    }
    test_total = int(len(mapped_test_metadata))
    endpoints = {}
    for endpoint in ENDPOINTS:
        included = int(mapped_test_metadata[endpoint].notna().sum())
        endpoints[endpoint] = {
            "included": included,
            "missing_label": int(test_total - included),
            "events": int(mapped_test_metadata[endpoint].fillna(0).sum()),
            "prevalence": float(mapped_test_metadata[endpoint].dropna().mean()),
            "definition": ENDPOINTS[endpoint]["definition"],
        }
    flow = {
        "metadata_total": int(len(all_metadata)),
        "split_counts": split_counts,
        "test_split_total": test_total,
        "endpoints": endpoints,
    }
    if split_counts.get("test") != test_total:
        raise ValueError("test split count does not reconcile")
    for endpoint, record in endpoints.items():
        if record["included"] + record["missing_label"] != test_total:
            raise ValueError(f"{endpoint} flow does not reconcile")
    return flow


def prepare_external_waveform(waveform: np.ndarray) -> np.ndarray:
    """Convert EchoNext 250-Hz N×1×2500×12 storage to BEACON 12×5000 input."""
    wave = np.asarray(waveform, dtype=np.float32).squeeze()
    if wave.shape == (12, 2500):
        wave = wave.T
    if wave.shape != (2500, 12):
        raise ValueError(f"expected EchoNext waveform shape (2500, 12), got {wave.shape}")
    resampled = resample_poly(wave, up=2, down=1, axis=0).astype(np.float32)
    output = np.nan_to_num(
        resampled.T.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    mean = output.mean(axis=1, keepdims=True)
    standard_deviation = output.std(axis=1, keepdims=True) + 1e-6
    return ((output - mean) / standard_deviation).astype(np.float32)


def weighted_ensemble(predictions: dict[str, np.ndarray]) -> np.ndarray:
    if set(predictions) != set(MEMBERS):
        raise ValueError("prediction members do not match locked ensemble")
    shapes = {np.asarray(value).shape for value in predictions.values()}
    if len(shapes) != 1:
        raise ValueError("all member predictions must have identical shapes")
    total_weight = sum(record["weight"] for record in MEMBERS.values())
    total = sum(
        np.asarray(predictions[name], dtype=np.float64) * record["weight"]
        for name, record in MEMBERS.items()
    )
    return total / total_weight


def calibration_summary(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(probabilities / (1.0 - probabilities)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
    model.fit(logit, y_true)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if mask.any():
            ece += abs(probabilities[mask].mean() - y_true[mask].mean()) * mask.mean()
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
        "ece_10_equal_width_bins": float(ece),
    }


def stratified_bootstrap_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    if len(negative) == 0 or len(positive) == 0:
        raise ValueError("external endpoint requires both classes")
    rng = np.random.default_rng(seed)
    values = {name: np.empty(replicates) for name in ("auroc", "auprc", "brier")}
    for index in range(replicates):
        draw = np.concatenate(
            [
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            ]
        )
        y = y_true[draw]
        p = probabilities[draw]
        values["auroc"][index] = roc_auc_score(y, p)
        values["auprc"][index] = average_precision_score(y, p)
        values["brier"][index] = brier_score_loss(y, p)
    estimates = {
        "auroc": roc_auc_score(y_true, probabilities),
        "auprc": average_precision_score(y_true, probabilities),
        "brier": brier_score_loss(y_true, probabilities),
    }
    return {
        name: {
            "estimate": float(estimates[name]),
            "ci_lower": float(np.quantile(samples, 0.025)),
            "ci_upper": float(np.quantile(samples, 0.975)),
        }
        for name, samples in values.items()
    }


def audit_archived_result(
    flow: dict[str, object], archived: dict[str, object]
) -> dict[str, object]:
    comparisons = {
        "n_total_matches": archived.get("n_total") == flow["test_split_total"],
        "endpoint_denominators": {},
        "endpoint_prevalence_to_4dp": {},
    }
    for endpoint in ENDPOINTS:
        record = flow["endpoints"][endpoint]
        archived_record = archived.get(endpoint, {})
        comparisons["endpoint_denominators"][endpoint] = archived_record.get("n") == record[
            "included"
        ]
        comparisons["endpoint_prevalence_to_4dp"][endpoint] = round(
            float(record["prevalence"]), 4
        ) == round(float(archived_record.get("prevalence", np.nan)), 4)
    comparisons["all_flow_checks_pass"] = bool(
        comparisons["n_total_matches"]
        and all(comparisons["endpoint_denominators"].values())
        and all(comparisons["endpoint_prevalence_to_4dp"].values())
    )
    return comparisons


def run_full_inference(
    metadata: pd.DataFrame,
    waveforms_path: Path,
    aiecg_root: Path,
    batch_size: int,
    workers: int,
    bootstrap: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    import torch
    from torch.utils.data import DataLoader, Dataset

    sys.path.insert(0, str(aiecg_root / "scripts"))
    import train_cognition
    import train_vigilance
    import train_waveform

    test = map_external_labels(metadata.loc[metadata["split"].eq("test")].copy())
    waves = np.load(waveforms_path, mmap_mode="r")
    if len(waves) != len(test):
        raise ValueError(f"waveform rows {len(waves)} do not match test metadata {len(test)}")

    class EchoNextDataset(Dataset):
        def __len__(self):
            return len(test)

        def __getitem__(self, index):
            wave = prepare_external_waveform(waves[index])
            row = test.iloc[index]
            age = float(row["age_at_ecg"]) if pd.notna(row.get("age_at_ecg")) else 65.0
            sex_text = str(row.get("sex", "")).upper()
            sex = 1.0 if sex_text.startswith("M") else 0.0 if sex_text.startswith("F") else 0.5
            context = np.array([(age - 65.0) / 16.0, sex - 0.5], dtype=np.float32)
            return torch.from_numpy(wave), torch.from_numpy(context)

    loader = DataLoader(
        EchoNextDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    member_predictions: dict[str, np.ndarray] = {}
    checkpoint_hashes = {}
    for name, record in MEMBERS.items():
        checkpoint = aiecg_root / record["checkpoint"]
        if record["kind"] == "waveform":
            model, uses_context = train_waveform.Net().to(device), False
        elif record["kind"] == "cognition":
            model, uses_context = train_cognition.CogNet().to(device), True
        else:
            model, uses_context = train_vigilance.CogNet().to(device), True
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        batches = []
        with torch.no_grad():
            for waves_batch, context_batch in loader:
                with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                    logits = (
                        model(waves_batch.to(device), context_batch.to(device))
                        if uses_context
                        else model(waves_batch.to(device))
                    )
                batches.append(torch.sigmoid(logits[:, :4]).float().cpu().numpy())
        member_predictions[name] = np.concatenate(batches, axis=0)
        checkpoint_hashes[name] = sha256_file(checkpoint)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    ensemble = weighted_ensemble(member_predictions)
    results: dict[str, object] = {
        "rerun_status": "complete",
        "waveform_preprocessing": "250 Hz to 500 Hz resample_poly; per-lead z normalization",
        "ensemble_weights": {name: record["weight"] for name, record in MEMBERS.items()},
        "checkpoint_sha256": checkpoint_hashes,
        "endpoints": {},
    }
    prediction_table = test[[column for column in ["ecg_key", "patient_key"] if column in test]].copy()
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        task_index = ENDPOINTS[endpoint]["task_index"]
        mask = test[endpoint].notna().to_numpy()
        y = test.loc[mask, endpoint].astype(int).to_numpy()
        p = ensemble[mask, task_index]
        per_member = {
            name: float(roc_auc_score(y, predictions[mask, task_index]))
            for name, predictions in member_predictions.items()
        }
        results["endpoints"][endpoint] = {
            "n": int(len(y)),
            "events": int(y.sum()),
            "prevalence": float(y.mean()),
            "metrics": stratified_bootstrap_metrics(y, p, bootstrap, seed + endpoint_index),
            "calibration": calibration_summary(y, p),
            "per_member_auroc": per_member,
        }
        prediction_table[f"label_{endpoint}"] = test[endpoint]
        prediction_table[f"probability_{endpoint}"] = ensemble[:, task_index]
    return results, prediction_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--archived-json", type=Path)
    parser.add_argument("--waveforms", type=Path)
    parser.add_argument("--aiecg-root", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    test = map_external_labels(metadata.loc[metadata["split"].eq("test")].copy())
    flow = build_flow(metadata, test)
    output: dict[str, object] = {
        "dataset": "EchoNext official test split",
        "metadata_sha256": sha256_file(args.metadata),
        "flow": flow,
        "severe_as_external_validation": "not performed because the external categorical definition is not aligned with the BEACON Vmax/gradient/AVA reference",
    }
    if args.archived_json:
        archived = json.loads(args.archived_json.read_text(encoding="utf-8"))
        output["archived_result_sha256"] = sha256_file(args.archived_json)
        output["archived_flow_audit"] = audit_archived_result(flow, archived)
    if args.waveforms:
        if not args.aiecg_root or not args.predictions_output:
            raise ValueError("full inference requires --aiecg-root and --predictions-output")
        inference, predictions = run_full_inference(
            metadata,
            args.waveforms,
            args.aiecg_root,
            args.batch_size,
            args.workers,
            args.bootstrap,
            args.seed,
        )
        output["inference"] = inference
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(args.predictions_output, index=False)
    else:
        output["rerun_status"] = "blocked_missing_authorized_EchoNext_waveform_access"
        output["required_action"] = (
            "A qualified PhysioNet account must sign the EchoNext data use agreement, then "
            "provide the official EchoNext_test_waveforms.npy file to this script."
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
