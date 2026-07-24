"""Recalculate all four frozen BEACON-ECG endpoints from one prediction file."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "40_submission_correction_metrics.py"
_SPEC = importlib.util.spec_from_file_location("submission_correction_metrics", METRICS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load metrics module from {METRICS_PATH}")
_METRICS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_METRICS)


ENDPOINTS = {
    "hfref_le40": {
        "display_name": "HFrEF",
        "label_col": "hfref_le40",
        "prediction_col": "pred_hfref_le40",
    },
    "as_severe": {
        "display_name": "Severe aortic stenosis",
        "label_col": "as_severe",
        "prediction_col": "pred_as_severe",
    },
    "lvh": {
        "display_name": "Left ventricular hypertrophy",
        "label_col": "lvh",
        "prediction_col": "pred_lvh",
    },
    "dead_365d": {
        "display_name": "One-year mortality",
        "label_col": "dead_365d",
        "prediction_col": "pred_dead_365d",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_all_endpoints(
    predictions: pd.DataFrame,
    n_boot: int = 5000,
    seed: int = 20260723,
) -> dict[str, object]:
    """Evaluate the prespecified endpoints after common validity filtering."""
    if "subject_id" not in predictions:
        raise ValueError("missing subject_id")
    eligible = predictions.copy()
    if "ok" in eligible:
        eligible = eligible.loc[eligible["ok"].fillna(False).astype(bool)].copy()

    output: dict[str, object] = {
        "schema_version": "1.0",
        "seed": int(seed),
        "bootstrap_replicates": int(n_boot),
        "endpoints": {},
    }
    endpoints: dict[str, object] = output["endpoints"]  # type: ignore[assignment]
    for offset, (name, specification) in enumerate(ENDPOINTS.items()):
        label_col = specification["label_col"]
        prediction_col = specification["prediction_col"]
        missing = [column for column in (label_col, prediction_col) if column not in eligible]
        if missing:
            raise ValueError(f"{name}: missing required columns {missing}")
        frame = eligible[["subject_id", label_col, prediction_col]].dropna().rename(
            columns={label_col: "label", prediction_col: "probability"}
        )
        checked = _METRICS.validate_predictions(frame)
        result = _METRICS.stratified_patient_bootstrap(
            checked["label"].to_numpy(),
            checked["probability"].to_numpy(),
            n_boot=n_boot,
            seed=seed + offset,
        )
        labels = checked["label"].astype(int)
        endpoints[name] = {
            "display_name": specification["display_name"],
            "n": int(len(checked)),
            "events": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "bootstrap_replicates": int(result.pop("n_valid")),
            "metrics": result,
        }
    return output


def write_json(result: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(result: dict[str, object], path: Path) -> None:
    rows = []
    for name, endpoint in result["endpoints"].items():  # type: ignore[union-attr]
        row = {
            "endpoint": name,
            "display_name": endpoint["display_name"],
            "n": endpoint["n"],
            "events": endpoint["events"],
            "prevalence": endpoint["prevalence"],
        }
        for metric_name, metric in endpoint["metrics"].items():
            row[f"{metric_name}_estimate"] = metric["estimate"]
            row[f"{metric_name}_ci_lower"] = metric["ci_lower"]
            row[f"{metric_name}_ci_upper"] = metric["ci_upper"]
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions)
    result = evaluate_all_endpoints(predictions, n_boot=args.bootstrap, seed=args.seed)
    result["input"] = {
        "path": str(args.predictions),
        "sha256": sha256_file(args.predictions),
    }
    write_json(result, args.output_json)
    write_csv(result, args.output_csv)


if __name__ == "__main__":
    main()
