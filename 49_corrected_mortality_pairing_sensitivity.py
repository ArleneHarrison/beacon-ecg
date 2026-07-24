"""Evaluate corrected mortality across ECG–echo pairing windows and directions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "40_submission_correction_metrics.py"
_SPEC = importlib.util.spec_from_file_location("submission_metrics", METRICS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load metrics module from {METRICS_PATH}")
_METRICS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_METRICS)


def pairing_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if "days_ecg_to_echo" not in frame:
        raise ValueError("missing days_ecg_to_echo")
    days = pd.to_numeric(frame["days_ecg_to_echo"], errors="coerce")
    return {
        "primary_7_day": days.notna(),
        "absolute_le_3_days": days.abs().le(3),
        "absolute_le_1_day": days.abs().le(1),
        "echo_before_or_on_ecg": days.le(0),
        "echo_after_or_on_ecg": days.ge(0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.predictions)
    output: dict[str, object] = {
        "analysis": "corrected_mortality_pairing_sensitivity",
        "bootstrap_replicates": int(args.bootstrap),
        "seed": int(args.seed),
        "subsets": {},
    }
    for index, (name, mask) in enumerate(pairing_masks(frame).items()):
        subset = frame.loc[mask].dropna(subset=["dead_365d", "pred_dead_365d"])
        metrics = _METRICS.stratified_patient_bootstrap(
            subset["dead_365d"],
            subset["pred_dead_365d"],
            n_boot=args.bootstrap,
            seed=args.seed + index,
        )
        output["subsets"][name] = {
            "n": int(len(subset)),
            "events": int(subset["dead_365d"].sum()),
            "metrics": metrics,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
