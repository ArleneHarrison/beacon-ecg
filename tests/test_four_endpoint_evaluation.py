from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "41_rerun_four_endpoint_evaluation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("four_endpoint_evaluation", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_predictions() -> pd.DataFrame:
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    frame = pd.DataFrame({"subject_id": np.arange(1001, 1007), "ok": True})
    for label_col, prediction_col in (
        ("hfref_le40", "pred_hfref_le40"),
        ("as_severe", "pred_as_severe"),
        ("lvh", "pred_lvh"),
        ("dead_365d", "pred_dead_365d"),
    ):
        frame[label_col] = labels
        frame[prediction_col] = np.array([0.05, 0.15, 0.30, 0.70, 0.85, 0.95])
    return frame


def test_endpoint_spec_is_exactly_the_four_submission_endpoints():
    module = load_module()
    assert list(module.ENDPOINTS) == ["hfref_le40", "as_severe", "lvh", "dead_365d"]
    assert module.ENDPOINTS["dead_365d"]["display_name"] == "One-year mortality"


def test_evaluate_all_endpoints_returns_counts_and_metrics():
    module = load_module()
    result = module.evaluate_all_endpoints(make_predictions(), n_boot=40, seed=20260723)
    assert set(result["endpoints"]) == set(module.ENDPOINTS)
    for endpoint in result["endpoints"].values():
        assert endpoint["n"] == 6
        assert endpoint["events"] == 3
        assert endpoint["prevalence"] == 0.5
        assert endpoint["metrics"]["auroc"]["estimate"] == 1.0
        assert endpoint["bootstrap_replicates"] == 40


def test_evaluation_filters_missing_labels_and_failed_waveforms():
    module = load_module()
    frame = make_predictions()
    frame.loc[0, "hfref_le40"] = np.nan
    frame.loc[1, "ok"] = False
    result = module.evaluate_all_endpoints(frame, n_boot=20, seed=20260723)
    assert result["endpoints"]["hfref_le40"]["n"] == 4
    assert result["endpoints"]["as_severe"]["n"] == 5


def test_same_seed_produces_byte_identical_json(tmp_path):
    module = load_module()
    first = module.evaluate_all_endpoints(make_predictions(), n_boot=30, seed=20260723)
    second = module.evaluate_all_endpoints(make_predictions(), n_boot=30, seed=20260723)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    module.write_json(first, first_path)
    module.write_json(second, second_path)
    assert first_path.read_bytes() == second_path.read_bytes()
    parsed = json.loads(first_path.read_text(encoding="utf-8"))
    assert parsed["seed"] == 20260723
