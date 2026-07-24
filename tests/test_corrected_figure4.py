from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "47_make_corrected_figure4.py"


def load_module():
    spec = importlib.util.spec_from_file_location("corrected_figure4", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_survival_groups_uses_corrected_event_and_full_followup():
    module = load_module()
    frame = pd.DataFrame(
        {
            "lvef_value": [60, 60, 60, 60, 45],
            "pred_hfref_le40": [0.1, 0.2, 0.3, 0.9, 0.8],
            "corrected_dead_365d": [0, 1, 0, 1, 1],
            "outcome_observed_365d": [True, True, True, True, True],
            "days_to_death": [np.nan, 40.0, np.nan, 20.0, 10.0],
        }
    )
    prepared, summary = module.prepare_survival_groups(frame, quantile=0.75)
    assert len(prepared) == 4
    assert prepared["group"].tolist() == ["remainder", "remainder", "remainder", "top quintile"]
    assert prepared["event"].tolist() == [0, 1, 0, 1]
    assert prepared["time"].tolist() == [365.0, 40.0, 365.0, 20.0]
    assert summary["high_n"] == 1
    assert summary["low_n"] == 3
    assert summary["high_event_rate"] == 1.0
    assert summary["low_event_rate"] == 1 / 3
