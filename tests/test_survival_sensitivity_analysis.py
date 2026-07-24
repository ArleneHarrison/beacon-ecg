from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "44_survival_sensitivity_analysis.py"


def load_module():
    spec = importlib.util.spec_from_file_location("survival_sensitivity", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_echo_normal_excludes_negative_death_times():
    module = load_module()
    frame = pd.DataFrame(
        {
            "lvef_value": [60, 55, 45],
            "pred_hfref_le40": [0.9, 0.1, 0.2],
            "days_to_death": [-2, 100, np.nan],
            "age_at_ecg": [70, 60, 50],
            "sex_male": [1, 0, 1],
        }
    )
    prepared, audit = module.prepare_echo_normal_survival(frame)
    assert len(prepared) == 1
    assert audit["negative_death_times_excluded"] == 1
    assert prepared.iloc[0]["event"] == 1
    assert prepared.iloc[0]["T"] == 100


def test_prepare_echo_normal_uses_corrected_followup_eligibility():
    module = load_module()
    frame = pd.DataFrame(
        {
            "lvef_value": [60, 55, 58],
            "pred_hfref_le40": [0.9, 0.1, 0.2],
            "days_to_death": [50.0, np.nan, np.nan],
            "outcome_observed_365d": [True, False, True],
            "corrected_dead_365d": [1.0, np.nan, 0.0],
            "age_at_ecg": [70, 60, 50],
            "sex_male": [1, 0, 1],
        }
    )
    prepared, audit = module.prepare_echo_normal_survival(frame)
    assert len(prepared) == 2
    assert prepared["event"].tolist() == [1, 0]
    assert prepared["T"].tolist() == [50.0, 365.0]
    assert audit["unobserved_followup_excluded"] == 1
    assert audit["outcome_source"] == "corrected follow-up eligibility"


def test_missingness_table_counts_and_percentages():
    module = load_module()
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [1, 2, 3, 4]})
    table = module.missingness_table(frame, ["a", "b"])
    assert table["a"] == {"missing": 2, "total": 4, "percent": 50.0}
    assert table["b"]["missing"] == 0


def test_piecewise_risk_sets_are_constructed_correctly():
    module = load_module()
    frame = pd.DataFrame(
        {
            "T": [30.0, 100.0, 365.0],
            "event": [1, 1, 0],
            "aiecg_high": [1, 0, 0],
        }
    )
    early, late = module.make_piecewise_risk_sets(frame, cut_day=90, horizon_day=365)
    assert len(early) == 3
    assert int(early["event"].sum()) == 1
    assert sorted(early["T"].tolist()) == [30.0, 90.0, 90.0]
    assert len(late) == 2
    assert int(late["event"].sum()) == 1
    assert sorted(late["T"].tolist()) == [10.0, 275.0]


def test_top_quintile_indicator_has_expected_size_without_ties():
    module = load_module()
    scores = pd.Series(np.arange(100, dtype=float))
    indicator, threshold = module.top_quantile_indicator(scores, quantile=0.8)
    assert indicator.sum() == 20
    assert threshold == np.quantile(scores, 0.8)
