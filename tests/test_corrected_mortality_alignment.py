from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "46_align_corrected_mortality.py"


def load_module():
    spec = importlib.util.spec_from_file_location("corrected_mortality_alignment", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "study_id": [11, 22, 33],
            "dead_365d": [1, 0, 1],
            "pred_dead_365d": [0.9, 0.2, 0.3],
            "ok": [True, True, True],
        }
    )


def make_followup() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "study_id": [11, 22, 33],
            "documented_followup_days": [50.0, 100.0, 500.0],
            "outcome_observed_365d": [True, False, True],
            "corrected_dead_365d": [1.0, np.nan, 0.0],
            "exclusion_reason": ["", "insufficient_followup", ""],
        }
    )


def test_alignment_excludes_unobserved_and_preserves_legacy_label():
    module = load_module()
    corrected, audit = module.align_corrected_mortality(
        make_predictions(), make_followup()
    )
    assert corrected["subject_id"].tolist() == [1, 3]
    assert corrected["dead_365d"].tolist() == [1, 0]
    assert corrected["legacy_dead_365d"].tolist() == [1, 1]
    assert audit["prediction_rows"] == 3
    assert audit["matched_rows"] == 3
    assert audit["observed_rows"] == 2
    assert audit["excluded_rows"] == 1
    assert audit["corrected_events"] == 1
    assert audit["legacy_disagreements_among_observed"] == 1


def test_alignment_rejects_duplicate_followup_keys():
    module = load_module()
    duplicated = pd.concat([make_followup(), make_followup().iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate follow-up key"):
        module.align_corrected_mortality(make_predictions(), duplicated)


def test_alignment_rejects_unmatched_prediction_key():
    module = load_module()
    incomplete = make_followup().iloc[:2].copy()
    with pytest.raises(ValueError, match="prediction rows lacked follow-up alignment"):
        module.align_corrected_mortality(make_predictions(), incomplete)


def test_expected_calibration_error_uses_equal_width_bins():
    module = load_module()
    assert module.expected_calibration_error([0, 1], [0.1, 0.9], bins=10) == pytest.approx(0.1)
