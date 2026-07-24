from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "48_corrected_mortality_comparator.py"


def load_module():
    spec = importlib.util.spec_from_file_location("corrected_mortality_comparator", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_training_cohort_excludes_unobserved_and_replaces_label():
    module = load_module()
    cohort = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "study_id": [11, 22, 33],
            "split": ["train", "train", "test"],
            "is_index_ecg": [1, 1, 1],
            "dead_365d": [1, 0, 0],
        }
    )
    followup = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "study_id": [11, 22, 33],
            "outcome_observed_365d": [True, False, True],
            "corrected_dead_365d": [0.0, np.nan, 1.0],
        }
    )
    prepared, audit = module.prepare_corrected_training_cohort(cohort, followup)
    assert prepared["subject_id"].tolist() == [1]
    assert prepared["dead_365d"].tolist() == [0]
    assert prepared["legacy_dead_365d"].tolist() == [1]
    assert audit["candidate_train_rows"] == 2
    assert audit["observed_train_rows"] == 1
    assert audit["excluded_train_rows"] == 1
    assert audit["legacy_disagreements_among_observed"] == 1
