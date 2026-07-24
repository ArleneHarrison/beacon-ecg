from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "49_corrected_mortality_pairing_sensitivity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("pairing_sensitivity", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pairing_masks_apply_absolute_window_and_direction():
    module = load_module()
    frame = pd.DataFrame({"days_ecg_to_echo": [-4.0, -1.0, 0.0, 2.0, 5.0]})
    masks = module.pairing_masks(frame)
    assert masks["primary_7_day"].tolist() == [True] * 5
    assert masks["absolute_le_3_days"].tolist() == [False, True, True, True, False]
    assert masks["absolute_le_1_day"].tolist() == [False, True, True, False, False]
    assert masks["echo_before_or_on_ecg"].tolist() == [True, True, True, False, False]
    assert masks["echo_after_or_on_ecg"].tolist() == [False, False, True, True, True]
