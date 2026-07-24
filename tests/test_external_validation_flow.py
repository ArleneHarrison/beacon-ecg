from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "45_external_validation_reproducible.py"


def load_module():
    spec = importlib.util.spec_from_file_location("external_validation", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_label_mapping_uses_locked_beacon_thresholds():
    module = load_module()
    frame = pd.DataFrame(
        {
            "split": ["test"] * 5,
            "lvef_value": [39.9, 40.0, 40.1, np.nan, 55.0],
            "ivs_measurement": [1.49, 1.50, np.nan, 1.2, np.nan],
            "lvpw_measurement": [1.4, 1.2, 1.51, np.nan, np.nan],
        }
    )
    mapped = module.map_external_labels(frame)
    assert mapped["hfref_le40"].tolist()[:3] == [1.0, 1.0, 0.0]
    assert pd.isna(mapped.loc[3, "hfref_le40"])
    assert mapped["lvh"].tolist()[:3] == [0.0, 1.0, 1.0]
    assert pd.isna(mapped.loc[4, "lvh"])


def test_flow_reconciles_total_and_endpoint_denominators():
    module = load_module()
    frame = pd.DataFrame(
        {
            "split": ["train", "test", "test", "test"],
            "lvef_value": [50.0, 35.0, np.nan, 60.0],
            "ivs_measurement": [1.0, 1.6, np.nan, 1.0],
            "lvpw_measurement": [1.0, 1.0, np.nan, 1.0],
        }
    )
    mapped = module.map_external_labels(frame.loc[frame["split"].eq("test")].copy())
    flow = module.build_flow(frame, mapped)
    assert flow["metadata_total"] == 4
    assert flow["test_split_total"] == 3
    assert flow["endpoints"]["hfref_le40"]["included"] == 2
    assert flow["endpoints"]["hfref_le40"]["missing_label"] == 1
    assert flow["endpoints"]["lvh"]["included"] == 2
    for endpoint in flow["endpoints"].values():
        assert endpoint["included"] + endpoint["missing_label"] == 3


def test_waveform_preprocessing_resamples_and_normalizes_each_lead():
    module = load_module()
    time = np.linspace(0, 10, 2500, endpoint=False)
    wave = np.stack([np.sin(time * (lead + 1)) + lead for lead in range(12)], axis=1)
    prepared = module.prepare_external_waveform(wave[np.newaxis, ...])
    assert prepared.shape == (12, 5000)
    np.testing.assert_allclose(prepared.mean(axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(prepared.std(axis=1), 1.0, atol=1e-4)
    assert np.isfinite(prepared).all()


def test_weighted_ensemble_uses_locked_one_one_two_two_weights():
    module = load_module()
    predictions = {
        "baseline": np.array([[0.1, 0.2]]),
        "ptbxl": np.array([[0.3, 0.4]]),
        "cognition": np.array([[0.5, 0.6]]),
        "vigilance": np.array([[0.7, 0.8]]),
    }
    result = module.weighted_ensemble(predictions)
    expected = (predictions["baseline"] + predictions["ptbxl"] + 2 * predictions["cognition"] + 2 * predictions["vigilance"]) / 6
    np.testing.assert_allclose(result, expected)
