from manuscript_number_validation import extract_endpoint_rows, validate_endpoint_rows


def test_extract_endpoint_rows_reads_submission_table():
    manuscript = """
| Endpoint | n | Events | AUROC (95% CI) | AUPRC (95% CI) | Brier score (95% CI) |
|---|---:|---:|---:|---:|---:|
| HFrEF (LVEF ≤40%) | 8,325 | 1,254 | 0.900 (0.890–0.909) | 0.668 (0.642–0.694) | 0.079 (0.075–0.082) |
"""
    rows = extract_endpoint_rows(manuscript)
    assert rows["HFrEF (LVEF ≤40%)"]["n"] == 8325
    assert rows["HFrEF (LVEF ≤40%)"]["events"] == 1254
    assert rows["HFrEF (LVEF ≤40%)"]["auroc"] == 0.900


def test_validate_endpoint_rows_accepts_consistent_rounded_values():
    rows = {
        "HFrEF (LVEF ≤40%)": {
            "n": 8325,
            "events": 1254,
            "auroc": 0.900,
            "auprc": 0.668,
            "brier": 0.079,
        }
    }
    metrics = {
        "hfref_le40": {
            "display_name": "HFrEF (LVEF ≤40%)",
            "n": 8325,
            "events": 1254,
            "metrics": {"auroc": 0.899874, "auprc": 0.667788, "brier": 0.078508},
        }
    }
    assert validate_endpoint_rows(rows, metrics) == []


def test_validate_endpoint_rows_reports_event_mismatch():
    rows = {
        "HFrEF (LVEF ≤40%)": {
            "n": 8325,
            "events": 1253,
            "auroc": 0.900,
            "auprc": 0.668,
            "brier": 0.079,
        }
    }
    metrics = {
        "hfref_le40": {
            "display_name": "HFrEF (LVEF ≤40%)",
            "n": 8325,
            "events": 1254,
            "metrics": {"auroc": 0.899874, "auprc": 0.667788, "brier": 0.078508},
        }
    }
    errors = validate_endpoint_rows(rows, metrics)
    assert errors == ["HFrEF (LVEF ≤40%): events 1253 != 1254"]
