from __future__ import annotations

import pytest


def test_suppressed_cell_reconstructs_prevalence_from_events_and_n():
    from table_s5_formatting import format_suppressed_prevalence

    assert format_suppressed_prevalence({"n": 1, "n_events": 1}) == "100.0"
    assert format_suppressed_prevalence({"n": 4, "n_events": 1}) == "25.0"


def test_explicit_prevalence_is_used_when_present():
    from table_s5_formatting import format_suppressed_prevalence

    assert format_suppressed_prevalence({"n": 10, "n_events": 2, "prevalence": 0.3}) == "30.0"


@pytest.mark.parametrize("record", [{"n": 0, "n_events": 0}, {"n": 3}, {}])
def test_unrecoverable_prevalence_is_not_rendered_as_zero(record):
    from table_s5_formatting import format_suppressed_prevalence

    assert format_suppressed_prevalence(record) == "–"
