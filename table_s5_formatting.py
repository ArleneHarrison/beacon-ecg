"""Formatting helpers for Supplementary Table S5."""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping


def format_suppressed_prevalence(record: Mapping[str, Any]) -> str:
    """Format prevalence without silently treating a missing value as zero."""

    prevalence = record.get("prevalence")
    if prevalence is not None:
        try:
            value = float(prevalence)
        except (TypeError, ValueError):
            pass
        else:
            if isfinite(value) and 0.0 <= value <= 1.0:
                return f"{100.0 * value:.1f}"

    try:
        n = float(record.get("n"))
        n_events = float(record.get("n_events"))
    except (TypeError, ValueError):
        return "–"

    if (
        not isfinite(n)
        or not isfinite(n_events)
        or n <= 0.0
        or n_events < 0.0
        or n_events > n
    ):
        return "–"

    return f"{100.0 * n_events / n:.1f}"
