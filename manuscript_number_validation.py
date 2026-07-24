"""Validate headline manuscript endpoint numbers against frozen JSON output."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any


DISPLAY_NAMES = {
    "hfref_le40": "HFrEF (LVEF ≤40%)",
    "as_severe": "Severe aortic stenosis",
    "lvh": "LV hypertrophy",
    "dead_365d": "One-year mortality",
}


def _point_estimate(value: Any) -> float:
    if isinstance(value, dict):
        return float(value["estimate"])
    return float(value)


def extract_endpoint_rows(text: str) -> dict[str, dict[str, float | int]]:
    rows: dict[str, dict[str, float | int]] = {}
    pattern = re.compile(
        r"^\|\s*(?P<endpoint>[^|]+?)\s*\|\s*(?P<n>[\d,]+)\s*\|\s*"
        r"(?P<events>[\d,]+)\s*\|\s*(?P<auroc>\d\.\d{3})\s*\([^|]+\)\s*\|\s*"
        r"(?P<auprc>\d\.\d{3})\s*\([^|]+\)\s*\|\s*"
        r"(?P<brier>\d\.\d{3})\s*\([^|]+\)\s*\|$",
        flags=re.MULTILINE,
    )
    for match in pattern.finditer(text):
        rows[match.group("endpoint").strip()] = {
            "n": int(match.group("n").replace(",", "")),
            "events": int(match.group("events").replace(",", "")),
            "auroc": float(match.group("auroc")),
            "auprc": float(match.group("auprc")),
            "brier": float(match.group("brier")),
        }
    return rows


def validate_endpoint_rows(
    rows: dict[str, dict[str, float | int]], metrics: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    for key, record in metrics.items():
        label = record.get("display_name") or DISPLAY_NAMES.get(key, key)
        if label not in rows and key in DISPLAY_NAMES:
            label = DISPLAY_NAMES[key]
        if label not in rows:
            errors.append(f"{label}: row missing")
            continue
        row = rows[label]
        for field in ("n", "events"):
            expected = int(record[field])
            if row[field] != expected:
                errors.append(f"{label}: {field} {row[field]} != {expected}")
        metric_record = record["metrics"]
        for field in ("auroc", "auprc", "brier"):
            expected = round(_point_estimate(metric_record[field]), 3)
            if row[field] != expected:
                errors.append(f"{label}: {field} {row[field]:.3f} != {expected:.3f}")
    return errors


def apply_corrected_mortality(
    metrics: dict[str, dict[str, Any]], corrected: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Replace the legacy mortality endpoint with the aligned follow-up result."""
    updated = copy.deepcopy(metrics)
    updated["dead_365d"] = {
        "display_name": DISPLAY_NAMES["dead_365d"],
        "n": int(corrected["analysis_rows"]),
        "events": int(corrected["analysis_events"]),
        "metrics": corrected["metrics"],
    }
    return updated


def manuscript_format_checks(text: str, allow_pending: bool = False) -> list[str]:
    errors: list[str] = []
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if title_match is None:
        errors.append("title missing")
    elif len(title_match.group(1).split()) > 15:
        errors.append("title exceeds 15 words")

    abstract_match = re.search(
        r"^## Abstract\s*$\n+(.*?)\n+^## Introduction\s*$",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if abstract_match is None:
        errors.append("abstract missing")
    else:
        words = re.findall(r"\S+", abstract_match.group(1))
        if len(words) > 150:
            errors.append(f"abstract exceeds 150 words ({len(words)})")

    if not allow_pending and "PENDING BEFORE FINAL SUBMISSION" in text:
        errors.append("pending final-submission marker remains")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--four-endpoint-json", type=Path, required=True)
    parser.add_argument("--corrected-mortality-json", type=Path)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()

    manuscript = args.manuscript.read_text(encoding="utf-8")
    payload = json.loads(args.four_endpoint_json.read_text(encoding="utf-8"))
    metrics = payload.get("endpoints", payload)
    if args.corrected_mortality_json is not None:
        corrected = json.loads(args.corrected_mortality_json.read_text(encoding="utf-8"))
        metrics = apply_corrected_mortality(metrics, corrected)
    errors = manuscript_format_checks(manuscript, allow_pending=args.allow_pending)
    errors.extend(validate_endpoint_rows(extract_endpoint_rows(manuscript), metrics))
    print(json.dumps({"ok": not errors, "errors": errors}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
