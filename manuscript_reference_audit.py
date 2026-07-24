"""Deterministic structural audit for numbered manuscript references."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def expand_numeric_citation_group(group: str) -> set[int]:
    """Expand a citation group such as ``1,3,5–7`` into integer identifiers."""

    values: set[int] = set()
    normalized = group.replace("—", "-").replace("–", "-")
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = (part.strip() for part in token.split("-", 1))
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Descending citation range: {token}")
            values.update(range(start, end + 1))
        else:
            values.add(int(token))
    return values


def parse_reference_entries(text: str) -> dict[int, str]:
    """Parse numbered entries from the manuscript's References section."""

    marker = re.search(r"^## References\s*$", text, flags=re.MULTILINE)
    if marker is None:
        return {}
    section = text[marker.end() :]
    next_heading = re.search(r"^##\s+", section, flags=re.MULTILINE)
    if next_heading is not None:
        section = section[: next_heading.start()]

    matches = list(re.finditer(r"^(\d+)\.\s+", section, flags=re.MULTILINE))
    entries: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section)
        number = int(match.group(1))
        entry = " ".join(section[match.end() : end].strip().split())
        entries[number] = entry
    return entries


def reference_set_audit(text: str) -> dict[str, list[int]]:
    """Compare numeric citation identifiers with bibliography entries."""

    references_marker = re.search(r"^## References\s*$", text, flags=re.MULTILINE)
    body = text[: references_marker.start()] if references_marker else text
    cited: set[int] = set()
    for match in re.finditer(r"\^([0-9,\s\-–—]+)\^", body):
        cited.update(expand_numeric_citation_group(match.group(1)))

    bibliography = set(parse_reference_entries(text))
    return {
        "cited": sorted(cited),
        "bibliography": sorted(bibliography),
        "missing_in_bibliography": sorted(cited - bibliography),
        "orphan_in_bibliography": sorted(bibliography - cited),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manuscript", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = args.manuscript.read_text(encoding="utf-8")
    result = reference_set_audit(text)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
