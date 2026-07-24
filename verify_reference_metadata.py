"""Verify manuscript reference metadata against PubMed and Crossref."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from manuscript_reference_audit import parse_reference_entries


USER_AGENT = "BEACON-ECG-reference-audit/1.0 (mailto:corresponding.author@example.org)"


def chunked(values: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [values[index : index + size] for index in range(0, len(values), size)]


def normalise_doi(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    cleaned = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", cleaned)
    return cleaned.rstrip("\t\r\n .,;:)]}")


def _normalise_text(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def parse_declared_metadata(entry: str) -> dict[str, Any]:
    doi_match = re.search(r"\bdoi:\s*(10\.\S+)", entry, flags=re.IGNORECASE)
    pmid_match = re.search(r"\bPMID:\s*(\d+)", entry, flags=re.IGNORECASE)
    year_volume_pages = re.search(
        r"\b((?:19|20)\d{2})\s*;\s*([^:.;\s]+)(?:\([^)]*\))?\s*:\s*([^.;]+)",
        entry,
    )
    year_match = re.search(r"\b((?:19|20)\d{2})\b", entry)
    first_author_match = re.match(r"\s*([^,\s]+)", entry)
    return {
        "first_author": first_author_match.group(1) if first_author_match else None,
        "year": int(year_volume_pages.group(1))
        if year_volume_pages
        else (int(year_match.group(1)) if year_match else None),
        "volume": year_volume_pages.group(2) if year_volume_pages else None,
        "pages": year_volume_pages.group(3).strip() if year_volume_pages else None,
        "doi": normalise_doi(doi_match.group(1)) if doi_match else None,
        "pmid": pmid_match.group(1) if pmid_match else None,
    }


def _fetch_json(url: str, retries: int = 4) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if isinstance(error, urllib.error.HTTPError) and error.code == 404:
                break
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Unable to fetch {url}: {last_error}")


def fetch_pubmed_records(pmids: list[str]) -> dict[str, dict[str, Any]]:
    if not pmids:
        return {}
    records: dict[str, dict[str, Any]] = {}
    for batch_index, batch in enumerate(chunked(pmids, size=20)):
        query = urllib.parse.urlencode(
            {"db": "pubmed", "id": ",".join(batch), "retmode": "json", "version": "2.0"}
        )
        payload = _fetch_json(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + query
        )
        result = payload.get("result", {})
        records.update({pmid: result[pmid] for pmid in batch if pmid in result})
        if batch_index + 1 < len(chunked(pmids, size=20)):
            time.sleep(0.4)
    return records


def fetch_crossref_record(doi: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(doi, safe="")
    payload = _fetch_json(f"https://api.crossref.org/works/{encoded}")
    return payload["message"]


def fetch_datacite_record(doi: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(doi, safe="")
    payload = _fetch_json(f"https://api.datacite.org/dois/{encoded}")
    return payload["data"]["attributes"]


def _pubmed_doi(record: dict[str, Any]) -> str | None:
    for item in record.get("articleids", []):
        if item.get("idtype") == "doi":
            return normalise_doi(item.get("value"))
    return None


def _crossref_year(record: dict[str, Any]) -> int | None:
    for field in ("published-print", "published", "published-online", "issued"):
        parts = record.get(field, {}).get("date-parts", [])
        if parts and parts[0]:
            return int(parts[0][0])
    return None


def verify_manuscript_references(text: str) -> dict[str, Any]:
    entries = parse_reference_entries(text)
    declared = {number: parse_declared_metadata(entry) for number, entry in entries.items()}
    pmids = [data["pmid"] for data in declared.values() if data["pmid"]]
    pubmed = fetch_pubmed_records(pmids)

    rows: list[dict[str, Any]] = []
    for number, entry in entries.items():
        data = declared[number]
        row: dict[str, Any] = {
            "reference": number,
            "declared": data,
            "status": "PASS",
            "issues": [],
            "warnings": [],
        }
        if data["pmid"]:
            record = pubmed.get(data["pmid"])
            if record is None:
                row["issues"].append("PMID not returned by PubMed")
            else:
                authors = record.get("authors") or []
                actual_first = authors[0].get("name", "").split()[0] if authors else None
                actual = {
                    "source": "PubMed",
                    "title": record.get("title"),
                    "first_author": actual_first,
                    "year": int(str(record.get("pubdate", ""))[:4])
                    if str(record.get("pubdate", ""))[:4].isdigit()
                    else None,
                    "volume": record.get("volume") or None,
                    "pages": record.get("pages") or record.get("elocationid") or None,
                    "doi": _pubmed_doi(record),
                    "pmid": data["pmid"],
                }
                row["registry"] = actual
                if _normalise_text(data["first_author"]) != _normalise_text(actual_first):
                    row["issues"].append("first author mismatch")
                if data["year"] != actual["year"]:
                    row["issues"].append("year mismatch")
                if data["volume"] and actual["volume"] and data["volume"] != actual["volume"]:
                    row["issues"].append("volume mismatch")
                if data["doi"] != actual["doi"]:
                    row["issues"].append("DOI mismatch")
        elif data["doi"]:
            try:
                record = fetch_crossref_record(data["doi"])
                registry = "Crossref"
            except RuntimeError:
                try:
                    record = fetch_datacite_record(data["doi"])
                    registry = "DataCite"
                except RuntimeError as error:
                    row["issues"].append(str(error))
                    record = None

            if record is not None and registry == "Crossref":
                authors = record.get("author") or []
                author_families = [author.get("family") for author in authors]
                actual_first = author_families[0] if author_families else None
                actual = {
                    "source": "Crossref",
                    "title": (record.get("title") or [None])[0],
                    "first_author": actual_first,
                    "all_author_families": author_families,
                    "year": _crossref_year(record),
                    "volume": record.get("volume"),
                    "pages": record.get("page") or record.get("article-number"),
                    "doi": normalise_doi(record.get("DOI")),
                }
                row["registry"] = actual
                if actual_first and _normalise_text(data["first_author"]) != _normalise_text(actual_first):
                    if _normalise_text(data["first_author"]) in {
                        _normalise_text(author) for author in author_families
                    }:
                        row["warnings"].append(
                            "registry author order differs; first author requires primary-source verification"
                        )
                    else:
                        row["issues"].append("first author mismatch")
                if data["year"] != actual["year"]:
                    row["issues"].append("year mismatch")
                if data["doi"] != actual["doi"]:
                    row["issues"].append("DOI mismatch")
            elif record is not None:
                creators = record.get("creators") or []
                actual_first = creators[0].get("familyName") if creators else None
                if not actual_first and creators:
                    actual_first = (creators[0].get("name") or "").split(",")[0]
                actual = {
                    "source": "DataCite",
                    "title": ((record.get("titles") or [{}])[0]).get("title"),
                    "first_author": actual_first,
                    "year": record.get("publicationYear"),
                    "publisher": record.get("publisher"),
                    "doi": normalise_doi(record.get("doi")),
                }
                row["registry"] = actual
                if actual_first and _normalise_text(data["first_author"]) != _normalise_text(actual_first):
                    row["issues"].append("first author mismatch")
                if data["year"] != actual["year"]:
                    row["issues"].append("year mismatch")
                if data["doi"] != actual["doi"]:
                    row["issues"].append("DOI mismatch")
        else:
            row["issues"].append("no DOI or PMID declared")

        if row["issues"]:
            row["status"] = "FAIL"
        row["entry"] = entry
        rows.append(row)

    years = [data["year"] for data in declared.values() if data["year"] is not None]
    recent_cutoff = 2021
    recent = sum(year >= recent_cutoff for year in years)
    return {
        "reference_count": len(entries),
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "fail_count": sum(row["status"] == "FAIL" for row in rows),
        "recent_cutoff": recent_cutoff,
        "recent_count": recent,
        "recent_fraction": recent / len(years) if years else None,
        "references": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manuscript", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.manuscript.read_text(encoding="utf-8")
    result = verify_manuscript_references(text)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
