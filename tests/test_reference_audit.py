from manuscript_reference_audit import (
    expand_numeric_citation_group,
    parse_reference_entries,
    reference_set_audit,
)


def test_expand_numeric_citation_group_handles_commas_and_ranges():
    assert expand_numeric_citation_group("1,3,5–7, 10-11") == {1, 3, 5, 6, 7, 10, 11}


def test_parse_reference_entries_reads_numbered_markdown_entries():
    text = """## References
1. First article. doi:10.1/a.
2. Second article. PMID:123.

## Acknowledgements
"""
    assert parse_reference_entries(text) == {
        1: "First article. doi:10.1/a.",
        2: "Second article. PMID:123.",
    }


def test_reference_set_audit_reports_missing_and_orphan_entries():
    text = """Claim.^1,3–4^

## References
1. Used.
2. Orphan.
3. Used.
"""
    audit = reference_set_audit(text)
    assert audit["cited"] == [1, 3, 4]
    assert audit["bibliography"] == [1, 2, 3]
    assert audit["missing_in_bibliography"] == [4]
    assert audit["orphan_in_bibliography"] == [2]
