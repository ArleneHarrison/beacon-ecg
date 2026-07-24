from verify_reference_metadata import normalise_doi, parse_declared_metadata


def test_normalise_doi_strips_url_prefix_and_terminal_punctuation():
    assert normalise_doi("https://doi.org/10.1000/Test.1).") == "10.1000/test.1"


def test_parse_declared_metadata_extracts_core_fields():
    entry = (
        "Attia ZI, Kapa S, Lopez-Jimenez F, et al. Example title. "
        "*Nature Medicine* 2019;25:70–74. doi:10.1038/example. PMID:30617318."
    )
    metadata = parse_declared_metadata(entry)
    assert metadata["first_author"] == "Attia"
    assert metadata["year"] == 2019
    assert metadata["volume"] == "25"
    assert metadata["pages"] == "70–74"
    assert metadata["doi"] == "10.1038/example"
    assert metadata["pmid"] == "30617318"
