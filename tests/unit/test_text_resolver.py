"""Unit tests for text catalog localization resolver."""

from pathlib import Path

from x4_advisor.ingestion.text_resolver import TextResolver


def test_text_resolver_clean_text():
    """Tests text cleaning for X4 comment formats."""
    resolver = TextResolver()
    assert resolver._clean_text("(Cerberus Vanguard){20101,10601}") == "Cerberus Vanguard"
    assert resolver._clean_text("Energy Cells") == "Energy Cells"
    assert resolver._clean_text("") == ""


def test_text_resolver_load_and_resolve(tmp_path: Path):
    """Tests loading XML text catalog and resolving text references."""
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
<language id="44">
  <page id="20101">
    <t id="10602">(Cerberus Vanguard){20101,10601}</t>
    <t id="10603">Cerberus Sentinel</t>
  </page>
</language>
"""
    lang_file = tmp_path / "0001-l044.xml"
    lang_file.write_text(xml_content, encoding="utf-8")

    resolver = TextResolver()
    resolver.load_from_file(lang_file)

    assert resolver.resolve("{20101,10602}") == "Cerberus Vanguard"
    assert resolver.resolve("{20101, 10603}") == "Cerberus Sentinel"
    assert resolver.resolve("Plain Text") == "Plain Text"
    assert resolver.resolve("{99999,99999}", default="Fallback") == "Fallback"
