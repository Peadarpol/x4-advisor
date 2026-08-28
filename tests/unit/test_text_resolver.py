"""Unit tests for text catalog localization resolver."""

from pathlib import Path

from x4_advisor.ingestion.text_resolver import TextResolver


def test_text_resolver_clean_text():
    """Tests text cleaning for plain strings and unescaping."""
    resolver = TextResolver()
    assert resolver._clean_text("Energy Cells") == "Energy Cells"
    assert resolver._clean_text("Magnetar \\(Gas\\)") == "Magnetar (Gas)"
    assert resolver._clean_text("(Allographyne Scrap Processor)") == "(Allographyne Scrap Processor)"
    assert resolver._clean_text("") == ""


def test_text_resolver_two_pass_and_compound_resolution(tmp_path: Path):
    """Tests loading XML text catalog and resolving complex, compound, and forward references."""
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
<language id="44">
  <page id="20003">
    <t id="10001">Grand Exchange</t>
  </page>
  <page id="20004">
    <t id="10011">{20003,10001} {20402,1}(Grand Exchange I)</t>
  </page>
  <page id="20101">
    <t id="10601">Cerberus Vanguard</t>
    <t id="10602">(Cerberus Vanguard){20101,10601}</t>
    <t id="10603">Cerberus Sentinel</t>
    <t id="11101">(Magnetar \\(Gas\\)){20101,10601}</t>
  </page>
  <page id="20201">
    <t id="10201">{20201,10203} \\({30000,101}\\)</t>
    <t id="10203">Digital Seminar</t>
    <t id="50000">(Allographyne Scrap Processor)</t>
    <t id="60000">The Cerberus {20101,10601} is a frigate (medium class) built by Argon.</t>
  </page>
  <page id="20402">
    <t id="1">I</t>
  </page>
  <!-- Forward referenced page that appears after page 20201 in document order -->
  <page id="30000">
    <t id="101">Boarding</t>
  </page>
  <!-- Recursive cycle test -->
  <page id="40000">
    <t id="1">{40000,2}</t>
    <t id="2">{40000,1}</t>
  </page>
</language>
"""
    lang_file = tmp_path / "0001-l044.xml"
    lang_file.write_text(xml_content, encoding="utf-8")

    resolver = TextResolver()
    resolver.load_from_file(lang_file)

    # 1. Paren-first title annotation
    assert resolver.resolve("{20101,10602}") == "Cerberus Vanguard"
    assert resolver.resolve("{20101, 10603}") == "Cerberus Sentinel"
    assert resolver.resolve("{20101,11101}") == "Magnetar (Gas)"

    # 2. Ref-first with trailing comment
    assert resolver.resolve("{20004,10011}") == "Grand Exchange I"

    # 3. Compound forward reference across pages
    assert resolver.resolve("{20201,10201}") == "Digital Seminar (Boarding)"

    # 4. Bare placeholder preservation
    assert resolver.resolve("{20201,50000}") == "(Allographyne Scrap Processor)"

    # 5. Ref-bearing prose description with mid-string paren
    assert resolver.resolve("{20201,60000}") == "The Cerberus Cerberus Vanguard is a frigate (medium class) built by Argon."

    # 6. Cycle detection safety
    assert resolver.resolve("{40000,1}") == ""

    # 7. Fallback behavior
    assert resolver.resolve("Plain Text") == "Plain Text"
    assert resolver.resolve("{99999,99999}", default="Fallback") == "Fallback"
