"""Unit tests for normalization adapter."""

from pathlib import Path

from x4_advisor.ingestion.adapter import NormalizationAdapter
from x4_advisor.ingestion.text_resolver import TextResolver


def test_adapter_parse_wares_and_recipes(tmp_path: Path):
    """Tests parsing raw wares XML into WareRecords and ProductionRecipeRecords."""
    wares_xml = """<?xml version="1.0" encoding="utf-8"?>
<wares>
  <ware id="energycells" name="Energy Cells" group="energy" volume="1">
    <price min="10" average="16" max="22" />
    <production time="60" amount="100" method="default" />
  </ware>
  <ware id="graphene" name="Graphene" group="refined" volume="20">
    <price min="100" average="150" max="200" />
    <production time="120" amount="40" method="default">
      <primary>
        <ware ware="energycells" amount="20" />
      </primary>
    </production>
  </ware>
</wares>
"""
    wares_file = tmp_path / "wares.xml"
    wares_file.write_text(wares_xml, encoding="utf-8")

    resolver = TextResolver()
    adapter = NormalizationAdapter(resolver)

    wares, recipes, macro_map = adapter.parse_wares_and_recipes(wares_file)

    assert len(wares) == 2
    assert wares[0].id == "energycells"
    assert wares[0].avg_price == 16
    assert wares[1].id == "graphene"

    assert len(recipes) == 1
    assert recipes[0].ware_id == "graphene"
    assert recipes[0].input_ware_id == "energycells"
    assert recipes[0].input_amount == 20
    assert recipes[0].output_amount == 40


def test_adapter_parse_factions(tmp_path: Path):
    """Tests parsing raw factions XML into FactionRecords."""
    factions_xml = """<?xml version="1.0" encoding="utf-8"?>
<factions>
  <faction id="argon" name="Argon Federation" shortname="Argon" description="Argon faction info" />
</factions>
"""
    factions_file = tmp_path / "factions.xml"
    factions_file.write_text(factions_xml, encoding="utf-8")

    resolver = TextResolver()
    adapter = NormalizationAdapter(resolver)

    factions = adapter.parse_factions(factions_file)
    assert len(factions) == 1
    assert factions[0].id == "argon"
    assert factions[0].name == "Argon Federation"
    assert factions[0].short_name == "Argon"
