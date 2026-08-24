"""Golden fixture regression tests asserting adapter output against real game XML samples."""

import json
from pathlib import Path

from x4_advisor.ingestion.extractor import process_extracted_directory


def test_golden_fixtures_extracted():
    """Runs extraction adapter against golden XML fixtures and asserts exact entity match."""
    fixture_dir = Path("tests/fixtures/golden_extracted")
    expected_json_path = fixture_dir / "expected_entities.json"

    assert fixture_dir.exists()
    assert expected_json_path.exists()

    with open(expected_json_path, "r", encoding="utf-8") as f:
        expected = json.load(f)

    (
        metadata,
        factions,
        wares,
        sectors,
        sector_resources,
        ships,
        recipes,
        report,
    ) = process_extracted_directory(fixture_dir)

    assert len(factions) == expected["factions_count"]
    assert len(wares) == expected["wares_count"]
    assert len(sectors) == expected["sectors_count"]
    assert len(sector_resources) == expected["sector_resources_count"]
    assert len(ships) == expected["ships_count"]
    assert len(recipes) == expected["recipes_count"]

    sample_ship = ships[0]
    expected_ship = expected["sample_ship"]

    assert sample_ship.id == expected_ship["id"]
    assert sample_ship.name == expected_ship["name"]
    assert sample_ship.ship_class == expected_ship["ship_class"]
    assert sample_ship.hull == expected_ship["hull"]
    assert sample_ship.cargo_capacity == expected_ship["cargo_capacity"]
    assert sample_ship.faction_id == expected_ship["faction_id"]
    assert sample_ship.ware_id == expected_ship["ware_id"]
