"""Unit tests for domain invariant validation engine."""

from x4_advisor.ingestion.validator import DomainValidator
from x4_advisor.storage.models import (
    FactionRecord,
    ProductionRecipeRecord,
    SectorRecord,
    SectorResourceRecord,
    ShipRecord,
    WareRecord,
)


def test_validator_valid_dataset():
    """Validates that valid records pass all invariant checks."""
    factions = [FactionRecord(id="argon", name="Argon Federation")]
    wares = [
        WareRecord(
            id="energycells",
            name="Energy Cells",
            category="energy",
            min_price=10,
            avg_price=16,
            max_price=22,
            volume=1,
        ),
        WareRecord(
            id="graphene",
            name="Graphene",
            category="refined",
            min_price=100,
            avg_price=150,
            max_price=200,
            volume=20,
        ),
        WareRecord(
            id="ship_arg_m_frigate_01_a",
            name="Cerberus Ware",
            category="ship",
            min_price=1000000,
            avg_price=1500000,
            max_price=2000000,
            volume=1,
        ),
    ]
    sectors = [SectorRecord(id="arg_prime", name="Argon Prime", faction_id="argon", sunlight=1.2)]
    resources = [SectorResourceRecord(sector_id="arg_prime", resource_id="ore", resource_yield=1.5)]
    ships = [
        ShipRecord(
            id="ship_arg_m_frigate_01_a_macro",
            name="Cerberus Vanguard",
            ship_class="ship_m",
            hull=19000.0,
            shields=1000.0,
            cargo_capacity=1760.0,
            speed=300.0,
            faction_id="argon",
            ware_id="ship_arg_m_frigate_01_a",
        )
    ]
    recipes = [
        ProductionRecipeRecord(
            ware_id="graphene",
            method="default",
            input_ware_id="energycells",
            input_amount=20,
            output_amount=40,
            production_time=120.0,
        )
    ]

    validator = DomainValidator()
    v_fact, v_ware, v_sec, v_res, v_ship, v_rec, report = validator.validate_dataset(
        factions, wares, sectors, resources, ships, recipes
    )

    assert len(v_fact) == 1
    assert len(v_ware) == 3
    assert len(v_sec) == 1
    assert len(v_res) == 1
    assert len(v_ship) == 1
    assert len(v_rec) == 1
    assert report.total_skipped == 0


def test_validator_unknown_ship_class_logs_warning_and_inserts():
    """Verifies that an unknown ship class logs a warning AND STILL INSERTS the record (does not skip)."""
    factions = [FactionRecord(id="argon", name="Argon Federation")]
    ships = [
        ShipRecord(
            id="ship_custom_macro",
            name="Custom Experimental Ship",
            ship_class="ship_experimental_unknown",
            hull=10000.0,
            shields=500.0,
            cargo_capacity=500.0,
            speed=200.0,
            faction_id="argon",
        )
    ]

    validator = DomainValidator()
    _, _, _, _, v_ships, _, report = validator.validate_dataset(
        factions, [], [], [], ships, []
    )

    assert len(v_ships) == 1
    assert v_ships[0].id == "ship_custom_macro"
    assert report.total_skipped == 0
    assert len(report.warnings) == 1
    assert "unverified class" in report.warnings[0]


def test_validator_skips_invalid_records():
    """Verifies rejection of negative bounds, missing FK references, and self-referencing recipes."""
    factions = [FactionRecord(id="argon", name="Argon Federation")]
    wares = [
        WareRecord(
            id="energycells",
            name="Energy Cells",
            category="energy",
            min_price=10,
            avg_price=16,
            max_price=22,
            volume=1,
        )
    ]

    # Invalid ship (zero hull)
    bad_ship = ShipRecord(
        id="bad_ship_macro",
        name="Bad Ship",
        ship_class="ship_m",
        hull=0.0,
        shields=0.0,
        cargo_capacity=0.0,
    )

    # Invalid recipe (self referencing)
    self_ref_recipe = ProductionRecipeRecord(
        ware_id="energycells",
        method="default",
        input_ware_id="energycells",
        input_amount=10,
        output_amount=10,
        production_time=60.0,
    )

    # Invalid recipe (referencing non-existent ware)
    missing_fk_recipe = ProductionRecipeRecord(
        ware_id="non_existent_ware",
        method="default",
        input_ware_id="energycells",
        input_amount=10,
        output_amount=10,
        production_time=60.0,
    )

    validator = DomainValidator()
    _, _, _, _, v_ships, v_recipes, report = validator.validate_dataset(
        factions, wares, [], [], [bad_ship], [self_ref_recipe, missing_fk_recipe]
    )

    assert len(v_ships) == 0
    assert len(v_recipes) == 0
    assert report.total_skipped == 3
