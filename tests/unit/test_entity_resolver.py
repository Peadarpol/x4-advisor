"""Unit tests for EntityResolver using isolated in-memory SQLite database."""

import sqlite3

import pytest

from x4_advisor.retrieval.entity_resolver import EntityResolver
from x4_advisor.retrieval.models import (
    AmbiguousEntityResult,
    EntityNotFoundResult,
    ResolvedEntity,
)
from x4_advisor.storage.models import (
    DatasetMetadata,
    FactionRecord,
    SectorRecord,
    ShipRecord,
    WareRecord,
)
from x4_advisor.storage.schema import init_db_schema


@pytest.fixture
def memory_conn() -> sqlite3.Connection:
    """Provides an isolated, in-memory SQLite database populated with synthetic test entities."""
    conn = sqlite3.connect(":memory:")
    init_db_schema(conn)

    cursor = conn.cursor()
    # Metadata
    cursor.execute(
        "INSERT INTO dataset_metadata (id, game_version, build, extraction_timestamp, is_base_game_only, schema_version) VALUES (1, '9.00', 'b900', '2026-08-24', 1, '1.1.0')"
    )
    # Factions
    cursor.execute("INSERT INTO factions (id, name, short_name) VALUES ('argon', 'Argon Federation', 'ARG')")
    cursor.execute("INSERT INTO factions (id, name, short_name) VALUES ('teladi', 'Teladi Company', 'TEL')")
    # Wares
    cursor.execute("INSERT INTO wares (id, name, category, min_price, avg_price, max_price, volume) VALUES ('claytronics', 'Claytronics', 'tech', 1000, 2000, 3000, 20)")
    cursor.execute("INSERT INTO wares (id, name, category, min_price, avg_price, max_price, volume) VALUES ('energycells', 'Energy Cells', 'energy', 10, 16, 22, 1)")
    # Sectors
    cursor.execute("INSERT INTO sectors (id, name, faction_id, sunlight) VALUES ('sec_argon_prime', 'Argon Prime', 'argon', 1.2)")
    # Ships
    cursor.execute(
        "INSERT INTO ships (id, name, class, hull, shields, cargo_capacity, cargo_type, speed, weapon_slots, turret_slots, shield_slots, purpose, faction_id, ware_id) "
        "VALUES ('ship_arg_m_frigate_01_a_macro', 'Cerberus Vanguard', 'ship_m', 19000.0, 1000.0, 1760.0, 'container', 300.0, 2, 2, 2, 'fight', 'argon', NULL)"
    )
    cursor.execute(
        "INSERT INTO ships (id, name, class, hull, shields, cargo_capacity, cargo_type, speed, weapon_slots, turret_slots, shield_slots, purpose, faction_id, ware_id) "
        "VALUES ('ship_tel_l_miner_solid_01_a_macro', 'Crane Vanguard', 'ship_l', 45000.0, 2500.0, 48000.0, 'solid', 120.0, 0, 4, 3, 'mine', 'teladi', NULL)"
    )
    cursor.execute(
        "INSERT INTO ships (id, name, class, hull, shields, cargo_capacity, cargo_type, speed, weapon_slots, turret_slots, shield_slots, purpose, faction_id, ware_id) "
        "VALUES ('ship_arg_l_miner_solid_01_a_macro', 'Magnetar Vanguard', 'ship_l', 40000.0, 2000.0, 38000.0, 'solid', 140.0, 0, 4, 2, 'mine', 'argon', NULL)"
    )
    conn.commit()
    return conn


def test_resolve_exact_match(memory_conn):
    """Verifies case-insensitive exact name and short_name matching."""
    resolver = EntityResolver(conn=memory_conn)

    # Display name exact
    res1 = resolver.resolve_entity("Cerberus Vanguard")
    assert isinstance(res1, ResolvedEntity)
    assert res1.id == "ship_arg_m_frigate_01_a_macro"
    assert res1.entity_type == "ship"

    # Case-insensitive exact ID
    res2 = resolver.resolve_entity("CLAYTRONICS")
    assert isinstance(res2, ResolvedEntity)
    assert res2.id == "claytronics"
    assert res2.entity_type == "ware"

    # Faction short_name match
    res3 = resolver.resolve_entity("ARG")
    assert isinstance(res3, ResolvedEntity)
    assert res3.id == "argon"
    assert res3.entity_type == "faction"


def test_resolve_partial_match(memory_conn):
    """Verifies single partial substring match."""
    resolver = EntityResolver(conn=memory_conn)

    res = resolver.resolve_entity("Cerberus")
    assert isinstance(res, ResolvedEntity)
    assert res.id == "ship_arg_m_frigate_01_a_macro"


def test_resolve_ambiguous_partial_match(memory_conn):
    """Verifies that multiple partial matches return AmbiguousEntityResult."""
    resolver = EntityResolver(conn=memory_conn)

    res = resolver.resolve_entity("Vanguard")
    assert isinstance(res, AmbiguousEntityResult)
    assert res.query_name == "Vanguard"
    assert len(res.candidates) == 3
    candidate_ids = {c.id for c in res.candidates}
    assert candidate_ids == {
        "ship_arg_m_frigate_01_a_macro",
        "ship_tel_l_miner_solid_01_a_macro",
        "ship_arg_l_miner_solid_01_a_macro",
    }


def test_resolve_entity_types_scoping(memory_conn):
    """Verifies entity_types filtering restricts search tables."""
    resolver = EntityResolver(conn=memory_conn)

    # Argon Prime exists as sector 'sec_argon_prime' and faction 'argon'
    res_ware_only = resolver.resolve_entity("Argon", entity_types=["ware"])
    assert isinstance(res_ware_only, EntityNotFoundResult)

    res_faction_only = resolver.resolve_entity("Argon Federation", entity_types=["faction"])
    assert isinstance(res_faction_only, ResolvedEntity)
    assert res_faction_only.id == "argon"


def test_resolve_wildcard_escaping(memory_conn):
    """Verifies % and _ in search terms do not trigger unescaped SQL LIKE expansion."""
    cursor = memory_conn.cursor()
    cursor.execute("INSERT INTO wares (id, name, category, min_price, avg_price, max_price, volume) VALUES ('special_ware', '50% Off Ware_1', 'special', 1, 2, 3, 1)")
    memory_conn.commit()

    resolver = EntityResolver(conn=memory_conn)
    res = resolver.resolve_entity("50% Off")
    assert isinstance(res, ResolvedEntity)
    assert res.id == "special_ware"


def test_resolve_not_found(memory_conn):
    """Verifies non-existent search terms return EntityNotFoundResult."""
    resolver = EntityResolver(conn=memory_conn)
    res = resolver.resolve_entity("NonExistentShip123")
    assert isinstance(res, EntityNotFoundResult)
    assert res.query_name == "NonExistentShip123"


def test_resolve_invalid_entity_types_raises(memory_conn):
    """Verifies passing invalid entity_types filter values raises ValueError immediately."""
    resolver = EntityResolver(conn=memory_conn)
    with pytest.raises(ValueError, match="Invalid entity_types filter"):
        resolver.resolve_entity("Argon", entity_types=["invalid_type"])


def test_resolve_entity_dedups_ship_wares(memory_conn):
    """Regression test: verifies that when an entity matches both a ship and its corresponding shipyard ware,

    EntityResolver deduplicates in favor of the ship entity and returns ResolvedEntity(entity_type='ship')
    rather than an AmbiguousEntityResult listing both.
    """
    cursor = memory_conn.cursor()
    # Insert a ware for a ship
    cursor.execute(
        "INSERT INTO wares (id, name, category, min_price, avg_price, max_price, volume) "
        "VALUES ('ship_arg_s_fighter_01_a_ware', 'Nova Vanguard', 'ships', 200000, 250000, 300000, 100)"
    )
    # Insert the corresponding ship record linking to the ware_id
    cursor.execute(
        "INSERT INTO ships (id, name, class, hull, shields, cargo_capacity, cargo_type, speed, "
        "weapon_slots, turret_slots, shield_slots, purpose, faction_id, ware_id) "
        "VALUES ('ship_arg_s_fighter_01_a_macro', 'Nova Vanguard', 'ship_s', 3200.0, 400.0, 180.0, 'container', 320.0, "
        "2, 0, 1, 'fight', 'argon', 'ship_arg_s_fighter_01_a_ware')"
    )
    memory_conn.commit()

    resolver = EntityResolver(conn=memory_conn)
    result = resolver.resolve_entity("Nova Vanguard")

    # Must resolve directly to the ship, NOT an AmbiguousEntityResult
    assert isinstance(result, ResolvedEntity), f"Expected ResolvedEntity, got {type(result)}: {result}"
    assert result.entity_type == "ship"
    assert result.id == "ship_arg_s_fighter_01_a_macro"
    assert result.name == "Nova Vanguard"
