"""Unit tests for StructuredQueryEngine using isolated in-memory SQLite database."""

import sqlite3

import pytest

from x4_advisor.retrieval.models import (
    CategoryListResult,
    ProductionChainResult,
    RankingResult,
    SingleEntityResult,
)
from x4_advisor.retrieval.structured_query import StructuredQueryEngine
from x4_advisor.storage.schema import init_db_schema


@pytest.fixture
def memory_conn() -> sqlite3.Connection:
    """Provides an isolated, in-memory SQLite database populated with synthetic test data."""
    conn = sqlite3.connect(":memory:")
    init_db_schema(conn)

    cursor = conn.cursor()
    # Metadata
    cursor.execute(
        "INSERT INTO dataset_metadata (id, game_version, build, extraction_timestamp, is_base_game_only, schema_version) VALUES (1, '9.00', 'b900', '2026-08-24', 1, '1.1.0')"
    )
    # Factions
    cursor.execute("INSERT INTO factions (id, name, short_name, relations_summary) VALUES ('argon', 'Argon Federation', 'ARG', 'Friendly')")
    cursor.execute("INSERT INTO factions (id, name, short_name, relations_summary) VALUES ('teladi', 'Teladi Company', 'TEL', 'Neutral')")

    # Wares
    cursor.execute("INSERT INTO wares (id, name, category, min_price, avg_price, max_price, volume) VALUES ('claytronics', 'Claytronics', 'tech', 1000, 2000, 3000, 20)")
    cursor.execute("INSERT INTO wares (id, name, category, min_price, avg_price, max_price, volume) VALUES ('quantumtubes', 'Quantum Tubes', 'tech', 200, 300, 400, 10)")
    cursor.execute("INSERT INTO wares (id, name, category, min_price, avg_price, max_price, volume) VALUES ('energycells', 'Energy Cells', 'energy', 10, 16, 22, 1)")
    cursor.execute("INSERT INTO wares (id, name, category, min_price, avg_price, max_price, volume) VALUES ('ore', 'Ore', 'minerals', 40, 50, 60, 10)")

    # Sectors & Yields
    cursor.execute("INSERT INTO sectors (id, name, faction_id, sunlight) VALUES ('sec_argon_prime', 'Argon Prime', 'argon', 1.2)")
    cursor.execute("INSERT INTO sectors (id, name, faction_id, sunlight) VALUES ('sec_grand_exchange', 'Grand Exchange', 'teladi', 1.0)")
    cursor.execute("INSERT INTO sector_resources (sector_id, resource_id, yield) VALUES ('sec_argon_prime', 'ore', 4.5)")
    cursor.execute("INSERT INTO sector_resources (sector_id, resource_id, yield) VALUES ('sec_grand_exchange', 'ore', 9.0)")

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

    # Production Recipes: claytronics requires 2 quantumtubes + 10 energycells; quantumtubes requires 5 energycells + 2 ore
    cursor.execute("INSERT INTO production_recipes (ware_id, method, input_ware_id, input_amount, output_amount, production_time) VALUES ('claytronics', 'default', 'quantumtubes', 2, 1, 60.0)")
    cursor.execute("INSERT INTO production_recipes (ware_id, method, input_ware_id, input_amount, output_amount, production_time) VALUES ('claytronics', 'default', 'energycells', 10, 1, 60.0)")
    cursor.execute("INSERT INTO production_recipes (ware_id, method, input_ware_id, input_amount, output_amount, production_time) VALUES ('quantumtubes', 'default', 'energycells', 5, 1, 30.0)")
    cursor.execute("INSERT INTO production_recipes (ware_id, method, input_ware_id, input_amount, output_amount, production_time) VALUES ('quantumtubes', 'default', 'ore', 2, 1, 30.0)")

    conn.commit()
    return conn


def test_t1_fact_lookup(memory_conn):
    """Verifies Template T1 single-entity fact lookup across domain entities."""
    with StructuredQueryEngine(conn=memory_conn) as engine:
        # Ship lookup
        ship_res = engine.query_t1_fact_lookup("ship_arg_m_frigate_01_a_macro")
        assert isinstance(ship_res, SingleEntityResult)
        assert ship_res.entity_type == "ship"
        assert ship_res.data["name"] == "Cerberus Vanguard"
        assert ship_res.data["cargo_capacity"] == 1760.0
        assert ship_res.data["purpose"] == "fight"

        # Ware lookup
        ware_res = engine.query_t1_fact_lookup("claytronics")
        assert isinstance(ware_res, SingleEntityResult)
        assert ware_res.entity_type == "ware"
        assert ware_res.data["avg_price"] == 2000

        # Sector lookup
        sector_res = engine.query_t1_fact_lookup("sec_argon_prime")
        assert isinstance(sector_res, SingleEntityResult)
        assert sector_res.entity_type == "sector"
        assert sector_res.data["resource_yields"] == {"ore": 4.5}

        # Non-existent lookup
        null_res = engine.query_t1_fact_lookup("non_existent_id")
        assert null_res is None


def test_t2_ranking_canonical_miners(memory_conn):
    """Verifies T2 ranking query for 'Which L-class miners have the most cargo?'."""
    with StructuredQueryEngine(conn=memory_conn) as engine:
        res = engine.query_t2_ranking(
            category_or_class="ship_l",
            metric="cargo_capacity",
            purpose="mine",
            limit=5,
        )
        assert isinstance(res, RankingResult)
        assert len(res.items) == 2
        # Crane Vanguard (48,000) > Magnetar Vanguard (38,000)
        assert res.items[0].id == "ship_tel_l_miner_solid_01_a_macro"
        assert res.items[0].value == 48000.0
        assert res.items[1].id == "ship_arg_l_miner_solid_01_a_macro"
        assert res.items[1].value == 38000.0


def test_t2_unwhitelisted_metric_raises_error(memory_conn):
    """Verifies that passing unwhitelisted SQL expressions to metric raises ValueError."""
    with StructuredQueryEngine(conn=memory_conn) as engine:
        with pytest.raises(ValueError, match="Invalid or unwhitelisted metric"):
            engine.query_t2_ranking(category_or_class="ship_l", metric="cargo_capacity; DROP TABLE ships;")


def test_t2_sector_yield_ranking(memory_conn):
    """Verifies T2 sector resource yield ranking."""
    with StructuredQueryEngine(conn=memory_conn) as engine:
        res = engine.query_t2_sector_yield_ranking("ore", limit=5)
        assert isinstance(res, RankingResult)
        assert len(res.items) == 2
        # Grand Exchange (9.0) > Argon Prime (4.5)
        assert res.items[0].id == "sec_grand_exchange"
        assert res.items[0].value == 9.0
        assert res.items[1].id == "sec_argon_prime"


def test_t3_production_chain_traversal(memory_conn):
    """Verifies Template T3 multi-tier production recipe tree and total raw material aggregation."""
    with StructuredQueryEngine(conn=memory_conn) as engine:
        res = engine.query_t3_production_chain("claytronics", method="default")
        assert isinstance(res, ProductionChainResult)
        assert res.target_ware_id == "claytronics"
        assert res.method == "default"

        # Check total raw material counts
        # 1 claytronics needs: 10 energycells (direct) + 2 quantumtubes * (5 energycells + 2 ore)
        # Total energycells = 10 + 10 = 20; Total ore = 4
        assert res.total_raw_materials["energycells"] == 20
        assert res.total_raw_materials["ore"] == 4


def test_t3_multi_method_fallback(memory_conn):
    """Verifies T3 falls back to first available recipe method when requested method does not exist."""
    cursor = memory_conn.cursor()
    cursor.execute("INSERT INTO wares (id, name, category, min_price, avg_price, max_price, volume) VALUES ('terran_ware', 'Terran Ware', 'tech', 10, 20, 30, 1)")
    cursor.execute("INSERT INTO production_recipes (ware_id, method, input_ware_id, input_amount, output_amount, production_time) VALUES ('terran_ware', 'terran', 'energycells', 5, 1, 10.0)")
    memory_conn.commit()

    with StructuredQueryEngine(conn=memory_conn) as engine:
        # Request 'default' method, but ware only has 'terran' method
        res = engine.query_t3_production_chain("terran_ware", method="default")
        assert isinstance(res, ProductionChainResult)
        assert res.method == "terran"
        assert res.was_method_fallback is True
        assert res.requested_method == "default"


def test_t3_no_fallback_when_requested_method_exists(memory_conn):
    """Verifies T3 sets was_method_fallback=False when requested recipe method exists directly."""
    with StructuredQueryEngine(conn=memory_conn) as engine:
        res = engine.query_t3_production_chain("claytronics", method="default")
        assert isinstance(res, ProductionChainResult)
        assert res.method == "default"
        assert res.was_method_fallback is False
        assert res.requested_method == "default"


def test_t3_cycle_detection(memory_conn):
    """Verifies cycle detection prevents stack overflow on recursive recipe graphs."""
    cursor = memory_conn.cursor()
    cursor.execute("INSERT INTO wares (id, name, category, min_price, avg_price, max_price, volume) VALUES ('ware_a', 'Ware A', 'tech', 1, 1, 1, 1)")
    cursor.execute("INSERT INTO wares (id, name, category, min_price, avg_price, max_price, volume) VALUES ('ware_b', 'Ware B', 'tech', 1, 1, 1, 1)")
    # A requires B, B requires A (circular dependency)
    cursor.execute("INSERT INTO production_recipes (ware_id, method, input_ware_id, input_amount, output_amount, production_time) VALUES ('ware_a', 'default', 'ware_b', 1, 1, 10.0)")
    cursor.execute("INSERT INTO production_recipes (ware_id, method, input_ware_id, input_amount, output_amount, production_time) VALUES ('ware_b', 'default', 'ware_a', 1, 1, 10.0)")
    memory_conn.commit()

    with StructuredQueryEngine(conn=memory_conn) as engine:
        res = engine.query_t3_production_chain("ware_a", method="default")
        assert isinstance(res, ProductionChainResult)
        assert res.target_ware_id == "ware_a"


def test_t4_category_listing(memory_conn):
    """Verifies Template T4 category listing by faction, ship class, purpose, and ware group."""
    with StructuredQueryEngine(conn=memory_conn) as engine:
        # Faction listing
        res_fact = engine.query_t4_category_listing("faction", "argon")
        assert isinstance(res_fact, CategoryListResult)
        assert len(res_fact.items) == 2  # Cerberus Vanguard & Magnetar Vanguard

        # Ship class listing
        res_class = engine.query_t4_category_listing("ship_class", "ship_l")
        assert isinstance(res_class, CategoryListResult)
        assert len(res_class.items) == 2

        # Purpose listing
        res_purpose = engine.query_t4_category_listing("purpose", "mine")
        assert len(res_purpose.items) == 2

        # Ware group listing
        res_ware = engine.query_t4_category_listing("ware_group", "tech")
        assert len(res_ware.items) == 2


def test_pragma_query_only_blocks_writes(memory_conn):
    """Verifies that PRAGMA query_only = ON causes write queries to fail with OperationalError."""
    with StructuredQueryEngine(conn=memory_conn) as engine:
        with pytest.raises(sqlite3.OperationalError, match="attempt to write a readonly database"):
            engine.conn.execute("CREATE TABLE test_table (id INT)")
