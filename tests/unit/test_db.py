"""Unit tests for SQLite database schema, foreign key enforcement, and atomic swap."""

import sqlite3

import pytest

from x4_advisor.storage.db import (
    atomic_ingest_to_db,
    get_connection,
    insert_domain_data,
)
from x4_advisor.storage.models import (
    DatasetMetadata,
    FactionRecord,
    ProductionRecipeRecord,
    SectorRecord,
    SectorResourceRecord,
    ShipRecord,
    WareRecord,
)
from x4_advisor.storage.schema import init_db_schema


def test_get_connection_parent_directory_creation(tmp_path):
    """Verifies that get_connection creates missing parent directories automatically."""
    deep_path = tmp_path / "nested" / "dir" / "test.db"
    assert not deep_path.parent.exists()

    conn = get_connection(deep_path)
    assert deep_path.parent.exists()
    assert deep_path.exists()
    conn.close()


def test_single_row_dataset_metadata_constraint(tmp_path):
    """Verifies single-row metadata constraint id=1."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_db_schema(conn)

    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO dataset_metadata (id, game_version, build, extraction_timestamp, is_base_game_only, schema_version)
        VALUES (1, '7.0', '123', '2026-08-24', 1, '1.0.0')
        """
    )
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute(
            """
            INSERT INTO dataset_metadata (id, game_version, build, extraction_timestamp, is_base_game_only, schema_version)
            VALUES (2, '7.0', '123', '2026-08-24', 1, '1.0.0')
            """
        )

    conn.close()


def test_foreign_key_dependency_insertion_order(tmp_path):
    """Verifies batch insertion respects dependency ordering and FK constraints."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_db_schema(conn)

    metadata = DatasetMetadata("7.0", "123", "2026-08-24", True)
    factions = [FactionRecord("argon", "Argon Federation")]
    wares = [
        WareRecord("energycells", "Energy Cells", "energy", 10, 16, 22, 1),
        WareRecord("graphene", "Graphene", "refined", 100, 150, 200, 20),
    ]
    sectors = [SectorRecord("arg_prime", "Argon Prime", "argon", 1.2)]
    resources = [SectorResourceRecord("arg_prime", "ore", 1.5)]
    ships = [
        ShipRecord(
            id="ship_arg_m_frigate_01_a_macro",
            name="Cerberus Vanguard",
            ship_class="ship_m",
            hull=19000.0,
            shields=1000.0,
            cargo_capacity=1760.0,
            cargo_type="container",
            speed=300.0,
            weapon_slots=2,
            turret_slots=2,
            shield_slots=2,
            purpose="fight",
            faction_id="argon",
            ware_id="energycells",
        )
    ]
    recipes = [
        ProductionRecipeRecord(
            "graphene", "default", "energycells", 20, 40, 120.0
        )
    ]

    inserted, skipped = insert_domain_data(
        conn, metadata, factions, wares, sectors, resources, ships, recipes
    )

    assert inserted == 7  # 1 metadata + 1 faction + 2 wares + 1 sector + 1 resource + 1 ship + 1 recipe
    assert skipped == 0
    conn.close()


def test_atomic_swap_success(tmp_path):
    """Verifies that atomic_ingest_to_db populates and replaces target database file cleanly."""
    target_db = tmp_path / "x4_advisor.db"

    # Step 1: Initial ingest
    def initial_pop(conn):
        metadata = DatasetMetadata("7.0", "123", "2026-08-24", True)
        factions = [FactionRecord("argon", "Argon Federation")]
        return insert_domain_data(conn, metadata, factions, [], [], [], [], [])

    inserted, skipped = atomic_ingest_to_db(target_db, initial_pop)
    assert inserted == 1
    assert target_db.exists()

    # Verify content
    conn = get_connection(target_db)
    rows = conn.execute("SELECT id, name FROM factions").fetchall()
    assert rows == [("argon", "Argon Federation")]
    conn.close()

    # Step 2: Re-ingest overwrite
    def second_pop(conn):
        metadata = DatasetMetadata("7.1", "456", "2026-08-25", True)
        factions = [FactionRecord("teladi", "Teladi Company")]
        return insert_domain_data(conn, metadata, factions, [], [], [], [], [])

    inserted2, skipped2 = atomic_ingest_to_db(target_db, second_pop)
    assert inserted2 == 1

    # Verify overwritten content
    conn = get_connection(target_db)
    rows2 = conn.execute("SELECT id, name FROM factions").fetchall()
    assert rows2 == [("teladi", "Teladi Company")]
    conn.close()


def test_atomic_swap_failure_preserves_existing_db(tmp_path):
    """Verifies that if ingestion encounters an error mid-write, original database survives intact."""
    target_db = tmp_path / "x4_advisor.db"

    # Step 1: Populate good DB
    def initial_pop(conn):
        metadata = DatasetMetadata("7.0", "123", "2026-08-24", True)
        factions = [FactionRecord("argon", "Argon Federation")]
        return insert_domain_data(conn, metadata, factions, [], [], [], [], [])

    atomic_ingest_to_db(target_db, initial_pop)

    # Step 2: Ingestion that raises an error
    def failing_pop(conn):
        raise RuntimeError("Simulated extraction error")

    with pytest.raises(RuntimeError, match="Simulated extraction error"):
        atomic_ingest_to_db(target_db, failing_pop)

    # Verify original DB survives
    conn = get_connection(target_db)
    rows = conn.execute("SELECT id, name FROM factions").fetchall()
    assert rows == [("argon", "Argon Federation")]
    conn.close()
