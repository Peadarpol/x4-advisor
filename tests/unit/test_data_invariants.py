"""Permanent data integrity and Phase 0 invariant regression tests."""

import sqlite3
import pytest

from x4_advisor.config import get_config
from x4_advisor.storage.db import get_connection


@pytest.fixture
def db_conn():
    config = get_config(validate=False)
    assert config.database_path.exists(), f"Database not found at {config.database_path}"
    conn = get_connection(config.database_path)
    yield conn
    conn.close()


def test_ship_ware_mapping_uniqueness(db_conn):
    """Every ship record must map to a unique ware_id."""
    rows = db_conn.execute(
        """
        SELECT ware_id, count(*) c
        FROM ships
        WHERE ware_id IS NOT NULL
        GROUP BY ware_id
        HAVING c > 1
        """
    ).fetchall()
    assert rows == [], f"Found duplicate ship-to-ware mappings: {rows}"


def test_ship_display_name_uniqueness(db_conn):
    """All Vanguard/Sentinel variants are uniquely disambiguated.

    Note: In base game wares.xml, Egosoft pointed both ship_arg_s_heavyfighter_01_a
    and ship_arg_s_heavyfighter_02_a at the identical string {20101,10402} ('Eclipse Vanguard').
    Aside from this single base-game XML duplicate, all 173 other ships have unique names.
    """
    rows = db_conn.execute(
        """
        SELECT name, count(*) c
        FROM ships
        WHERE ware_id IS NOT NULL AND name != 'Eclipse Vanguard'
        GROUP BY name
        HAVING c > 1
        """
    ).fetchall()
    assert rows == [], f"Found unexpected duplicate ship display names: {rows}"


def test_no_demo_sectors_in_base_game_database(db_conn):
    """Tutorial/demo map sectors (demo_*) must be filtered out at extraction and never present in production DB."""
    rows = db_conn.execute(
        """
        SELECT id, name FROM sectors WHERE id LIKE 'demo_%' OR LOWER(name) LIKE '%demo%'
        """
    ).fetchall()
    assert rows == [], f"Found demo map sectors in database: {rows}"


def test_no_backslash_corruption(db_conn):
    """No backslash escapes from XML non-greedy regex should exist in ship or ware names."""
    ship_rows = db_conn.execute("SELECT id, name FROM ships WHERE name LIKE '%\\%'").fetchall()
    ware_rows = db_conn.execute("SELECT id, name FROM wares WHERE name LIKE '%\\%'").fetchall()
    assert ship_rows == [], f"Found backslashes in ships: {ship_rows}"
    assert ware_rows == [], f"Found backslashes in wares: {ware_rows}"


def test_no_unwanted_outer_parentheses_in_sectors(db_conn):
    """Sector names must have XML catalog wrapper parentheses stripped (e.g. 'Grand Exchange I', not '(Grand Exchange I)')."""
    rows = db_conn.execute("SELECT id, name FROM sectors WHERE name LIKE '(%'").fetchall()
    assert rows == [], f"Found parenthesized sector names: {rows}"


def test_no_empty_or_placeholder_wares(db_conn):
    """Ware names must not be empty strings or bare '()' corruption."""
    rows = db_conn.execute(
        "SELECT id, name FROM wares WHERE name = '' OR name = '()' OR name = '(-)'"
    ).fetchall()
    assert rows == [], f"Found corrupt ware names: {rows}"


def test_curation_tables_preserved(db_conn):
    """All curation registry, manifest, and chunk embeddings must be preserved."""
    registry_count = db_conn.execute("SELECT count(*) FROM source_registry").fetchone()[0]
    manifest_count = db_conn.execute("SELECT count(*) FROM source_manifest").fetchone()[0]
    chunks_count = db_conn.execute("SELECT count(*) FROM knowledge_chunks").fetchone()[0]
    vec_count = db_conn.execute("SELECT count(*) FROM knowledge_chunks_vec").fetchone()[0]

    assert registry_count == 25, f"Expected 25 source_registry rows, got {registry_count}"
    assert manifest_count == 7, f"Expected 7 source_manifest rows, got {manifest_count}"
    assert chunks_count == 53, f"Expected 53 knowledge_chunks rows, got {chunks_count}"
    assert vec_count == 53, f"Expected 53 knowledge_chunks_vec rows, got {vec_count}"
