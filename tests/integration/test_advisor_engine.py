"""Integration tests for AdvisorEngine coordinating routing, retrieval, disambiguation, and synthesis."""

import json
from pathlib import Path
import sqlite3
from unittest.mock import MagicMock

import pytest

from x4_advisor.config import Config
from x4_advisor.retrieval.advisor_engine import AdvisorEngine
from x4_advisor.retrieval.models import (
    AbstainReason,
    DatabaseNotReadyError,
    ResolvedEntity,
    RouteType,
)


@pytest.fixture
def test_db_conn(tmp_path: Path) -> sqlite3.Connection:
    """Creates an in-memory or temporary SQLite database populated with minimal M1 fixtures."""
    db_file = tmp_path / "test_advisor.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA foreign_keys = ON;")

    # Schema setup for 6 core tables + sqlite-vec vector table
    conn.executescript(
        """
        CREATE TABLE factions (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            short_name TEXT
        );

        CREATE TABLE ships (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            class TEXT NOT NULL,
            hull REAL NOT NULL,
            shields REAL NOT NULL,
            cargo_capacity REAL NOT NULL,
            cargo_type TEXT,
            speed REAL NOT NULL,
            weapon_slots INTEGER NOT NULL DEFAULT 0,
            turret_slots INTEGER NOT NULL DEFAULT 0,
            shield_slots INTEGER NOT NULL DEFAULT 0,
            purpose TEXT,
            faction_id TEXT,
            ware_id TEXT,
            FOREIGN KEY (faction_id) REFERENCES factions(id)
        );

        CREATE TABLE wares (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            min_price REAL NOT NULL,
            avg_price REAL NOT NULL,
            max_price REAL NOT NULL,
            volume REAL NOT NULL
        );

        CREATE TABLE sectors (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            faction_id TEXT,
            sunlight REAL,
            FOREIGN KEY (faction_id) REFERENCES factions(id)
        );

        CREATE TABLE sector_resources (
            sector_id TEXT NOT NULL,
            resource_id TEXT NOT NULL,
            yield REAL NOT NULL,
            PRIMARY KEY (sector_id, resource_id),
            FOREIGN KEY (sector_id) REFERENCES sectors(id)
        );

        CREATE TABLE production_recipes (
            ware_id TEXT NOT NULL,
            method TEXT NOT NULL,
            output_amount INTEGER NOT NULL,
            production_time REAL NOT NULL,
            input_ware_id TEXT NOT NULL,
            input_amount INTEGER NOT NULL,
            PRIMARY KEY (ware_id, method, input_ware_id),
            FOREIGN KEY (ware_id) REFERENCES wares(id)
        );

        -- Minimal fixtures
        INSERT INTO factions (id, name, short_name) VALUES ('argon', 'Argon Federation', 'ARG');
        INSERT INTO ships (id, name, class, hull, shields, cargo_capacity, speed, purpose, faction_id)
        VALUES ('ship_arg_m_frigate_01_a', 'Cerberus Vanguard', 'ship_m', 21000, 4200, 840, 172, 'fight', 'argon'),
               ('ship_arg_s_fighter_01_a', 'Buster Sentinel', 'ship_s', 4500, 1200, 150, 130, 'fight', 'argon'),
               ('ship_arg_l_destroyer_01_a', 'Behemoth Vanguard', 'ship_l', 180000, 32000, 5000, 120, 'fight', 'argon');

        INSERT INTO wares (id, name, category, min_price, avg_price, max_price, volume)
        VALUES ('hullparts', 'Hull Parts', 'construction', 150, 210, 270, 12),
               ('graphene', 'Graphene', 'refined', 80, 120, 160, 6),
               ('energycells', 'Energy Cells', 'energy', 10, 16, 22, 1),
               ('ore', 'Ore', 'minerals', 30, 50, 70, 10);

        INSERT INTO sectors (id, name, faction_id) VALUES ('sec_arg_prime', 'Argon Prime', 'argon');
        INSERT INTO sector_resources (sector_id, resource_id, yield) VALUES ('sec_arg_prime', 'ore', 54000.0);

        INSERT INTO production_recipes (ware_id, method, output_amount, production_time, input_ware_id, input_amount)
        VALUES ('hullparts', 'default', 3, 60.0, 'graphene', 2),
               ('hullparts', 'default', 3, 60.0, 'energycells', 8);
        """
    )
    conn.commit()
    return conn


def test_advisor_engine_database_not_ready_raises() -> None:
    """Tests that AdvisorEngine raises DatabaseNotReadyError when tables are missing or empty."""
    empty_conn = sqlite3.connect(":memory:")
    with pytest.raises(DatabaseNotReadyError, match="Database is not initialized or missing core tables"):
        AdvisorEngine(conn=empty_conn, config=Config(validate=False))


def test_advisor_engine_end_to_end_t1_structured_flow(test_db_conn: sqlite3.Connection) -> None:
    """Tests end-to-end T1 fact lookup flow with mocked client."""
    mock_client = MagicMock()
    # 1. Router JSON schema response
    router_payload = {
        "route_type": "STRUCTURED",
        "structured": {
            "operation": "lookup_entity",
            "query_name": "Cerberus Vanguard",
        },
        "vector": {"query_text": ""},
        "abstain_reason": "NONE",
    }
    router_resp = {"message": {"role": "assistant", "content": json.dumps(router_payload)}}

    # 2. Synthesizer chat response
    synth_resp = {
        "message": {
            "content": "The Cerberus Vanguard has a cargo capacity of 840 m³ and speed of 172 m/s."
        }
    }
    mock_client.chat.side_effect = [router_resp, synth_resp]

    engine = AdvisorEngine(conn=test_db_conn, client=mock_client, config=Config(validate=False))
    resp = engine.answer("What is the cargo capacity of Cerberus Vanguard?")

    assert resp.route_result.route_type == RouteType.STRUCTURED
    assert resp.structured_result is not None
    assert resp.structured_result.entity_name == "Cerberus Vanguard"
    assert "840 m³" in resp.synthesis_result.answer_text


def test_advisor_engine_ambiguous_entity_and_resumption(test_db_conn: sqlite3.Connection) -> None:
    """Tests returning candidate choices on ambiguous entity match and resuming with user choice."""
    test_db_conn.execute(
        "INSERT INTO ships (id, name, class, hull, shields, cargo_capacity, speed, purpose, faction_id) "
        "VALUES ('ship_arg_m_frigate_01_b', 'Cerberus Sentinel', 'ship_m', 24000, 5200, 900, 150, 'fight', 'argon')"
    )
    test_db_conn.commit()

    mock_client = MagicMock()
    router_payload = {
        "route_type": "STRUCTURED",
        "structured": {
            "operation": "lookup_entity",
            "query_name": "Cerberus",
        },
        "vector": {"query_text": ""},
        "abstain_reason": "NONE",
    }
    router_resp = {"message": {"role": "assistant", "content": json.dumps(router_payload)}}
    synth_resp = {
        "message": {
            "content": "The Cerberus Sentinel has 900 m³ cargo capacity."
        }
    }
    mock_client.chat.side_effect = [router_resp, synth_resp]

    engine = AdvisorEngine(conn=test_db_conn, client=mock_client, config=Config(validate=False))

    # Turn 1: user asks about 'Cerberus' -> returns candidates
    resp1 = engine.answer("Tell me about the Cerberus")
    assert resp1.ambiguous_candidates is not None
    assert len(resp1.ambiguous_candidates) == 2
    assert resp1.pending_route is not None
    assert "Cerberus Vanguard" in resp1.synthesis_result.answer_text

    # Turn 2: user selects 'ship_arg_m_frigate_01_b' -> executes without re-routing
    resp2 = engine.answer(
        question="Tell me about the Cerberus",
        pending_route=resp1.pending_route,
        resolved_entity_id="ship_arg_m_frigate_01_b",
    )
    assert resp2.structured_result.entity_id == "ship_arg_m_frigate_01_b"
    assert resp2.structured_result.entity_name == "Cerberus Sentinel"
    assert "900 m³" in resp2.synthesis_result.answer_text


def test_advisor_engine_t4_category_listing_ware_redirection(test_db_conn: sqlite3.Connection) -> None:
    """Tests that T4 category listing redirects ware name 'ore' to category 'minerals'."""
    mock_client = MagicMock()
    router_payload = {
        "route_type": "STRUCTURED",
        "structured": {
            "operation": "list_category",
            "category": "ore",
        },
        "vector": {"query_text": ""},
        "abstain_reason": "NONE",
    }
    router_resp = {"message": {"role": "assistant", "content": json.dumps(router_payload)}}
    synth_resp = {"message": {"content": "Ore is in the minerals category."}}
    mock_client.chat.side_effect = [router_resp, synth_resp]

    engine = AdvisorEngine(conn=test_db_conn, client=mock_client, config=Config(validate=False))
    resp = engine.answer("List items in category ore")

    assert resp.structured_result.category_value == "minerals"
    assert resp.structured_result.redirected_from == "ore"
    assert any("redirected" in n.lower() or "ore" in n.lower() for n in resp.synthesis_result.notes)


def test_advisor_engine_dlc_abstention(test_db_conn: sqlite3.Connection) -> None:
    """Tests abstaining when router flags out_of_scope_dlc."""
    mock_client = MagicMock()
    router_payload = {
        "route_type": "ABSTAIN",
        "structured": {"operation": "none"},
        "vector": {"query_text": ""},
        "abstain_reason": "OUT_OF_SCOPE_DLC",
    }
    router_resp = {"message": {"role": "assistant", "content": json.dumps(router_payload)}}
    mock_client.chat.return_value = router_resp

    engine = AdvisorEngine(conn=test_db_conn, client=mock_client, config=Config(validate=False))
    resp = engine.answer("Stats for the Syn battleship?")

    assert resp.route_result.route_type == RouteType.ABSTAIN
    assert resp.route_result.abstain_reason == AbstainReason.OUT_OF_SCOPE_DLC
    assert "DLC expansion" in resp.synthesis_result.answer_text
    assert resp.synthesis_result.has_evidence is False


def test_advisor_engine_shared_connection_lifecycle(test_db_conn: sqlite3.Connection) -> None:
    """Tests that sub-engines do not close the shared connection, and AdvisorEngine closes it properly."""
    engine = AdvisorEngine(conn=test_db_conn, client=MagicMock(), config=Config(validate=False))
    assert engine.structured_engine._close_conn_on_exit is False
    assert engine.vector_engine._close_conn_on_exit is False
    assert engine._close_conn_on_exit is False

    engine.close()
    cursor = test_db_conn.cursor()
    assert cursor.execute("SELECT 1").fetchone()[0] == 1


def test_advisor_engine_catches_unknown_filter_value_and_retries(test_db_conn: sqlite3.Connection) -> None:
    """Confirms AdvisorEngine catches UnknownFilterValue and feeds valid values into router retry path."""
    mock_client = MagicMock()
    # First router call: invalid ware category 'unobtainium'
    first_payload = {
        "route_type": "STRUCTURED",
        "structured": {
            "operation": "list_category",
            "category": "unobtainium",
        },
        "vector": {"query_text": ""},
        "abstain_reason": "NONE",
    }
    # Second router call (retry): corrected category 'minerals'
    second_payload = {
        "route_type": "STRUCTURED",
        "structured": {
            "operation": "list_category",
            "category": "minerals",
        },
        "vector": {"query_text": ""},
        "abstain_reason": "NONE",
    }
    synth_resp = {"message": {"content": "Found 1 mineral ware: Ore."}}
    mock_client.chat.side_effect = [
        {"message": {"role": "assistant", "content": json.dumps(first_payload)}},
        {"message": {"role": "assistant", "content": json.dumps(second_payload)}},
        synth_resp,
    ]

    engine = AdvisorEngine(conn=test_db_conn, client=mock_client, config=Config(validate=False))
    resp = engine.answer("List all unobtainium items")

    assert resp.structured_result is not None
    assert resp.structured_result.category_type == "category"
    assert resp.structured_result.category_value == "minerals"
    assert len(resp.structured_result.items) == 1
    assert resp.structured_result.items[0]["name"] == "Ore"


def test_advisor_engine_unresolved_unknown_filter_value_names_invalid_value(test_db_conn: sqlite3.Connection) -> None:
    """Confirms that when retry still fails, the user message explicitly names the invalid value asked about."""
    mock_client = MagicMock()
    first_payload = {
        "route_type": "STRUCTURED",
        "structured": {
            "operation": "list_category",
            "category": "nonexistent_space_magic",
        },
        "vector": {"query_text": ""},
        "abstain_reason": "NONE",
    }
    retry_payload = {
        "route_type": "ABSTAIN",
        "structured": {"operation": "none"},
        "vector": {"query_text": ""},
        "abstain_reason": "OUT_OF_SCOPE_OTHER",
    }
    mock_client.chat.side_effect = [
        {"message": {"role": "assistant", "content": json.dumps(first_payload)}},
        {"message": {"role": "assistant", "content": json.dumps(retry_payload)}},
    ]

    engine = AdvisorEngine(conn=test_db_conn, client=mock_client, config=Config(validate=False))
    resp = engine.answer("List all nonexistent_space_magic items")

    assert resp.synthesis_result is not None
    ans = resp.synthesis_result.answer_text
    assert "nonexistent_space_magic" in ans
    assert "category" in ans
    assert resp.synthesis_result.has_evidence is False
