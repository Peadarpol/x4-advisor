"""Integrity test for eval_corpus.json ensuring all referenced entities, chunks, and facts exist in SQLite."""

import json
from pathlib import Path
import sqlite3
import pytest

from x4_advisor.config import get_config
from x4_advisor.storage.db import get_connection


@pytest.fixture
def eval_corpus_data():
    corpus_path = Path("tests/fixtures/eval_corpus.json")
    assert corpus_path.exists(), f"Evaluation corpus fixture not found at {corpus_path}"
    with open(corpus_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def db_conn():
    config = get_config(validate=False)
    assert config.database_path.exists(), f"Database not found at {config.database_path}"
    conn = get_connection(config.database_path)
    yield conn
    conn.close()


def test_eval_corpus_case_count_and_schema(eval_corpus_data):
    """Asserts that corpus contains exactly 36 cases and conforms to schema."""
    assert len(eval_corpus_data) == 36

    case_ids = set()
    required_keys = {
        "case_id",
        "category",
        "question",
        "expected_route",
        "expected_tool_calls",
        "expected_entity_ids",
        "expected_chunk_ids",
        "expected_abstention",
        "expected_structured_facts",
        "allowed_inferences",
        "prohibited_unsupported_claims",
    }

    for case in eval_corpus_data:
        assert required_keys.issubset(case.keys()), f"Case {case.get('case_id')} missing required keys"
        assert case["case_id"] not in case_ids, f"Duplicate case_id: {case['case_id']}"
        case_ids.add(case["case_id"])
        assert case["expected_route"] in ("STRUCTURED", "VECTOR", "BOTH", "ABSTAIN")


def test_eval_corpus_entities_exist_in_db(eval_corpus_data, db_conn):
    """Asserts that every expected_entity_id exists in ships, wares, sectors, or factions."""
    for case in eval_corpus_data:
        for entity_id in case["expected_entity_ids"]:
            found = False
            # Check ships
            if db_conn.execute("SELECT 1 FROM ships WHERE id = ? OR ware_id = ?", (entity_id, entity_id)).fetchone():
                found = True
            # Check wares
            elif db_conn.execute("SELECT 1 FROM wares WHERE id = ?", (entity_id,)).fetchone():
                found = True
            # Check sectors
            elif db_conn.execute("SELECT 1 FROM sectors WHERE id = ?", (entity_id,)).fetchone():
                found = True
            # Check factions
            elif db_conn.execute("SELECT 1 FROM factions WHERE id = ?", (entity_id,)).fetchone():
                found = True

            assert found, f"Case {case['case_id']}: Entity ID '{entity_id}' not found in any SQLite table."


def test_eval_corpus_chunks_exist_in_db(eval_corpus_data, db_conn):
    """Asserts that every expected_chunk_id exists in knowledge_chunks table."""
    for case in eval_corpus_data:
        for chunk_id in case["expected_chunk_ids"]:
            row = db_conn.execute("SELECT 1 FROM knowledge_chunks WHERE id = ?", (chunk_id,)).fetchone()
            assert row is not None, f"Case {case['case_id']}: Chunk ID '{chunk_id}' not found in knowledge_chunks."


def test_eval_corpus_structured_facts_match_db(eval_corpus_data, db_conn):
    """Asserts that facts asserted in expected_structured_facts match exact DB ground truth."""
    # 1. Cerberus Vanguard stats
    cerberus_row = db_conn.execute(
        "SELECT cargo_capacity, speed FROM ships WHERE id = 'ship_arg_m_frigate_01_a_macro'"
    ).fetchone()
    assert cerberus_row == (1760.0, 300.0)

    # 2. Claytronics prices
    claytronics_row = db_conn.execute(
        "SELECT min_price, avg_price, max_price FROM wares WHERE id = 'claytronics'"
    ).fetchone()
    assert claytronics_row == (1734, 2040, 2346)

    # 3. Grand Exchange I sunlight
    ge_row = db_conn.execute(
        "SELECT sunlight FROM sectors WHERE id = 'Cluster_01_Sector001_macro'"
    ).fetchone()
    assert abs(ge_row[0] - 1.23) < 0.01

    # 4. Hull parts recipe output
    hp_recipe = db_conn.execute(
        "SELECT output_amount, production_time FROM production_recipes WHERE ware_id = 'hullparts' AND method = 'default'"
    ).fetchone()
    assert hp_recipe == (294, 900.0)
