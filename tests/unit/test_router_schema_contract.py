"""Contract qualification test asserting bidirectional agreement between JSON Schema enums and code/database allowlists."""

from pathlib import Path
import sqlite3
import pytest

from x4_advisor.retrieval.vocabularies import (
    ALLOWED_METRICS,
    VALID_OPERATIONS,
    VALID_PURPOSES,
    VALID_SHIP_CLASSES,
    DynamicVocabularies,
    build_router_json_schema,
)
from x4_advisor.retrieval.structured_query import ALLOWED_SHIP_METRICS, ALLOWED_WARE_METRICS

# Explicitly documented intentional exclusions per R5.7 / R6.1-R6.6
INTENTIONAL_EXCLUSIONS = {
    "metric": {"cargo"},  # 'cargo' is an alias for 'cargo_capacity' in ALLOWED_SHIP_METRICS, canonical schema uses 'cargo_capacity'
    "ship_class": {"spacesuit"},  # 'spacesuit' is present in raw DB ships, but excluded from tactical ship ranking schema
}


@pytest.fixture
def db_conn():
    db_path = Path("data/db/x4_advisor.db")
    if not db_path.exists():
        pytest.skip("Database file data/db/x4_advisor.db not found for schema contract test.")
    conn = sqlite3.connect(str(db_path))
    yield conn
    conn.close()


def test_router_schema_enum_bidirectional_contract(db_conn):
    """Asserts that all router JSON schema enums match the code allowlists and database distinct values exactly."""
    vocab = DynamicVocabularies(db_conn)
    schema = build_router_json_schema(vocab)
    props = schema["properties"]["structured"]["properties"]

    # 1. Metric: Schema enum vs (ALLOWED_SHIP_METRICS | ALLOWED_WARE_METRICS) + 'none'
    code_metrics = set(ALLOWED_SHIP_METRICS.keys()) | set(ALLOWED_WARE_METRICS.keys())
    code_metrics_canonical = (code_metrics - INTENTIONAL_EXCLUSIONS.get("metric", set())) | {"none"}
    schema_metrics = set(props["metric"]["enum"])
    assert schema_metrics == code_metrics_canonical, f"Metric mismatch: {schema_metrics ^ code_metrics_canonical}"

    # 2. Ship Class: Schema enum vs VALID_SHIP_CLASSES
    schema_classes = set(props["ship_class"]["enum"])
    assert schema_classes == set(VALID_SHIP_CLASSES), f"Ship class mismatch: {schema_classes ^ set(VALID_SHIP_CLASSES)}"

    # 3. Purpose: Schema enum vs VALID_PURPOSES
    schema_purposes = set(props["purpose"]["enum"])
    assert schema_purposes == set(VALID_PURPOSES), f"Purpose mismatch: {schema_purposes ^ set(VALID_PURPOSES)}"

    # 4. Operations: Schema enum vs VALID_OPERATIONS
    schema_ops = set(props["operation"]["enum"])
    assert schema_ops == set(VALID_OPERATIONS), f"Operations mismatch: {schema_ops ^ set(VALID_OPERATIONS)}"

    # 5. Database-derived: Categories
    db_categories = {r[0].lower() for r in db_conn.execute("SELECT DISTINCT category FROM wares WHERE category IS NOT NULL AND category != ''").fetchall()} | {"none"}
    schema_categories = set(props["category"]["enum"])
    assert schema_categories == db_categories, f"Category mismatch: {schema_categories ^ db_categories}"

    # 6. Database-derived: Resources
    db_resources = {r[0].lower() for r in db_conn.execute("SELECT DISTINCT resource_id FROM sector_resources WHERE resource_id IS NOT NULL AND resource_id != ''").fetchall()} | {"none"}
    schema_resources = set(props["resource_id"]["enum"])
    assert schema_resources == db_resources, f"Resource mismatch: {schema_resources ^ db_resources}"

    # 7. Database-derived: Production Methods
    db_methods = {r[0].lower() for r in db_conn.execute("SELECT DISTINCT method FROM production_recipes WHERE method IS NOT NULL AND method != ''").fetchall()} | {"none"}
    schema_methods = set(props["production_method"]["enum"])
    assert schema_methods == db_methods, f"Production method mismatch: {schema_methods ^ db_methods}"
