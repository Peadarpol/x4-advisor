"""Integration tests for StructuredQueryEngine against extracted real SQLite database data/db/x4_advisor.db."""

from pathlib import Path
import pytest

from x4_advisor.retrieval.entity_resolver import EntityResolver
from x4_advisor.retrieval.models import (
    CategoryListResult,
    ProductionChainResult,
    RankingResult,
    ResolvedEntity,
    SingleEntityResult,
)
from x4_advisor.retrieval.structured_query import StructuredQueryEngine

DB_PATH = Path("data/db/x4_advisor.db")

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="Real extracted database 'data/db/x4_advisor.db' not available.",
)


def test_integration_entity_resolver_and_t1_fact_lookup():
    """Integration test: Resolves display name 'Cerberus Vanguard' and fetches T1 fact record."""
    with EntityResolver(db_path=DB_PATH) as resolver, StructuredQueryEngine(db_path=DB_PATH) as engine:
        res = resolver.resolve_entity("Cerberus Vanguard", entity_types=["ship"])
        assert isinstance(res, ResolvedEntity)
        assert res.entity_type == "ship"

        fact = engine.query_t1_fact_lookup(res.id)
        assert isinstance(fact, SingleEntityResult)
        assert fact.data["name"] == "Cerberus Vanguard"
        assert fact.data["purpose"] == "fight"


def test_integration_t2_canonical_miner_ranking():
    """Integration test: 'Which L-class miners have the most cargo?' executed against real database."""
    with StructuredQueryEngine(db_path=DB_PATH) as engine:
        res = engine.query_t2_ranking(
            category_or_class="ship_l",
            metric="cargo_capacity",
            purpose="mine",
            limit=5,
        )
        assert isinstance(res, RankingResult)
        assert len(res.items) > 0
        for item in res.items:
            assert item.ship_class == "ship_l"
            assert item.purpose == "mine"
            assert item.value > 0


def test_integration_t3_claytronics_production_chain():
    """Integration test: 'What do I need to produce Claytronics?' production chain traversal."""
    with StructuredQueryEngine(db_path=DB_PATH) as engine:
        res = engine.query_t3_production_chain("claytronics")
        assert isinstance(res, ProductionChainResult)
        assert res.target_ware_id == "claytronics"
        assert len(res.total_raw_materials) > 0


def test_integration_t4_argon_category_listing():
    """Integration test: 'List all Argon ships' category listing."""
    with StructuredQueryEngine(db_path=DB_PATH) as engine:
        res = engine.query_t4_category_listing("faction", "argon", limit=10)
        assert isinstance(res, CategoryListResult)
        assert len(res.items) > 0
        for item in res.items:
            assert item["faction_name"] == "Argon Federation"
