"""Retrieval package exposing EntityResolver, StructuredQueryEngine, and query result models."""

from x4_advisor.retrieval.entity_resolver import EntityResolver
from x4_advisor.retrieval.models import (
    AmbiguousEntityResult,
    CategoryListResult,
    EntityNotFoundResult,
    ProductionChainResult,
    ProductionNode,
    RankingItem,
    RankingResult,
    ResolvedEntity,
    SingleEntityResult,
)
from x4_advisor.retrieval.structured_query import StructuredQueryEngine

__all__ = [
    "EntityResolver",
    "StructuredQueryEngine",
    "ResolvedEntity",
    "AmbiguousEntityResult",
    "EntityNotFoundResult",
    "SingleEntityResult",
    "RankingItem",
    "RankingResult",
    "ProductionNode",
    "ProductionChainResult",
    "CategoryListResult",
]
