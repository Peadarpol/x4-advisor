"""Retrieval package exposing EntityResolver, StructuredQueryEngine, VectorQueryEngine, and query result models."""

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
    RetrievedChunk,
    SingleEntityResult,
    VectorSearchResult,
)
from x4_advisor.retrieval.structured_query import StructuredQueryEngine
from x4_advisor.retrieval.vector_query import VectorQueryEngine

__all__ = [
    "EntityResolver",
    "StructuredQueryEngine",
    "VectorQueryEngine",
    "ResolvedEntity",
    "AmbiguousEntityResult",
    "EntityNotFoundResult",
    "SingleEntityResult",
    "RankingItem",
    "RankingResult",
    "ProductionNode",
    "ProductionChainResult",
    "CategoryListResult",
    "RetrievedChunk",
    "VectorSearchResult",
]
