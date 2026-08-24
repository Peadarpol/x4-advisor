"""Retrieval models and dataclasses for query engine results and entity resolution."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ResolvedEntity:
    """Single resolved entity identifier and metadata."""

    id: str
    name: str
    entity_type: str  # 'ship', 'ware', 'sector', 'faction'


@dataclass
class AmbiguousEntityResult:
    """Returned when a natural-language query resolves to multiple entity candidates."""

    query_name: str
    candidates: List[ResolvedEntity]


@dataclass
class EntityNotFoundResult:
    """Returned when a natural-language query resolves to zero entities."""

    query_name: str
    message: str = "No matching entity found."


@dataclass
class SingleEntityResult:
    """Template T1: Fact lookup result for a single entity."""

    entity_id: str
    entity_name: str
    entity_type: str
    data: Dict[str, Any]


@dataclass
class RankingItem:
    """Item within a Template T2 ranking query result."""

    id: str
    name: str
    value: float
    unit: str
    metric_name: str
    purpose: Optional[str] = None
    ship_class: Optional[str] = None


@dataclass
class RankingResult:
    """Template T2: Comparison / Ranking query result."""

    category: str
    metric: str
    sort_order: str  # 'DESC' or 'ASC'
    items: List[RankingItem]


@dataclass
class ProductionNode:
    """Node in a Template T3 multi-tier production recipe tree."""

    ware_id: str
    ware_name: str
    method: str
    amount_needed: int
    direct_inputs: List["ProductionNode"] = field(default_factory=list)


@dataclass
class ProductionChainResult:
    """Template T3: Production chain traversal result."""

    target_ware_id: str
    target_ware_name: str
    method: str
    output_amount: int
    production_time: float
    tree: ProductionNode
    total_raw_materials: Dict[str, int]
    was_method_fallback: bool = False
    requested_method: Optional[str] = None


@dataclass
class CategoryListResult:
    """Template T4: Category listing and filtering query result."""

    category_type: str  # e.g., 'faction', 'ship_class', 'purpose', 'ware_group'
    category_value: str
    items: List[Dict[str, Any]]


# Type alias for entity resolution outcomes
EntityResolutionOutcome = Union[ResolvedEntity, AmbiguousEntityResult, EntityNotFoundResult]


# ---------------------------------------------------------------------------
# M4 Vector Retrieval Result Models
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """Single chunk retrieved from vector similarity search.

    Attributes:
        similarity_score: Primary metric for downstream consumers (M5 router,
            M6 grounding). Cosine similarity = 1.0 - cosine_distance,
            bounded [0.0, 1.0].
        distance: Raw cosine distance from sqlite-vec. Retained for
            diagnostic/logging purposes only — not for threshold comparison.
    """

    chunk_id: str
    manifest_id: str
    heading_hierarchy: str
    content: str
    similarity_score: float
    distance: float
    source_attribution: str
    topic: Optional[str] = None
    game_version_scope: str = "base_game"


@dataclass
class VectorSearchResult:
    """Complete result from a VectorQueryEngine.search() call.

    Attributes:
        status: One of 'success', 'no_relevant_chunks', 'embedding_failed',
            'database_not_ready'.
        total_candidates: Number of KNN candidates returned by the vector
            index *before* similarity threshold filtering. Distinct from
            len(chunks), which is the post-threshold count. Gives M5 both
            signals (e.g., "retrieved 5 candidates, 2 above threshold").
        message: Human-readable detail for non-success statuses (e.g.,
            embedding error description, missing-table explanation).
    """

    query_text: str
    chunks: List[RetrievedChunk]
    status: str  # 'success' | 'no_relevant_chunks' | 'embedding_failed' | 'database_not_ready'
    total_candidates: int
    threshold_used: float
    message: str = ""

