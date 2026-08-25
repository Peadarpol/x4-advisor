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
    total_available: int = 0
    redirected_from: Optional[str] = None


# Type alias for entity resolution outcomes
EntityResolutionOutcome = Union[ResolvedEntity, AmbiguousEntityResult, EntityNotFoundResult]


# ---------------------------------------------------------------------------
# M5 Exceptions and Enums
# ---------------------------------------------------------------------------


class DatabaseNotReadyError(RuntimeError):
    """Raised when the database does not exist or core tables are missing/empty."""

    pass


class UnknownFilterValue(ValueError):
    """Raised when a structured query filter value is not found in the valid database domain."""

    def __init__(
        self,
        field: str,
        attempted_value: str,
        valid_values: List[str],
        message: Optional[str] = None,
    ) -> None:
        self.field = field
        self.attempted_value = attempted_value
        self.valid_values = valid_values
        msg = message or f"Unknown value '{attempted_value}' for filter '{field}'. Valid values: {valid_values}"
        super().__init__(msg)


from enum import Enum


class RouteType(str, Enum):
    """Routing decision enum for LLM Router."""

    STRUCTURED = "STRUCTURED"
    VECTOR = "VECTOR"
    BOTH = "BOTH"
    ABSTAIN = "ABSTAIN"


class AbstainReason(str, Enum):
    """Distinct abstention reasons per SPEC-001 §7."""

    NO_EVIDENCE = "NO_EVIDENCE"
    OUT_OF_SCOPE_DLC = "OUT_OF_SCOPE_DLC"
    OUT_OF_SCOPE_OTHER = "OUT_OF_SCOPE_OTHER"
    MALFORMED_TOOL_CALL = "MALFORMED_TOOL_CALL"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


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


# ---------------------------------------------------------------------------
# M5 Router & Synthesizer Data Models
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """Single parsed tool call produced by the LLM Router."""

    name: str
    arguments: Dict[str, Any]


@dataclass
class RouterResult:
    """Outcome of LLMRouter classification."""

    route_type: RouteType
    tool_calls: List[ToolCall] = field(default_factory=list)
    abstain_reason: Optional[AbstainReason] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class SynthesisResult:
    """Outcome of GroundedSynthesizer generation."""

    answer_text: str
    has_evidence: bool = True
    abstain_reason: Optional[AbstainReason] = None
    evidence_chunk_ids: List[str] = field(default_factory=list)
    was_method_fallback: bool = False
    notes: List[str] = field(default_factory=list)
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class AdvisorResponse:
    """End-to-end response returned by AdvisorEngine."""

    question: str
    route_result: Optional[RouterResult] = None
    structured_result: Optional[Any] = None
    vector_result: Optional[VectorSearchResult] = None
    synthesis_result: Optional[SynthesisResult] = None
    ambiguous_candidates: Optional[List[ResolvedEntity]] = None
    pending_route: Optional[RouterResult] = None


