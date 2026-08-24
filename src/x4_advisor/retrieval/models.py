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
