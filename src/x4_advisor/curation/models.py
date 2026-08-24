"""Dataclasses for unstructured curation, claim extraction, verification diffs, and chunking."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TypedClaim:
    """Single typed factual claim extracted from source text."""

    subject: str  # Entity name or topic (e.g. 'Cerberus Vanguard')
    predicate: str  # Property or relation (e.g. 'has_cargo_capacity')
    object: str  # Claimed value (e.g. '1760')
    unit: Optional[str] = None  # Measurement unit (e.g. 'm3', 'm/s')
    qualifier: Optional[str] = None  # Epistemic context/attribution (e.g. 'base game', 'often recommended')

    def to_dict(self) -> Dict[str, Any]:
        """Converts claim to dictionary representation."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "unit": self.unit,
            "qualifier": self.qualifier,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TypedClaim":
        """Instantiates TypedClaim from dictionary."""
        return cls(
            subject=str(data.get("subject", "")).strip(),
            predicate=str(data.get("predicate", "")).strip(),
            object=str(data.get("object", "")).strip(),
            unit=str(data["unit"]).strip() if data.get("unit") else None,
            qualifier=str(data["qualifier"]).strip() if data.get("qualifier") else None,
        )


@dataclass
class FidelityDiffResult:
    """Result of C1 vs C2 claim pair comparison."""

    c1_claim: TypedClaim
    c2_claim: Optional[TypedClaim]
    status: str  # 'match', 'mismatch', 'missing'
    drift_category: Optional[str] = None  # 'polarity', 'quantifier', 'modality', 'attribution', 'numeric', 'entity', 'unit'
    details: Optional[str] = None


@dataclass
class DBVerificationResult:
    """Result of claim verification against M1 SQLite database."""

    claim: TypedClaim
    status: str  # 'verified', 'mismatch', 'unverified_entity'
    db_value: Optional[Any] = None
    details: Optional[str] = None


@dataclass
class TextChunk:
    """Heading-aware text chunk produced by the chunker."""

    content: str
    heading_hierarchy: str
    chunk_index: int
    word_count: int
    source_attribution: str
    topic: Optional[str] = None
    related_entity_ids: List[str] = field(default_factory=list)
