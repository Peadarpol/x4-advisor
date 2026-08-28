"""Claim taxonomy enums and grounding report data structures for Milestone M6."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ClaimClass(str, Enum):
    """5-Class claim taxonomy per SPEC-001 §6."""
    FACT = "FACT"
    SUPPORTED_INFERENCE = "SUPPORTED_INFERENCE"
    ADVICE = "ADVICE"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


@dataclass
class GroundedClaim:
    """Individual propositional claim extracted from synthesizer answer."""
    claim_id: str
    text: str
    classification: ClaimClass
    evidence_id: Optional[str] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    value: Optional[Any] = None
    unit: Optional[str] = None
    rationale: str = ""


@dataclass
class GroundingReport:
    """Evaluation report analyzing grounding fidelity across all claims in an answer."""
    total_claims: int
    facts_count: int
    inferences_count: int
    advice_count: int
    unsupported_count: int
    contradicted_count: int
    unsupported_claim_rate: float  # unsupported_count / total_claims
    claims: List[GroundedClaim] = field(default_factory=list)
    is_grounded: bool = True
    abstain_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @classmethod
    def from_claims(cls, claims: List[GroundedClaim], notes: Optional[List[str]] = None) -> "GroundingReport":
        total = len(claims)
        if total == 0:
            return cls(
                total_claims=0,
                facts_count=0,
                inferences_count=0,
                advice_count=0,
                unsupported_count=0,
                contradicted_count=0,
                unsupported_claim_rate=0.0,
                claims=[],
                is_grounded=True,
                notes=notes or [],
            )

        facts = sum(1 for c in claims if c.classification == ClaimClass.FACT)
        inferences = sum(1 for c in claims if c.classification == ClaimClass.SUPPORTED_INFERENCE)
        advice = sum(1 for c in claims if c.classification == ClaimClass.ADVICE)
        unsupported = sum(1 for c in claims if c.classification == ClaimClass.UNSUPPORTED)
        contradicted = sum(1 for c in claims if c.classification == ClaimClass.CONTRADICTED)

        ucr = unsupported / total if total > 0 else 0.0
        # Answer is grounded if 0 contradicted claims and unsupported count <= 1
        is_grounded = (contradicted == 0) and (unsupported <= 1)

        return cls(
            total_claims=total,
            facts_count=facts,
            inferences_count=inferences,
            advice_count=advice,
            unsupported_count=unsupported,
            contradicted_count=contradicted,
            unsupported_claim_rate=ucr,
            claims=claims,
            is_grounded=is_grounded,
            notes=notes or [],
        )
