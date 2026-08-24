"""Curated epistemic marker categories for detecting epistemic drift between claim sets."""

import re
from typing import Dict, List, Optional, Tuple

POLARITY_WEAK = {"does not", "cannot", "rarely", "unlikely", "seldom", "never", "hardly"}
POLARITY_STRONG = {"does", "commonly", "certainly", "definitely"}

QUANTIFIER_WEAK = {"often", "usually", "sometimes", "frequently", "mostly", "generally", "typically"}
QUANTIFIER_STRONG = {"always", "every", "all", "invariably", "without exception"}

MODALITY_WEAK = {"can", "may", "might", "could", "possibly", "optionally"}
MODALITY_STRONG = {"will", "must", "should", "definitely", "requires"}

ATTRIBUTION_WEAK = {"recommends", "suggests", "prefers", "opinion", "considered", "popular choice"}
ATTRIBUTION_STRONG = {"is the best", "objectively", "optimal", "undisputed", "factually superior", "is best"}


MARKER_CATEGORIES: Dict[str, Tuple[List[str], List[str]]] = {
    "polarity": (list(POLARITY_WEAK), list(POLARITY_STRONG)),
    "quantifier": (list(QUANTIFIER_WEAK), list(QUANTIFIER_STRONG)),
    "modality": (list(MODALITY_WEAK), list(MODALITY_STRONG)),
    "attribution": (list(ATTRIBUTION_WEAK), list(ATTRIBUTION_STRONG)),
}


def detect_epistemic_drift(
    c1_qualifier: Optional[str],
    c2_qualifier: Optional[str],
    c1_predicate: Optional[str] = None,
    c2_predicate: Optional[str] = None,
) -> Optional[str]:
    """Detects epistemic drift (category name) between C1 and C2 text fields.

    Checks across both qualifier and predicate text fields. Returns the name of the drift category
    ('polarity', 'quantifier', 'modality', 'attribution') if a weaker marker in C1 shifts to a stronger marker in C2,
    or None if no epistemic drift is detected.
    """
    text1 = f"{c1_qualifier or ''} {c1_predicate or ''}".lower()
    text2 = f"{c2_qualifier or ''} {c2_predicate or ''}".lower()

    if not text1.strip() or not text2.strip():
        return None

    for category, (weak_terms, strong_terms) in MARKER_CATEGORIES.items():
        has_weak_in_c1 = any(_contains_word_or_phrase(text1, term) for term in weak_terms)
        has_strong_in_c2 = any(_contains_word_or_phrase(text2, term) for term in strong_terms)

        if has_weak_in_c1 and has_strong_in_c2:
            # Check C1 didn't already have the strong term
            c1_has_strong = any(_contains_word_or_phrase(text1, term) for term in strong_terms)
            if not c1_has_strong:
                return category

    return None


def _contains_word_or_phrase(text: str, term: str) -> bool:
    pattern = r"\b" + re.escape(term) + r"\b"
    return bool(re.search(pattern, text))
