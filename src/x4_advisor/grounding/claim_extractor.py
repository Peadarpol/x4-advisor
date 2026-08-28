"""Deterministic proposition extractor and modality classifier for grounding verification."""

import re
from typing import List, Tuple

ADVICE_MODAL_KEYWORDS = [
    "recommend",
    "recommended",
    "should",
    "suggest",
    "suggested",
    "advisable",
    "best to",
    "it is best",
    "consider",
    "useful to",
    "useful for",
    "strategy",
    "heuristic",
    "tip:",
    "note:",
]


SYSTEM_TEMPLATE_PATTERNS = [
    "your query matches multiple possible entities",
    "this question concerns content from an x4: foundations dlc",
    "no matching records or relevant strategic guidance were found",
    "i was unable to classify or map this query",
    "this question is outside the scope of x4: foundations",
    "please check the spelling",
    "please verify the entity",
]

STRUCTURAL_LEADIN_PATTERNS = [
    r"^the following (?:ships|wares|sectors|items|prices|inputs|recipes) (?:are|belong to|have|listed)",
    r"^here is the (?:complete )?list of",
    r"^based on the provided (?:data|evidence|structured data),? (?:the following|there are|the ships|the wares|the inputs)",
    r"^to produce .*?,? (?:the following|the required|the primary|these|inputs)",
    r"^these \d+ entries represent",
    r"^each of these entries is listed",
    r"^the wares in the .*? category are as follows",
    r"^an internal error occurred",
    r"^\*\*(?:large|medium|small|extra large) ships.*?\*\*$",
    r"^\*\*orders:?\*\*$",
    r"^\*\*default behaviours:?\*\*$",
    r"^\*\*set the default behavior.*?\*\*$",
    r"^\*\*minimum price:?\*\*$",
    r"^\*\*average price:?\*\*$",
    r"^\*\*maximum price:?\*\*$",
]


class ClaimExtractor:
    """Extracts propositional statements from synthesis answers and tags their epistemic modality."""

    def extract_propositions(self, text: str) -> List[Tuple[str, str]]:
        """Decomposes markdown text into atomic proposition strings and provisional modality (FACT, ADVICE, INFERENCE).

        Returns list of (proposition_text, provisional_modality).
        """
        if not text:
            return []

        propositions: List[Tuple[str, str]] = []
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        for line in lines:
            # Clean leading bullet markers (*, -, 1.)
            cleaned_line = re.sub(r"^(\*|\-|\d+\.)\s+", "", line).strip()
            if not cleaned_line or cleaned_line.startswith("#"):
                continue

            # Check if standalone line is purely a structural markdown header or category marker
            lower_line = cleaned_line.lower()
            if any(re.match(p, lower_line) for p in STRUCTURAL_LEADIN_PATTERNS):
                continue
            if lower_line.endswith(":") and ("following" in lower_line or "list" in lower_line or "below" in lower_line):
                continue

            # Check if line matches standard system templates
            if any(p in lower_line for p in SYSTEM_TEMPLATE_PATTERNS):
                continue

            # Split line into sentences on periods/semicolons followed by space
            sentences = re.split(r"(?<=[.;])\s+", cleaned_line)
            for s in sentences:
                s_clean = s.strip().rstrip(".")
                if len(s_clean) < 5:
                    continue

                lower_s = s_clean.lower()

                # Filter out structural sentence fragments
                if any(re.match(p, lower_s) for p in STRUCTURAL_LEADIN_PATTERNS):
                    continue
                if any(p in lower_s for p in SYSTEM_TEMPLATE_PATTERNS):
                    continue

                # Classify modality
                if any(kw in lower_s for kw in ADVICE_MODAL_KEYWORDS):
                    modality = "ADVICE"
                elif any(word in lower_s for word in ["therefore", "combined", "total", "approximately", "meaning", "takes", "spread", "difference", "more expensive", "cheaper"]) or re.search(r"\bmore\b.*\bthan\b|\bless\b.*\bthan\b", lower_s):
                    modality = "SUPPORTED_INFERENCE"
                elif any(phrase in lower_s for phrase in ["no information", "not provided in", "does not contain", "no data available"]):
                    modality = "NEGATIVE_EVIDENCE"
                else:
                    modality = "FACT"

                propositions.append((s_clean, modality))

        return propositions
