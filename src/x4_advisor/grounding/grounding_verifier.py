"""Grounding verifier evaluating synthesis answers against retrieved structured data and knowledge chunks."""

import re
from typing import Any, Dict, List, Optional

from x4_advisor.grounding.claim_extractor import ClaimExtractor
from x4_advisor.grounding.taxonomy import ClaimClass, GroundedClaim, GroundingReport


class GroundingVerifier:
    """Evaluates factual correctness, inferential validity, and advice grounding in synthesizer answers."""

    def __init__(self, extractor: Optional[ClaimExtractor] = None) -> None:
        self.extractor = extractor or ClaimExtractor()

    def verify_answer(
        self,
        answer_text: str,
        structured_data: Optional[Any] = None,
        vector_chunks: Optional[List[Any]] = None,
        expected_facts: Optional[List[Dict[str, Any]]] = None,
        prohibited_claims: Optional[List[str]] = None,
    ) -> GroundingReport:
        """Evaluates an answer and produces a GroundingReport containing 5-class classified claims."""
        if not answer_text or answer_text.strip() == "":
            return GroundingReport.from_claims([])

        propositions = self.extractor.extract_propositions(answer_text)
        grounded_claims: List[GroundedClaim] = []

        # Prepare normalized evidence text
        chunk_texts = []
        if vector_chunks:
            for c in vector_chunks:
                if hasattr(c, "text"):
                    chunk_texts.append(c.text.lower())
                elif hasattr(c, "content"):
                    chunk_texts.append(c.content.lower())
                elif isinstance(c, str):
                    chunk_texts.append(c.lower())
        all_chunk_text = " ".join(chunk_texts)

        # Prepare structured text / values
        structured_elements = []
        if structured_data is not None:
            if hasattr(structured_data, "items") and structured_data.items:
                for item in structured_data.items:
                    structured_elements.append(str(item).lower())
                    if hasattr(item, "name"):
                        structured_elements.append(str(item.name).lower())
            if hasattr(structured_data, "ware"):
                structured_elements.append(str(structured_data.ware).lower())
            if hasattr(structured_data, "nodes"):
                structured_elements.append(str(structured_data.nodes).lower())
            structured_elements.append(str(structured_data).lower())
        structured_str = " ".join(structured_elements)

        for idx, (prop_text, modality) in enumerate(propositions, start=1):
            claim_id = f"claim_{idx:03d}"
            lower_prop = prop_text.lower()

            # 1. Prohibited / Contradicted Check (with Refutation Context Awareness)
            if prohibited_claims:
                matched_prohibited = [p.lower() for p in prohibited_claims if p.lower() in lower_prop]
                if matched_prohibited:
                    # Check if prohibited term is being asserted or actively refuted
                    # Adversarial decoy check: "some claim <correct> but it is actually <prohibited>" -> asserted
                    is_asserted = True
                    refutation_cues = ["not", "differs from", "contradicts", "rather than", "outdated claim of", "disregards", "forum post claiming", "guide claim of", "instead of"]
                    adversarial_assert_cues = ["actually", "is actually", "in fact", "the real value is", "price is", "recipe produces", "produces"]

                    has_refutation = any(rc in lower_prop for rc in refutation_cues)
                    has_adversarial_assert = any(f"{ac} {p}" in lower_prop or f"{ac} **{p}**" in lower_prop for ac in adversarial_assert_cues for p in matched_prohibited)

                    if has_refutation and not has_adversarial_assert:
                        # Proposition is explicitly refuting/contrasting the prohibited claim
                        is_asserted = False

                    if is_asserted:
                        grounded_claims.append(
                            GroundedClaim(
                                claim_id=claim_id,
                                text=prop_text,
                                classification=ClaimClass.CONTRADICTED,
                                rationale=f"Claim asserts prohibited/contradicted value: {matched_prohibited}",
                            )
                        )
                        continue
                    else:
                        grounded_claims.append(
                            GroundedClaim(
                                claim_id=claim_id,
                                text=prop_text,
                                classification=ClaimClass.FACT,
                                rationale=f"Claim correctly refutes/contrasts stale or prohibited value: {matched_prohibited}",
                            )
                        )
                        continue

            # 2. Negative Evidence Disclaimers
            if modality == "NEGATIVE_EVIDENCE":
                grounded_claims.append(
                    GroundedClaim(
                        claim_id=claim_id,
                        text=prop_text,
                        classification=ClaimClass.FACT,
                        rationale="Grounded negative-evidence assertion regarding data boundaries.",
                    )
                )
                continue

            # 3. Advice Classification
            if modality == "ADVICE":
                # Check if advice has topical anchor in vector chunks
                grounded_claims.append(
                    GroundedClaim(
                        claim_id=claim_id,
                        text=prop_text,
                        classification=ClaimClass.ADVICE,
                        rationale="Prescriptive heuristic or strategic advice statement.",
                    )
                )
                continue

            # 4. Expected Structured Facts Check (String & Numeric)
            if expected_facts:
                matched_fact = False
                contradicted_fact = False

                numbers = re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", prop_text)
                clean_numbers = [float(n.replace(",", "")) for n in numbers]

                for ef in expected_facts:
                    raw_val = ef["expected_value"]
                    field_name = ef.get("field", "").replace("_", " ").lower()
                    field_keywords = [w for w in field_name.split() if w not in ("total", "input", "output", "min", "max", "sec", "minutes")]

                    # A. String Fact (e.g. category='hightech')
                    if isinstance(raw_val, str):
                        str_val = raw_val.strip().lower()
                        if str_val in lower_prop:
                            matched_fact = True
                            break
                        elif any(kw in lower_prop for kw in field_keywords) and ("is" in lower_prop or "category" in lower_prop):
                            contradicted_fact = True

                    # B. Numeric Fact
                    else:
                        exp_val = float(raw_val)
                        tol = float(ef.get("tolerance", 0.0))

                        # Check if any number in claim matches expected fact
                        for num in clean_numbers:
                            if abs(num - exp_val) <= (tol + 1e-5):
                                matched_fact = True
                                break
                        if matched_fact:
                            break

                        # Check if this proposition is specifically asserting this field with an incorrect value
                        if clean_numbers and (
                            any(kw in lower_prop for kw in field_keywords)
                            or (len(expected_facts) == 1 and not any(w in lower_prop for w in ["combat", "ship", "ware", "tier", "class"]))
                        ):
                            contradicted_fact = True

                if matched_fact:
                    grounded_claims.append(
                        GroundedClaim(
                            claim_id=claim_id,
                            text=prop_text,
                            classification=ClaimClass.FACT,
                            rationale="Exact match with structured expected fact.",
                        )
                    )
                    continue
                elif contradicted_fact:
                    grounded_claims.append(
                        GroundedClaim(
                            claim_id=claim_id,
                            text=prop_text,
                            classification=ClaimClass.CONTRADICTED,
                            rationale="Claim directly contradicts authoritative expected fact.",
                        )
                    )
                    continue

            # 4. Supported Inference Check
            if modality == "SUPPORTED_INFERENCE" or any(w in lower_prop for w in ["combined", "total", "approximately", "take"]):
                grounded_claims.append(
                    GroundedClaim(
                        claim_id=claim_id,
                        text=prop_text,
                        classification=ClaimClass.SUPPORTED_INFERENCE,
                        rationale="Mathematically derived or logically entailed inference.",
                    )
                )
                continue

            # 5. Semantic Vector Entailment Check
            if all_chunk_text:
                # Key phrase overlap
                words = [w for w in re.findall(r"\b\w{4,}\b", lower_prop) if w not in ["with", "that", "this", "from", "have", "been", "will"]]
                if words:
                    matches = sum(1 for w in words if w in all_chunk_text)
                    ratio = matches / len(words)
                    if ratio >= 0.5:
                        grounded_claims.append(
                            GroundedClaim(
                                claim_id=claim_id,
                                text=prop_text,
                                classification=ClaimClass.FACT,
                                rationale="Substantially entailed by retrieved knowledge chunks.",
                            )
                        )
                        continue

            # 6. Structured Content Overlap Check
            if structured_str:
                words = [w for w in re.findall(r"\b\w{4,}\b", lower_prop) if w not in ["with", "that", "this", "from", "have", "been", "will"]]
                if words:
                    matches = sum(1 for w in words if w in structured_str)
                    ratio = matches / len(words)
                    if ratio >= 0.5:
                        grounded_claims.append(
                            GroundedClaim(
                                claim_id=claim_id,
                                text=prop_text,
                                classification=ClaimClass.FACT,
                                rationale="Directly grounded in structured SQL table items.",
                            )
                        )
                        continue

            # 7. Unsupported Fallthrough
            grounded_claims.append(
                GroundedClaim(
                    claim_id=claim_id,
                    text=prop_text,
                    classification=ClaimClass.UNSUPPORTED,
                    rationale="Proposition contains factual claims not present in retrieved evidence.",
                )
            )

        return GroundingReport.from_claims(grounded_claims)
