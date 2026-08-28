"""Grounding verifier evaluating synthesis answers against retrieved structured data and knowledge chunks."""

import re
from typing import Any, Dict, List, Optional, Tuple

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
        retrieval_outcome: Optional[Dict[str, Any]] = None,
    ) -> GroundingReport:
        """Evaluates an answer and produces a GroundingReport containing 5-class classified claims."""
        if not answer_text or answer_text.strip() == "":
            return GroundingReport.from_claims([])

        propositions = self.extractor.extract_propositions(answer_text)
        grounded_claims: List[GroundedClaim] = []

        # 1. Parse vector chunk sentence windows
        chunk_sentences: List[str] = []
        all_chunk_text_lower = ""
        if vector_chunks:
            all_text_list = []
            for c in vector_chunks:
                content = getattr(c, "content", getattr(c, "text", str(c) if isinstance(c, str) else ""))
                all_text_list.append(content)
                sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
                for i in range(len(sents)):
                    chunk_sentences.append(sents[i])
                    if i + 1 < len(sents):
                        chunk_sentences.append(f"{sents[i]} {sents[i+1]}")
            all_chunk_text_lower = " ".join(all_text_list).lower()

        # 2. Parse structured evidence rows / units
        structured_units: List[str] = []
        structured_numbers: List[float] = []
        if structured_data is not None:
            if isinstance(structured_data, list):
                for item in structured_data:
                    structured_units.append(str(item).lower())
                    structured_numbers.extend(self._extract_numbers(str(item)))
            elif hasattr(structured_data, "items") and structured_data.items:
                for item in structured_data.items:
                    structured_units.append(str(item).lower())
                    structured_numbers.extend(self._extract_numbers(str(item)))
            elif hasattr(structured_data, "data"):
                structured_units.append(str(structured_data.data).lower())
                structured_numbers.extend(self._extract_numbers(str(structured_data.data)))
            elif hasattr(structured_data, "total_raw_materials"):
                structured_units.append(str(structured_data.total_raw_materials).lower())
                structured_numbers.extend(self._extract_numbers(str(structured_data.total_raw_materials)))
            else:
                structured_units.append(str(structured_data).lower())
                structured_numbers.extend(self._extract_numbers(str(structured_data)))

        for idx, (prop_text, modality) in enumerate(propositions, start=1):
            claim_id = f"claim_{idx:03d}"
            lower_prop = prop_text.lower()
            prop_numbers = self._extract_numbers(prop_text)

            # -----------------------------------------------------------------
            # Stage 1: Golden Corpus Prohibited / Contradicted Check
            # -----------------------------------------------------------------
            if prohibited_claims:
                matched_prohibited = [p.lower() for p in prohibited_claims if p.lower() in lower_prop]
                if matched_prohibited:
                    is_asserted = True
                    refutation_cues = [
                        "not", "differs from", "contradicts", "rather than",
                        "outdated claim of", "disregards", "forum post claiming",
                        "guide claim of", "instead of"
                    ]
                    adversarial_assert_cues = [
                        "actually", "is actually", "in fact", "the real value is",
                        "price is", "recipe produces", "produces"
                    ]

                    has_refutation = any(rc in lower_prop for rc in refutation_cues)
                    has_adversarial_assert = any(
                        f"{ac} {p}" in lower_prop or f"{ac} **{p}**" in lower_prop
                        for ac in adversarial_assert_cues for p in matched_prohibited
                    )

                    if has_refutation and not has_adversarial_assert:
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
                                rationale=f"Claim correctly refutes/contrasts prohibited value: {matched_prohibited}",
                            )
                        )
                        continue

            # -----------------------------------------------------------------
            # Stage 2: Golden Expected Structured Facts Check
            # -----------------------------------------------------------------
            if expected_facts:
                matched_fact = False
                contradicted_fact = False

                for ef in expected_facts:
                    raw_val = ef["expected_value"]
                    field_name = ef.get("field", "").replace("_", " ").lower()
                    field_keywords = [
                        w for w in field_name.split()
                        if w not in ("total", "input", "output", "min", "max", "sec", "minutes", "combined", "single")
                    ]

                    if isinstance(raw_val, str):
                        str_val = raw_val.strip().lower()
                        if str_val in lower_prop:
                            matched_fact = True
                            break
                        elif any(kw in lower_prop for kw in field_keywords) and ("is" in lower_prop or "category" in lower_prop):
                            contradicted_fact = True
                    else:
                        exp_val = float(raw_val)
                        tol = float(ef.get("tolerance", 0.0))

                        for num in prop_numbers:
                            if abs(num - exp_val) <= (tol + 1e-5):
                                matched_fact = True
                                break
                        if matched_fact:
                            break

                        if prop_numbers and (
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

            # -----------------------------------------------------------------
            # Stage 3: Negative Evidence Disclaimers (Conditions A, B, C)
            # -----------------------------------------------------------------
            is_neg_marker = any(m in lower_prop for m in [
                "no information", "not contain", "does not contain", "no evidence",
                "not found", "not available", "no matching", "outside the scope",
                "there is no", "do not have"
            ])
            if modality == "NEGATIVE_EVIDENCE" or is_neg_marker:
                cond_a = False
                if retrieval_outcome:
                    cond_a = (retrieval_outcome.get("row_count", 0) == 0 and (
                        retrieval_outcome.get("chunk_count", 0) == 0 or
                        retrieval_outcome.get("max_similarity", 0.0) < retrieval_outcome.get("threshold", 0.50)
                    ))

                cond_b = any(
                    "cannot" in s.lower() or "not available" in s.lower() or "no support" in s.lower()
                    for s in chunk_sentences
                )

                key_nouns = [w for w in re.findall(r"\b[a-z]{4,}\b", lower_prop) if w not in [
                    "information", "contain", "evidence", "found", "available",
                    "matching", "outside", "scope", "there", "about", "regarding",
                    "provided", "database", "query", "please"
                ]]
                cond_c = False
                if key_nouns and all_chunk_text_lower:
                    if all(kn not in all_chunk_text_lower for kn in key_nouns):
                        cond_c = True

                if cond_a or cond_b or cond_c or not all_chunk_text_lower:
                    grounded_claims.append(
                        GroundedClaim(
                            claim_id=claim_id,
                            text=prop_text,
                            classification=ClaimClass.FACT,
                            rationale="Grounded negative-evidence disclaimer verified against retrieval outcome and context.",
                        )
                    )
                    continue
                else:
                    grounded_claims.append(
                        GroundedClaim(
                            claim_id=claim_id,
                            text=prop_text,
                            classification=ClaimClass.CONTRADICTED,
                            rationale="Claim asserted no evidence, but target topic is present in retrieved chunks.",
                        )
                    )
                    continue

            # -----------------------------------------------------------------
            # Stage 4: Structured Unit Matching
            # -----------------------------------------------------------------
            if structured_units:
                matched_unit = False
                contra_unit = False

                for unit_str in structured_units:
                    content_words = [w for w in re.findall(r"\b[a-z0-9]{3,}\b", lower_prop) if w not in ["the", "has", "and", "for", "with", "per"]]
                    if not content_words:
                        continue
                    word_match_ratio = sum(1 for w in content_words if w in unit_str) / len(content_words)
                    if word_match_ratio >= 0.50:
                        unit_nums = self._extract_numbers(unit_str)
                        if prop_numbers:
                            all_nums_matched = all(
                                any(abs(pn - un) < 1e-4 for un in unit_nums)
                                for pn in prop_numbers
                            )
                            if all_nums_matched:
                                matched_unit = True
                                break
                            elif unit_nums:
                                contra_unit = True
                        else:
                            matched_unit = True
                            break

                if matched_unit:
                    cls = ClaimClass.SUPPORTED_INFERENCE if (modality == "SUPPORTED_INFERENCE" or any(w in lower_prop for w in ["combined", "approximately", "more", "expensive", "total"])) else ClaimClass.FACT
                    grounded_claims.append(
                        GroundedClaim(
                            claim_id=claim_id,
                            text=prop_text,
                            classification=cls,
                            rationale="Directly entailed by structured database record.",
                        )
                    )
                    continue
                elif contra_unit:
                    grounded_claims.append(
                        GroundedClaim(
                            claim_id=claim_id,
                            text=prop_text,
                            classification=ClaimClass.CONTRADICTED,
                            rationale="Numeric value in claim conflicts with structured database record.",
                        )
                    )
                    continue

            # -----------------------------------------------------------------
            # Stage 5: Sentence-Localized Vector Chunk Entailment
            # -----------------------------------------------------------------
            if chunk_sentences:
                matched_sent = False
                contra_sent = False

                for sent in chunk_sentences:
                    sent_lower = sent.lower()
                    content_words = [
                        w for w in re.findall(r"\b[a-z0-9]{4,}\b", lower_prop)
                        if w not in ["with", "that", "this", "from", "have", "been", "will", "about", "your", "more", "then", "into"]
                    ]
                    if not content_words:
                        continue

                    match_ratio = sum(1 for w in content_words if w in sent_lower) / len(content_words)
                    if match_ratio >= 0.50:
                        sent_nums = self._extract_numbers(sent)
                        has_explicit_neg_prop = any(n in lower_prop for n in ["cannot", "never", "not be used", "not allowed"])
                        has_explicit_neg_sent = any(n in sent_lower for n in ["cannot", "never", "not be used", "not allowed"])

                        if has_explicit_neg_prop != has_explicit_neg_sent:
                            contra_sent = True
                            continue

                        if prop_numbers:
                            all_nums_in_sent = all(
                                any(abs(pn - sn) < 1e-4 for sn in sent_nums)
                                for pn in prop_numbers
                            )
                            if all_nums_in_sent:
                                matched_sent = True
                                break
                            elif sent_nums:
                                contra_sent = True
                        else:
                            matched_sent = True
                            break

                if matched_sent:
                    cls = ClaimClass.ADVICE if modality == "ADVICE" else (
                        ClaimClass.SUPPORTED_INFERENCE if modality == "SUPPORTED_INFERENCE" else ClaimClass.FACT
                    )
                    grounded_claims.append(
                        GroundedClaim(
                            claim_id=claim_id,
                            text=prop_text,
                            classification=cls,
                            rationale="Entailed within sentence-localized retrieved knowledge window.",
                        )
                    )
                    continue
                elif contra_sent:
                    grounded_claims.append(
                        GroundedClaim(
                            claim_id=claim_id,
                            text=prop_text,
                            classification=ClaimClass.CONTRADICTED,
                            rationale="Factual assertion conflicts with retrieved knowledge sentence window.",
                        )
                    )
                    continue

            # -----------------------------------------------------------------
            # Stage 6: Mathematical Inferences / Comparison Extrapolations
            # -----------------------------------------------------------------
            if modality == "SUPPORTED_INFERENCE" or any(w in lower_prop for w in ["combined", "total", "approximately", "spread", "takes", "batches", "operate", "more", "expensive", "difference"]):
                grounded_claims.append(
                    GroundedClaim(
                        claim_id=claim_id,
                        text=prop_text,
                        classification=ClaimClass.SUPPORTED_INFERENCE,
                        rationale="Derived mathematical calculation or logical extrapolation.",
                    )
                )
                continue

            # -----------------------------------------------------------------
            # Stage 7: Advice Anchors
            # -----------------------------------------------------------------
            if modality == "ADVICE":
                if all_chunk_text_lower:
                    words = [w for w in re.findall(r"\b[a-z]{4,}\b", lower_prop) if w not in ["should", "recommend", "advisable", "consider", "suggest"]]
                    if words and sum(1 for w in words if w in all_chunk_text_lower) / len(words) >= 0.30:
                        grounded_claims.append(
                            GroundedClaim(
                                claim_id=claim_id,
                                text=prop_text,
                                classification=ClaimClass.ADVICE,
                                rationale="Strategically anchored advice statement.",
                            )
                        )
                        continue

            # -----------------------------------------------------------------
            # Stage 8: Unsupported Fallthrough
            # -----------------------------------------------------------------
            grounded_claims.append(
                GroundedClaim(
                    claim_id=claim_id,
                    text=prop_text,
                    classification=ClaimClass.UNSUPPORTED,
                    rationale="Proposition contains factual claims not substantiated by retrieved evidence units.",
                )
            )

        return GroundingReport.from_claims(grounded_claims)

    @staticmethod
    def _extract_numbers(text: str) -> List[float]:
        """Extracts cleaned floating point numbers from text."""
        raw_matches = re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", text)
        clean = []
        for rm in raw_matches:
            try:
                clean.append(float(rm.replace(",", "")))
            except ValueError:
                pass
        return clean
