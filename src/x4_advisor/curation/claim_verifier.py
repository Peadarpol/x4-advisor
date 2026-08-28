"""Dual-loop claim verifier performing paraphrase fidelity check (C1 vs C2) and database fact check (C1 vs M1 DB)."""

import difflib
import logging
from pathlib import Path
import re
import sqlite3
from typing import List, Optional, Set, Tuple

from x4_advisor.curation.epistemic_markers import detect_epistemic_drift
from x4_advisor.curation.models import (
    DBVerificationResult,
    FidelityDiffResult,
    TypedClaim,
)

logger = logging.getLogger(__name__)


def _extract_first_number(val: Optional[str]) -> Optional[float]:
    """Extracts the first embedded numerical float value from a string (e.g. 'about 1' -> 1.0)."""
    if not val:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", str(val))
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def _string_similarity(s1: str, s2: str) -> float:
    """Computes normalized sequence similarity ratio between two strings."""
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1.strip().lower(), s2.strip().lower()).ratio()


class ClaimVerifier:
    """Automated claim verifier executing dual-loop verification."""

    def verify_fidelity(
        self,
        c1_claims: List[TypedClaim],
        c2_claims: List[TypedClaim],
    ) -> List[FidelityDiffResult]:
        """Compares initial claims C1 against re-extracted claims C2 for entity/numeric/unit mismatches and epistemic drift."""
        results: List[FidelityDiffResult] = []
        c1_count = len(c1_claims)
        c2_count = len(c2_claims)

        used_c2_indices: Set[int] = set()

        for c1 in c1_claims:
            subj1 = c1.subject.strip().lower()
            pred1 = c1.predicate.strip().lower()

            c2: Optional[TypedClaim] = None
            best_c2_idx: Optional[int] = None

            # Pass A: Exact (subject, predicate) key match
            for j, candidate in enumerate(c2_claims):
                if j in used_c2_indices:
                    continue
                if candidate.subject.strip().lower() == subj1 and candidate.predicate.strip().lower() == pred1:
                    best_c2_idx = j
                    c2 = candidate
                    break

            # Pass B: Fallback subject-only match with candidate scoring & threshold
            if best_c2_idx is None:
                c2_candidates = [
                    (j, candidate) for j, candidate in enumerate(c2_claims)
                    if j not in used_c2_indices and candidate.subject.strip().lower() == subj1
                ]
                best_cand_score = 0.0
                best_cand_idx = None
                for j, candidate in c2_candidates:
                    pred_sim = _string_similarity(pred1, candidate.predicate)
                    obj_sim = _string_similarity(c1.object, candidate.object)
                    num1 = _extract_first_number(c1.object)
                    num2 = _extract_first_number(candidate.object)
                    num_bonus = 0.5 if (num1 is not None and num2 is not None and abs(num1 - num2) < 1e-3) else 0.0
                    score = max(pred_sim * 0.4 + obj_sim * 0.6, pred_sim * 0.7 + num_bonus)
                    if score > best_cand_score:
                        best_cand_score = score
                        best_cand_idx = j

                # Enforce minimum similarity threshold before pairing
                if best_cand_idx is not None and (best_cand_score >= 0.35 or _string_similarity(pred1, c2_claims[best_cand_idx].predicate) >= 0.40):
                    best_c2_idx = best_cand_idx
                    c2 = c2_claims[best_cand_idx]

            # Pass C: Fallback positional window / object similarity match
            if best_c2_idx is None:
                for j, candidate in enumerate(c2_claims):
                    if j in used_c2_indices:
                        continue
                    subj_sim = _string_similarity(subj1, candidate.subject)
                    obj_sim = _string_similarity(c1.object, candidate.object)
                    if subj_sim >= 0.70 and obj_sim >= 0.70:
                        best_c2_idx = j
                        c2 = candidate
                        break

            if best_c2_idx is not None and c2 is not None:
                used_c2_indices.add(best_c2_idx)

                # 1. Object comparison (numeric vs string fuzzy)
                is_obj_mismatch, obj_details = _compare_objects(c1.object, c2.object)
                if is_obj_mismatch:
                    results.append(
                        FidelityDiffResult(
                            c1_claim=c1,
                            c2_claim=c2,
                            status="mismatch",
                            drift_category="numeric" if _extract_first_number(c1.object) is not None and _extract_first_number(c2.object) is not None else "entity",
                            details=f"Value mismatch: C1='{c1.object}' vs C2='{c2.object}'. {obj_details}",
                        )
                    )
                    continue

                # 2. Unit comparison
                if (c1.unit or "").lower() != (c2.unit or "").lower():
                    results.append(
                        FidelityDiffResult(
                            c1_claim=c1,
                            c2_claim=c2,
                            status="mismatch",
                            drift_category="unit",
                            details=f"Unit mismatch: C1='{c1.unit}' vs C2='{c2.unit}'.",
                        )
                    )
                    continue

                # 3. Epistemic drift check
                drift = detect_epistemic_drift(
                    c1_qualifier=c1.qualifier,
                    c2_qualifier=c2.qualifier,
                    c1_predicate=c1.predicate,
                    c2_predicate=c2.predicate,
                )
                if drift:
                    results.append(
                        FidelityDiffResult(
                            c1_claim=c1,
                            c2_claim=c2,
                            status="mismatch",
                            drift_category=drift,
                            details=f"Epistemic drift detected ({drift}): C1='{c1.qualifier}' vs C2='{c2.qualifier}'.",
                        )
                    )
                    continue

                # Full Match
                results.append(
                    FidelityDiffResult(
                        c1_claim=c1,
                        c2_claim=c2,
                        status="match",
                    )
                )
            else:
                # Missing match
                results.append(
                    FidelityDiffResult(
                        c1_claim=c1,
                        c2_claim=None,
                        status="missing",
                        details=f"Claim for '{c1.subject}' -> '{c1.predicate}' missing from re-extracted paraphrase claims (C2). (Claim count C1={c1_count}, C2={c2_count}).",
                    )
                )

        return results

    def verify_against_db(
        self,
        c1_claims: List[TypedClaim],
        conn: Optional[sqlite3.Connection] = None,
        db_path: Optional[Path] = None,
    ) -> List[DBVerificationResult]:
        """Validates factual C1 claims against normalized M1 SQLite database."""
        if conn is None and db_path is None:
            raise ValueError("Either conn or db_path must be provided to verify_against_db.")

        should_close = False
        target_conn = conn
        if target_conn is None and db_path is not None:
            target_conn = sqlite3.connect(str(db_path))
            should_close = True

        prev_query_only = 0
        if target_conn and not should_close:
            try:
                row = target_conn.execute("PRAGMA query_only;").fetchone()
                if row:
                    prev_query_only = row[0]
            except sqlite3.OperationalError:
                pass

        results: List[DBVerificationResult] = []
        try:
            # Security invariant: read-only access on database connection
            target_conn.execute("PRAGMA query_only = ON;")
            cursor = target_conn.cursor()

            for claim in c1_claims:
                subj_clean = claim.subject.strip().lower()
                pred_clean = claim.predicate.strip().lower()

                # Search Ships
                ship_row = cursor.execute(
                    "SELECT id, name, cargo_capacity, hull, shields, speed, weapon_slots, turret_slots, shield_slots "
                    "FROM ships WHERE LOWER(name) = ? OR LOWER(id) = ?",
                    (subj_clean, subj_clean),
                ).fetchone()

                if ship_row:
                    res = self._verify_ship_claim(claim, ship_row)
                    if res:
                        results.append(res)
                        continue

                # Search Wares
                ware_row = cursor.execute(
                    "SELECT id, name, min_price, avg_price, max_price, volume FROM wares WHERE LOWER(name) = ? OR LOWER(id) = ?",
                    (subj_clean, subj_clean),
                ).fetchone()

                if ware_row:
                    res = self._verify_ware_claim(claim, ware_row)
                    if res:
                        results.append(res)
                        continue

                # If entity not found or claim predicate is non-numerical/strategic (e.g. strategy advice)
                results.append(
                    DBVerificationResult(
                        claim=claim,
                        status="unverified_entity",
                        details=f"Entity or attribute '{claim.subject}'/'{claim.predicate}' not directly mappable to structured SQL table.",
                    )
                )
        finally:
            if should_close and target_conn:
                target_conn.close()
            elif target_conn:
                # Restore previous pragma state so caller's connection isn't modified
                target_conn.execute(f"PRAGMA query_only = {'ON' if prev_query_only else 'OFF'};")

        return results

    def _verify_ship_claim(self, claim: TypedClaim, ship_row: tuple) -> Optional[DBVerificationResult]:
        field_map = {
            "cargo_capacity": ship_row[2],
            "cargo": ship_row[2],
            "hull": ship_row[3],
            "shields": ship_row[4],
            "speed": ship_row[5],
            "weapon_slots": ship_row[6],
            "turret_slots": ship_row[7],
            "shield_slots": ship_row[8],
        }

        pred_clean = claim.predicate.strip().lower()
        if pred_clean not in field_map:
            return None

        db_val = field_map[pred_clean]
        claim_num = _extract_first_number(claim.object)
        if claim_num is not None:
            if abs(claim_num - float(db_val)) < 1e-3:
                return DBVerificationResult(
                    claim=claim,
                    status="verified",
                    db_value=db_val,
                )
            else:
                return DBVerificationResult(
                    claim=claim,
                    status="mismatch",
                    db_value=db_val,
                    details=f"Database fact mismatch for ship {ship_row[1]} property '{pred_clean}': claimed {claim_num} vs DB {db_val}.",
                )
        else:
            return DBVerificationResult(
                claim=claim,
                status="mismatch",
                db_value=db_val,
                details=f"Non-numeric claimed value '{claim.object}' for numerical property '{pred_clean}'.",
            )

    def _verify_ware_claim(self, claim: TypedClaim, ware_row: tuple) -> Optional[DBVerificationResult]:
        field_map = {
            "min_price": ware_row[2],
            "avg_price": ware_row[3],
            "max_price": ware_row[4],
            "volume": ware_row[5],
        }

        pred_clean = claim.predicate.strip().lower()
        if pred_clean not in field_map:
            return None

        db_val = field_map[pred_clean]
        claim_num = _extract_first_number(claim.object)
        if claim_num is not None:
            if abs(claim_num - float(db_val)) < 1e-3:
                return DBVerificationResult(
                    claim=claim,
                    status="verified",
                    db_value=db_val,
                )
            else:
                return DBVerificationResult(
                    claim=claim,
                    status="mismatch",
                    db_value=db_val,
                    details=f"Database fact mismatch for ware {ware_row[1]} property '{pred_clean}': claimed {claim_num} vs DB {db_val}.",
                )
        else:
            return DBVerificationResult(
                claim=claim,
                status="mismatch",
                db_value=db_val,
                details=f"Non-numeric claimed value '{claim.object}' for numerical property '{pred_clean}'.",
            )

DOMAIN_DISCRIMINATORS = {"container", "solid", "liquid", "mineral", "gas", "buy", "sell", "import", "export"}


def _compare_objects(obj1: str, obj2: str) -> Tuple[bool, str]:
    """Compares claim objects with embedded number extraction, domain term checks, and conservative fuzzy threshold."""
    num1 = _extract_first_number(obj1)
    num2 = _extract_first_number(obj2)

    if num1 is not None and num2 is not None:
        if abs(num1 - num2) < 1e-3:
            return False, ""
        return True, f"Numeric diff ({num1} != {num2})"

    # Domain discriminator check: if key game domain terms differ, flag as mismatch
    words1 = set(re.findall(r"\b\w+\b", obj1.lower()))
    words2 = set(re.findall(r"\b\w+\b", obj2.lower()))
    disc1 = words1.intersection(DOMAIN_DISCRIMINATORS)
    disc2 = words2.intersection(DOMAIN_DISCRIMINATORS)
    if disc1 and disc2 and disc1 != disc2:
        return True, f"Domain keyword mismatch: C1={disc1} vs C2={disc2}"

    # Conservative free-text fuzzy ratio comparison (threshold 0.85)
    sim = _string_similarity(obj1, obj2)
    if sim >= 0.85:
        return False, ""
    return True, f"String diff (sim={sim:.2f}: '{obj1}' != '{obj2}')"
