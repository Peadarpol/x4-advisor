"""Unit tests for ClaimVerifier (fidelity check C1 vs C2 and DB fact check C1 vs M1 DB)."""

import sqlite3

import pytest

from x4_advisor.curation.claim_verifier import ClaimVerifier
from x4_advisor.curation.models import TypedClaim
from x4_advisor.storage.schema import init_db_schema


@pytest.fixture
def memory_conn() -> sqlite3.Connection:
    """Provides an isolated, in-memory SQLite database with sample M1 domain data."""
    conn = sqlite3.connect(":memory:")
    init_db_schema(conn)

    cursor = conn.cursor()
    # Factions
    cursor.execute("INSERT INTO factions (id, name, short_name) VALUES ('argon', 'Argon Federation', 'ARG')")
    # Ships
    cursor.execute(
        "INSERT INTO ships (id, name, class, hull, shields, cargo_capacity, cargo_type, speed, weapon_slots, turret_slots, shield_slots, purpose, faction_id, ware_id) "
        "VALUES ('ship_arg_m_frigate_01_a_macro', 'Cerberus Vanguard', 'ship_m', 19000.0, 1000.0, 1760.0, 'container', 300.0, 2, 2, 2, 'fight', 'argon', NULL)"
    )
    # Wares
    cursor.execute(
        "INSERT INTO wares (id, name, category, min_price, avg_price, max_price, volume) "
        "VALUES ('claytronics', 'Claytronics', 'tech', 1000, 2000, 3000, 20)"
    )
    conn.commit()
    return conn


def test_verify_fidelity_exact_match():
    """Verifies matching C1 and C2 claims return match status with zero flags."""
    verifier = ClaimVerifier()
    c1 = [TypedClaim(subject="Cerberus Vanguard", predicate="cargo_capacity", object="1760", unit="m3")]
    c2 = [TypedClaim(subject="Cerberus Vanguard", predicate="cargo_capacity", object="1760", unit="m3")]

    diffs = verifier.verify_fidelity(c1, c2)
    assert len(diffs) == 1
    assert diffs[0].status == "match"


def test_verify_fidelity_numeric_drift():
    """Verifies altered number between C1 and C2 flags numeric drift."""
    verifier = ClaimVerifier()
    c1 = [TypedClaim(subject="Cerberus Vanguard", predicate="cargo_capacity", object="1760", unit="m3")]
    c2 = [TypedClaim(subject="Cerberus Vanguard", predicate="cargo_capacity", object="1800", unit="m3")]

    diffs = verifier.verify_fidelity(c1, c2)
    assert len(diffs) == 1
    assert diffs[0].status == "mismatch"
    assert diffs[0].drift_category == "numeric"


def test_verify_fidelity_epistemic_drift():
    """Verifies shift from 'often' to 'always' in qualifier flags epistemic drift."""
    verifier = ClaimVerifier()
    c1 = [TypedClaim(subject="Mining", predicate="best_sector", object="Argon Prime", qualifier="often recommended")]
    c2 = [TypedClaim(subject="Mining", predicate="best_sector", object="Argon Prime", qualifier="always recommended")]

    diffs = verifier.verify_fidelity(c1, c2)
    assert len(diffs) == 1
    assert diffs[0].status == "mismatch"
    assert diffs[0].drift_category == "quantifier"


def test_verify_against_db_match(memory_conn):
    """Verifies claim matching database fact passes verification."""
    verifier = ClaimVerifier()
    claims = [TypedClaim(subject="Cerberus Vanguard", predicate="cargo_capacity", object="1760.0")]

    res = verifier.verify_against_db(claims, memory_conn)
    assert len(res) == 1
    assert res[0].status == "verified"
    assert res[0].db_value == 1760.0


def test_verify_against_db_mismatch(memory_conn):
    """Verifies incorrect claimed value flags database mismatch."""
    verifier = ClaimVerifier()
    claims = [TypedClaim(subject="Cerberus Vanguard", predicate="cargo_capacity", object="2000.0")]

    res = verifier.verify_against_db(claims, memory_conn)
    assert len(res) == 1
    assert res[0].status == "mismatch"
    assert res[0].db_value == 1760.0


def test_verify_against_db_connection_writeable_after_verification(memory_conn):
    """Verifies that calling verify_against_db resets PRAGMA query_only = OFF so connection remains writeable."""
    verifier = ClaimVerifier()
    claims = [TypedClaim(subject="Cerberus Vanguard", predicate="cargo_capacity", object="1760.0")]

    # Execute read-only verification
    verifier.verify_against_db(claims, memory_conn)

    # Perform write operation on same connection (must not raise sqlite3.OperationalError: read-only)
    cursor = memory_conn.cursor()
    cursor.execute(
        "INSERT INTO source_registry (source_id, url, title, proposed_by, category, proposed_date) "
        "VALUES ('s1', 'http://example.com', 'Test Source', 'peter_manual', 'forum_guide', '2026-08-24')"
    )
    cursor.execute(
        "INSERT INTO source_manifest (manifest_id, source_id, title, file_path, curation_status, raw_hash) "
        "VALUES ('m1', 's1', 'Test', 'path', 'approved', 'hash')"
    )
    memory_conn.commit()
    row = cursor.execute("SELECT manifest_id FROM source_manifest WHERE manifest_id = 'm1'").fetchone()
    assert row[0] == "m1"


def test_verify_against_db_with_db_path(tmp_path):
    """Verifies verify_against_db when passed a Path object directly instead of a Connection."""
    db_file = tmp_path / "test_verify.db"
    conn = sqlite3.connect(str(db_file))
    init_db_schema(conn)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO factions (id, name, short_name) VALUES ('argon', 'Argon Federation', 'ARG')")
    cursor.execute(
        "INSERT INTO ships (id, name, class, hull, shields, cargo_capacity, cargo_type, speed, weapon_slots, turret_slots, shield_slots, purpose, faction_id, ware_id) "
        "VALUES ('ship_arg_m_frigate_01_a_macro', 'Cerberus Vanguard', 'ship_m', 19000.0, 1000.0, 1760.0, 'container', 300.0, 2, 2, 2, 'fight', 'argon', NULL)"
    )
    conn.commit()
    conn.close()

    verifier = ClaimVerifier()
    claims = [TypedClaim(subject="Cerberus Vanguard", predicate="cargo_capacity", object="1760.0")]

    res = verifier.verify_against_db(claims, db_path=db_file)
    assert len(res) == 1
    assert res[0].status == "verified"
    assert res[0].db_value == 1760.0


def test_verify_fidelity_subject_fallback_matching():
    """Verifies subject-only fallback matching when predicate wording varies (e.g. type vs category)."""
    verifier = ClaimVerifier()
    c1 = [TypedClaim(subject="UI Extensions and HUD", predicate="type", object="quality-of-life mod")]
    c2 = [TypedClaim(subject="UI Extensions and HUD", predicate="category", object="quality-of-life mod")]

    diffs = verifier.verify_fidelity(c1, c2)
    assert len(diffs) == 1
    assert diffs[0].status == "match"


def test_verify_fidelity_qualified_numeric_matching():
    """Verifies numeric extraction parses embedded numbers from qualified strings (e.g. '1' vs 'about 1')."""
    verifier = ClaimVerifier()
    c1 = [TypedClaim(subject="10x Modules", predicate="build_time", object="1")]
    c2 = [TypedClaim(subject="10x Modules", predicate="build_time", object="about 1")]

    diffs = verifier.verify_fidelity(c1, c2)
    assert len(diffs) == 1
    assert diffs[0].status == "match"


def test_verify_fidelity_conservative_fuzzy_threshold_flags_domain_diff():
    """Verifies conservative fuzzy threshold (0.85) flags meaningful domain differences like container vs solid storage."""
    verifier = ClaimVerifier()
    c1 = [TypedClaim(subject="Builders Can Haul", predicate="effect", object="gives builder ships container storage space")]
    c2 = [TypedClaim(subject="Builders Can Haul", predicate="effect", object="gives builder ships solid storage space")]

    diffs = verifier.verify_fidelity(c1, c2)
    assert len(diffs) == 1
    assert diffs[0].status == "mismatch"
    assert diffs[0].drift_category == "entity"


def test_verify_fidelity_multi_candidate_subject_scoring():
    """Verifies that multiple claims sharing a subject score candidates and select their correct counterpart."""
    verifier = ClaimVerifier()
    c1 = [
        TypedClaim(subject="Kha'ak installation", predicate="consequence_if_not_destroyed", object="continues spawning raiders"),
        TypedClaim(subject="Kha'ak installation", predicate="destroyers_needed", object="3 destroyers required"),
    ]
    # C2 has candidate claims in reversed order with phrasing variations
    c2 = [
        TypedClaim(subject="Kha'ak installation", predicate="requirement_for_destruction", object="needs 3 destroyers"),
        TypedClaim(subject="Kha'ak installation", predicate="result_if_spared", object="will continue spawning raiders"),
    ]

    diffs = verifier.verify_fidelity(c1, c2)
    assert len(diffs) == 2
    # Claim 1 should pair with C2[1] (spawning raiders)
    assert diffs[0].c2_claim is not None
    assert "spawning raiders" in diffs[0].c2_claim.object

    # Claim 2 should pair with C2[0] (3 destroyers)
    assert diffs[1].c2_claim is not None
    assert "3 destroyers" in diffs[1].c2_claim.object




