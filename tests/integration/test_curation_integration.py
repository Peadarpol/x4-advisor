"""Integration tests for unstructured curation pipeline against golden unstructured fixtures."""

import json
from pathlib import Path
import sqlite3

import pytest

from x4_advisor.curation.claim_verifier import ClaimVerifier
from x4_advisor.curation.models import TypedClaim
from x4_advisor.storage.schema import init_db_schema


@pytest.fixture
def golden_fixtures_path() -> Path:
    return Path("tests/fixtures/golden_unstructured")


def test_golden_unstructured_claims_loading(golden_fixtures_path: Path):
    """Verifies that golden unstructured claims C1 and C2 load cleanly."""
    c1_file = golden_fixtures_path / "extracted_claims.json"
    c2_file = golden_fixtures_path / "reextracted_claims.json"

    assert c1_file.exists()
    assert c2_file.exists()

    with open(c1_file, "r", encoding="utf-8") as f:
        c1_data = json.load(f)
    with open(c2_file, "r", encoding="utf-8") as f:
        c2_data = json.load(f)

    c1_claims = [TypedClaim.from_dict(d) for d in c1_data]
    c2_claims = [TypedClaim.from_dict(d) for d in c2_data]

    assert len(c1_claims) > 50
    assert len(c2_claims) > 50

    verifier = ClaimVerifier()
    diffs = verifier.verify_fidelity(c1_claims, c2_claims)
    assert len(diffs) == len(c1_claims)


def test_schema_idempotency():
    """Verifies that init_db_schema can be safely executed multiple times on the same connection."""
    conn = sqlite3.connect(":memory:")
    init_db_schema(conn)
    init_db_schema(conn)

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM source_registry")
    conn.close()


def test_curation_cli_ingest_force_guard(tmp_path: Path, golden_fixtures_path: Path):
    """Verifies that ingest_manifest respects force=False and requires force=True to overwrite existing chunks."""
    from unittest.mock import MagicMock, patch
    from x4_advisor.curation.cli import approve_manifest, ingest_manifest, register_source, verify_source_claims

    db_file = tmp_path / "curation_test.db"
    c1_path = golden_fixtures_path / "extracted_claims.json"
    c2_path = golden_fixtures_path / "reextracted_claims.json"
    p_path = golden_fixtures_path / "paraphrased_content.md"

    register_source(db_file, "src_t1", "http://test", "Test Source", "peter_manual", "forum_guide")
    verify_source_claims(db_file, "man_t1", "src_t1", "Test Source", c1_path, c2_path)
    approve_manifest(db_file, "man_t1")

    # Mock OllamaEmbedder so test runs without network
    mock_embedder = MagicMock()
    mock_embedder.embed_text.return_value = [0.1] * 1024

    with patch("x4_advisor.curation.cli.OllamaEmbedder", return_value=mock_embedder):
        # Initial Ingestion (force=False)
        ingest_manifest(db_file, "man_t1", p_path, c1_path, force=False)

        conn = sqlite3.connect(str(db_file))
        c_count1 = conn.execute("SELECT COUNT(*) FROM knowledge_chunks WHERE manifest_id = 'man_t1'").fetchone()[0]
        conn.close()
        assert c_count1 > 0

        # Attempt Second Ingestion with force=False (should be no-op)
        ingest_manifest(db_file, "man_t1", p_path, c1_path, force=False)

        conn = sqlite3.connect(str(db_file))
        c_count2 = conn.execute("SELECT COUNT(*) FROM knowledge_chunks WHERE manifest_id = 'man_t1'").fetchone()[0]
        conn.close()
        assert c_count2 == c_count1

        # Second Ingestion with force=True (should clear and re-ingest)
        ingest_manifest(db_file, "man_t1", p_path, c1_path, force=True)

        conn = sqlite3.connect(str(db_file))
        c_count3 = conn.execute("SELECT COUNT(*) FROM knowledge_chunks WHERE manifest_id = 'man_t1'").fetchone()[0]
        conn.close()
        assert c_count3 == c_count1
