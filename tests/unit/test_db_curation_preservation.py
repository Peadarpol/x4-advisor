import sqlite3
from pathlib import Path
import pytest
import struct

from x4_advisor.storage.db import atomic_ingest_to_db, get_connection
from x4_advisor.storage.schema import init_db_schema


def _dummy_populate(conn: sqlite3.Connection):
    conn.execute("INSERT INTO factions (id, name) VALUES ('argon', 'Argon Federation');")
    return 1, 0


def test_atomic_ingest_first_run(tmp_path: Path):
    """First-run with no pre-existing target database succeeds cleanly."""
    target_db = tmp_path / "test_first_run.db"
    inserted, skipped = atomic_ingest_to_db(target_db, _dummy_populate)
    assert inserted == 1
    assert skipped == 0
    assert target_db.exists()

    conn = get_connection(target_db)
    try:
        count = conn.execute("SELECT count(*) FROM factions").fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_atomic_ingest_preserves_curation_data_and_vec0(tmp_path: Path):
    """Target DB with curation records and vec0 embeddings is preserved bit-exactly across atomic swap."""
    target_db = tmp_path / "test_curation.db"
    
    # 1. Initialize DB with domain data and curation data
    conn = get_connection(target_db)
    init_db_schema(conn)
    
    # Insert curation chain: source_registry -> source_manifest -> knowledge_chunks -> knowledge_chunks_vec
    conn.execute(
        """
        INSERT INTO source_registry (source_id, url, title, proposed_by, category, proposed_date, status)
        VALUES ('src_001', 'https://example.com', 'Test Source', 'agent_test', 'guide', '2026-08-28', 'trusted');
        """
    )
    conn.execute(
        """
        INSERT INTO source_manifest (manifest_id, source_id, title, file_path, curation_status, raw_hash)
        VALUES ('man_001', 'src_001', 'Test Manifest', 'data/sources/test.json', 'approved', 'rawhash123');
        """
    )
    conn.execute(
        """
        INSERT INTO knowledge_chunks (id, manifest_id, heading_hierarchy, chunk_index, content, token_count, source_attribution, created_at)
        VALUES ('chunk_001', 'man_001', 'Fleet > Commands', 0, 'Fleet chunk text', 10, 'Test Source', '2026-08-28T00:00:00Z');
        """
    )
    
    # Pack 1024 floats as binary embedding
    sample_floats = [0.123456 + i * 0.0001 for i in range(1024)]
    embedding_bytes = struct.pack(f"{len(sample_floats)}f", *sample_floats)
    
    has_vec = False
    try:
        conn.execute("INSERT INTO knowledge_chunks_vec (chunk_id, embedding) VALUES (?, ?)", ('chunk_001', embedding_bytes))
        has_vec = True
    except sqlite3.OperationalError:
        pass  # If sqlite-vec extension not compiled in this test environment
        
    conn.commit()
    conn.close()

    # 2. Run atomic ingestion with new populate function
    def _new_populate(new_conn: sqlite3.Connection):
        new_conn.execute("INSERT INTO factions (id, name) VALUES ('argon', 'Argon Federation');")
        new_conn.execute("INSERT INTO factions (id, name) VALUES ('paranid', 'Godrealm');")
        return 2, 0

    inserted, skipped = atomic_ingest_to_db(target_db, _new_populate)
    assert inserted == 2
    assert skipped == 0

    # 3. Verify domain data was updated AND curation data was preserved
    conn = get_connection(target_db)
    try:
        faction_count = conn.execute("SELECT count(*) FROM factions").fetchone()[0]
        assert faction_count == 2

        reg_count = conn.execute("SELECT count(*) FROM source_registry").fetchone()[0]
        assert reg_count == 1

        man_count = conn.execute("SELECT count(*) FROM source_manifest").fetchone()[0]
        assert man_count == 1

        chunk_count = conn.execute("SELECT count(*) FROM knowledge_chunks").fetchone()[0]
        assert chunk_count == 1

        # Check embedding bit-exactness if table was populated
        if has_vec:
            vec_row = conn.execute("SELECT chunk_id, embedding FROM knowledge_chunks_vec WHERE chunk_id = 'chunk_001'").fetchone()
            assert vec_row is not None
            assert vec_row[0] == 'chunk_001'
            assert vec_row[1] == embedding_bytes
    finally:
        conn.close()


def test_atomic_ingest_abort_on_error_leaves_target_untouched(tmp_path: Path):
    """If populate_fn raises an error, the target DB is completely untouched."""
    target_db = tmp_path / "test_abort.db"
    
    # Initialize DB with 1 faction
    conn = get_connection(target_db)
    init_db_schema(conn)
    conn.execute("INSERT INTO factions (id, name) VALUES ('argon', 'Argon Federation');")
    conn.commit()
    conn.close()

    def _failing_populate(new_conn: sqlite3.Connection):
        new_conn.execute("INSERT INTO factions (id, name) VALUES ('paranid', 'Godrealm');")
        raise RuntimeError("Synthetic population failure midway")

    with pytest.raises(RuntimeError, match="Synthetic population failure midway"):
        atomic_ingest_to_db(target_db, _failing_populate)

    # Verify original target DB was NOT overwritten
    conn = get_connection(target_db)
    try:
        factions = conn.execute("SELECT id FROM factions").fetchall()
        assert len(factions) == 1
        assert factions[0][0] == "argon"
    finally:
        conn.close()
