"""Component tests for VectorQueryEngine.

These are component tests, not pure unit tests: they require the sqlite-vec
native C extension (a pyproject.toml dependency always present in dev) but
mock the OllamaEmbedder to avoid requiring a live Ollama endpoint.
"""

import sqlite3
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sqlite_vec

from x4_advisor.embeddings.ollama_embedder import OllamaEmbeddingError
from x4_advisor.retrieval.models import VectorSearchResult
from x4_advisor.retrieval.vector_query import VectorQueryEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 1024


def _make_vector(seed: float) -> list[float]:
    """Creates a deterministic 1024-dim vector for testing.

    Different seeds produce vectors at different distances from each other.
    """
    import math

    return [math.sin(seed + i * 0.01) for i in range(EMBEDDING_DIM)]


def _vector_bytes(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _create_test_db(tmp_path: Path) -> Path:
    """Creates a test database with schema, sample chunks, and sample vectors."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    # Create tables
    conn.execute("PRAGMA foreign_keys = ON;")

    # source_registry (minimal for FK)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_registry (
            source_id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            proposed_by TEXT NOT NULL,
            category TEXT NOT NULL,
            proposed_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'trusted'
        )
        """
    )
    conn.execute(
        "INSERT INTO source_registry VALUES ('src_test', 'http://test', 'Test', 'test', 'other', '2026-01-01', 'trusted')"
    )

    # source_manifest (minimal for FK)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_manifest (
            manifest_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES source_registry(source_id),
            title TEXT NOT NULL,
            file_path TEXT NOT NULL,
            curation_status TEXT NOT NULL DEFAULT 'approved',
            raw_hash TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO source_manifest VALUES ('man_test', 'src_test', 'Test', '/test', 'approved', 'abc123')"
    )

    # knowledge_chunks
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_chunks (
            id TEXT PRIMARY KEY,
            manifest_id TEXT NOT NULL REFERENCES source_manifest(manifest_id),
            heading_hierarchy TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            source_attribution TEXT NOT NULL,
            topic TEXT,
            related_entity_ids TEXT,
            game_version_scope TEXT NOT NULL DEFAULT 'base_game',
            created_at TEXT NOT NULL
        )
        """
    )

    # knowledge_chunks_vec (cosine distance)
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_vec USING vec0(
            chunk_id TEXT PRIMARY KEY,
            embedding float[1024] distance_metric=cosine
        )
        """
    )

    # Insert test chunks with different scopes
    chunks = [
        ("chunk_01", "base_game", "Trading strategy for transport ships", "Trading > Transport", "trading"),
        ("chunk_02", "base_game", "Mining silicon in Argon space", "Mining > Silicon", "mining"),
        ("chunk_03", "base_game", "Fleet management basics", "Combat > Fleets", "combat"),
        ("chunk_04", "dlc_timelines", "DLC-specific timeline content", "DLC > Timelines", "dlc"),
    ]

    vecs = [
        _make_vector(1.0),  # chunk_01 — "trading" topic
        _make_vector(2.0),  # chunk_02 — "mining" topic
        _make_vector(3.0),  # chunk_03 — "combat" topic
        _make_vector(100.0),  # chunk_04 — DLC, very different vector
    ]

    for i, (cid, scope, content, heading, topic) in enumerate(chunks):
        conn.execute(
            """
            INSERT INTO knowledge_chunks (id, manifest_id, heading_hierarchy, chunk_index,
                content, token_count, source_attribution, topic, game_version_scope, created_at)
            VALUES (?, 'man_test', ?, ?, ?, 50, 'Test Source', ?, ?, '2026-01-01T00:00:00')
            """,
            (cid, heading, i, content, topic, scope),
        )
        conn.execute(
            "INSERT INTO knowledge_chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            (cid, _vector_bytes(vecs[i])),
        )

    conn.commit()
    conn.close()
    return db_path


def _mock_embedder(seed: float = 1.0) -> MagicMock:
    """Creates a mock OllamaEmbedder that returns a deterministic vector."""
    embedder = MagicMock()
    embedder.embed_text.return_value = _make_vector(seed)
    return embedder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVectorQueryEngineSuccess:
    """Tests for successful search paths."""

    def test_search_similar_chunks_success(self, tmp_path: Path) -> None:
        """KNN search returns expected chunks with similarity scores and metadata."""
        db_path = _create_test_db(tmp_path)
        # Seed 1.0 matches chunk_01's vector exactly
        embedder = _mock_embedder(seed=1.0)

        with VectorQueryEngine(db_path=db_path, embedder=embedder, default_threshold=0.0) as engine:
            result = engine.search("transport trading", top_k=3, min_similarity=0.0)

        assert result.status == "success"
        assert len(result.chunks) > 0
        assert result.chunks[0].chunk_id == "chunk_01"
        assert result.chunks[0].similarity_score > 0.0
        assert result.chunks[0].similarity_score <= 1.0
        assert result.chunks[0].distance >= 0.0
        assert result.chunks[0].manifest_id == "man_test"
        assert result.chunks[0].source_attribution == "Test Source"
        assert result.chunks[0].topic == "trading"
        assert result.total_candidates >= len(result.chunks)
        embedder.embed_text.assert_called_once_with("transport trading")

    def test_total_candidates_is_pre_threshold_count(self, tmp_path: Path) -> None:
        """total_candidates reflects KNN candidates before threshold filtering."""
        db_path = _create_test_db(tmp_path)
        # Use seed=1.5 — close to chunk_01 (seed=1.0) but not an exact match,
        # so similarity < 1.0 and a high threshold can filter it out.
        embedder = _mock_embedder(seed=1.5)

        with VectorQueryEngine(db_path=db_path, embedder=embedder) as engine:
            # Very high threshold — should filter out all non-exact matches
            result = engine.search("test", top_k=3, min_similarity=0.999)

        # KNN returned candidates, but threshold filtered them
        assert result.total_candidates > 0
        assert len(result.chunks) == 0
        assert result.status == "no_relevant_chunks"


class TestVectorQueryEngineFiltering:
    """Tests for scope and threshold filtering."""

    def test_base_game_scope_filtering(self, tmp_path: Path) -> None:
        """Non-base_game chunks are excluded when scope filter is 'base_game'."""
        db_path = _create_test_db(tmp_path)
        # Seed 100.0 matches chunk_04 (DLC) vector most closely
        embedder = _mock_embedder(seed=100.0)

        with VectorQueryEngine(db_path=db_path, embedder=embedder, default_threshold=0.0) as engine:
            result = engine.search("DLC timelines", top_k=5, min_similarity=0.0, game_version_scope="base_game")

        # chunk_04 has game_version_scope='dlc_timelines', should be excluded
        chunk_ids = [c.chunk_id for c in result.chunks]
        assert "chunk_04" not in chunk_ids
        # All returned chunks should be base_game
        for chunk in result.chunks:
            assert chunk.game_version_scope == "base_game"

    def test_relevance_threshold_filtering(self, tmp_path: Path) -> None:
        """Chunks below min_similarity are filtered out."""
        db_path = _create_test_db(tmp_path)
        # Seed=1.5 — near chunk_01 but not exact, so similarity < 1.0
        embedder = _mock_embedder(seed=1.5)

        with VectorQueryEngine(db_path=db_path, embedder=embedder) as engine:
            # Low threshold — should return results
            result_low = engine.search("test", top_k=3, min_similarity=0.0)
            # High threshold — should filter them out
            result_high = engine.search("test", top_k=3, min_similarity=0.999)

        assert result_low.status == "success"
        assert len(result_low.chunks) > 0
        assert result_high.status == "no_relevant_chunks"
        assert len(result_high.chunks) == 0

    def test_empty_results_abstention_status(self, tmp_path: Path) -> None:
        """Returns status='no_relevant_chunks' when no candidates meet threshold."""
        db_path = _create_test_db(tmp_path)
        # Seed=1.5 — near chunk_01 but not exact, so similarity < 1.0
        embedder = _mock_embedder(seed=1.5)

        with VectorQueryEngine(db_path=db_path, embedder=embedder) as engine:
            result = engine.search("anything", top_k=3, min_similarity=1.0)

        assert result.status == "no_relevant_chunks"
        assert result.chunks == []
        assert result.threshold_used == 1.0


class TestVectorQueryEngineErrorHandling:
    """Tests for failure modes — search() never raises, always returns clean status."""

    def test_database_not_ready_status(self, tmp_path: Path) -> None:
        """Returns status='database_not_ready' when knowledge_chunks_vec table is missing."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        # Deliberately do NOT create knowledge_chunks_vec
        conn.commit()

        engine = VectorQueryEngine(conn=conn, embedder=_mock_embedder())
        result = engine.search("test query")

        assert result.status == "database_not_ready"
        assert "ingestion has not been run" in result.message.lower()
        assert result.chunks == []
        assert result.total_candidates == 0
        conn.close()

    def test_empty_query_returns_embedding_failed(self, tmp_path: Path) -> None:
        """search('') returns status='embedding_failed' without calling the embedder."""
        db_path = _create_test_db(tmp_path)
        embedder = _mock_embedder()

        with VectorQueryEngine(db_path=db_path, embedder=embedder) as engine:
            result_empty = engine.search("")
            result_whitespace = engine.search("   ")

        assert result_empty.status == "embedding_failed"
        assert "empty" in result_empty.message.lower()
        assert result_whitespace.status == "embedding_failed"
        # Embedder should never have been called
        embedder.embed_text.assert_not_called()

    def test_embedding_error_returns_clean_status(self, tmp_path: Path) -> None:
        """OllamaEmbeddingError is caught, returns status='embedding_failed' with message."""
        db_path = _create_test_db(tmp_path)
        embedder = MagicMock()
        embedder.embed_text.side_effect = OllamaEmbeddingError(
            "Failed to connect to Ollama endpoint at 'http://localhost:11434'"
        )

        with VectorQueryEngine(db_path=db_path, embedder=embedder) as engine:
            result = engine.search("test query")

        assert result.status == "embedding_failed"
        assert "Ollama" in result.message
        assert result.chunks == []


class TestVectorQueryEngineSafety:
    """Tests for connection safety invariants."""

    def test_read_only_pragma_isolation(self, tmp_path: Path) -> None:
        """PRAGMA query_only = ON is enforced during search."""
        db_path = _create_test_db(tmp_path)
        embedder = _mock_embedder()

        with VectorQueryEngine(db_path=db_path, embedder=embedder) as engine:
            # Attempt a write — should fail due to query_only
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                engine.conn.execute(
                    "INSERT INTO knowledge_chunks (id, manifest_id, heading_hierarchy, chunk_index, "
                    "content, token_count, source_attribution, game_version_scope, created_at) "
                    "VALUES ('x', 'man_test', 'x', 0, 'x', 1, 'x', 'base_game', '2026-01-01')"
                )

    def test_concurrent_read_safety(self, tmp_path: Path) -> None:
        """Two VectorQueryEngine instances can query the same database concurrently."""
        db_path = _create_test_db(tmp_path)

        def run_search(seed: float) -> VectorSearchResult:
            embedder = _mock_embedder(seed=seed)
            with VectorQueryEngine(db_path=db_path, embedder=embedder, default_threshold=0.0) as engine:
                return engine.search("concurrent test", top_k=3, min_similarity=0.0)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_search, seed) for seed in [1.0, 2.0]]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 2
        for result in results:
            assert result.status == "success"
            assert len(result.chunks) > 0


class TestVectorQueryEngineContextManager:
    """Tests for constructor and lifecycle patterns."""

    def test_context_manager_closes_owned_connection(self, tmp_path: Path) -> None:
        """Connection opened from db_path is closed on __exit__."""
        db_path = _create_test_db(tmp_path)
        embedder = _mock_embedder()

        engine = VectorQueryEngine(db_path=db_path, embedder=embedder)
        assert engine._close_conn_on_exit is True
        conn_ref = engine.conn
        engine.close()

        # Connection should be closed — attempting to use it should fail
        with pytest.raises(Exception):
            conn_ref.execute("SELECT 1")

    def test_external_conn_not_closed(self, tmp_path: Path) -> None:
        """Connection passed via conn= is not closed by the engine."""
        db_path = _create_test_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        embedder = _mock_embedder()

        engine = VectorQueryEngine(conn=conn, embedder=embedder)
        assert engine._close_conn_on_exit is False
        engine.close()

        # Connection should still be usable
        result = conn.execute("SELECT 1").fetchone()
        assert result == (1,)
        conn.close()

    def test_requires_db_path_or_conn(self) -> None:
        """Raises ValueError if neither db_path nor conn is provided."""
        with pytest.raises(ValueError, match="Either db_path or conn"):
            VectorQueryEngine()
