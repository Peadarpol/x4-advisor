"""Vector query engine executing KNN similarity search against sqlite-vec."""

import logging
import sqlite3
import struct
from pathlib import Path
from typing import List, Optional

from x4_advisor.embeddings.ollama_embedder import OllamaEmbedder, OllamaEmbeddingError
from x4_advisor.retrieval.models import RetrievedChunk, VectorSearchResult

logger = logging.getLogger(__name__)

# CTE query isolating KNN vector index search before relational join/filter.
# Defensive structural default — guarantees sqlite-vec's MATCH/k executes on
# the vec0 virtual table before any relational predicate, regardless of how
# the query planner might reorder at larger corpus scales.
_VECTOR_SEARCH_SQL = """
WITH vec_search AS (
    SELECT chunk_id, distance
    FROM knowledge_chunks_vec
    WHERE embedding MATCH ? AND k = ?
)
SELECT
    kc.id, kc.manifest_id, kc.heading_hierarchy, kc.content,
    kc.source_attribution, kc.topic, kc.game_version_scope, v.distance
FROM vec_search v
JOIN knowledge_chunks kc ON kc.id = v.chunk_id
WHERE kc.game_version_scope = ?
ORDER BY v.distance ASC
"""


class VectorQueryEngine:
    """KNN similarity search engine over embedded knowledge chunks.

    Mirrors StructuredQueryEngine's constructor pattern: accepts either
    db_path (engine owns and closes the connection) or an external conn
    (caller owns lifetime). Sets PRAGMA query_only = ON once in __init__.

    Args:
        db_path: Path to the SQLite database file. Mutually exclusive with conn.
        conn: Pre-existing SQLite connection with sqlite-vec already loaded.
            Mutually exclusive with db_path.
        embedder: OllamaEmbedder instance for query vectorization. If None,
            a default instance is created.
        default_threshold: Minimum cosine similarity for results. M5 must pass
            config.vector_relevance_threshold here — the 0.40 default is a
            placeholder that should not silently become the production value.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        conn: Optional[sqlite3.Connection] = None,
        embedder: Optional[OllamaEmbedder] = None,
        default_threshold: float = 0.40,
    ) -> None:
        if conn is not None:
            self.conn = conn
            self._close_conn_on_exit = False
        elif db_path is not None:
            import sqlite_vec

            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(db_path))
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self._close_conn_on_exit = True
        else:
            raise ValueError(
                "Either db_path or conn must be provided to VectorQueryEngine."
            )

        self.conn.execute("PRAGMA query_only = ON;")
        self._embedder = embedder or OllamaEmbedder()
        self._default_threshold = default_threshold

        # Pre-check table existence (SPEC-001 §8: "Database doesn't exist yet"
        # must produce a clear message, not a raw SQL error).
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_chunks_vec'"
        ).fetchone()
        self._table_ready: bool = row is not None

    def close(self) -> None:
        """Closes database connection if owned by this engine instance."""
        if self._close_conn_on_exit and self.conn:
            self.conn.close()

    def __enter__(self) -> "VectorQueryEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        min_similarity: Optional[float] = None,
        game_version_scope: str = "base_game",
    ) -> VectorSearchResult:
        """Execute KNN vector similarity search against the knowledge base.

        This method never raises exceptions — all failure modes return a clean
        VectorSearchResult with an appropriate status and message.

        Args:
            query_text: Natural language query string to embed and search.
            top_k: Number of nearest-neighbor candidates to retrieve from
                the vector index (pre-threshold count).
            min_similarity: Minimum cosine similarity threshold. Chunks below
                this are filtered from the result. Falls back to
                self._default_threshold if not provided.
            game_version_scope: Filter for knowledge_chunks.game_version_scope.
                Default 'base_game' per SPEC-001 §3. Parameterized (not
                hardcoded) for Phase 2 DLC extensibility.

        Returns:
            VectorSearchResult with status, chunks, and metadata.
        """
        threshold = min_similarity if min_similarity is not None else self._default_threshold

        # Guard: empty/whitespace query
        if not query_text or not query_text.strip():
            return VectorSearchResult(
                query_text=query_text or "",
                chunks=[],
                status="embedding_failed",
                total_candidates=0,
                threshold_used=threshold,
                message="Query text is empty.",
            )

        # Guard: knowledge base not ingested yet
        if not self._table_ready:
            return VectorSearchResult(
                query_text=query_text,
                chunks=[],
                status="database_not_ready",
                total_candidates=0,
                threshold_used=threshold,
                message=(
                    "Knowledge base ingestion has not been run yet. "
                    "Run the curation CLI ingest command first."
                ),
            )

        # Embed query
        try:
            query_vector = self._embedder.embed_text(query_text)
        except OllamaEmbeddingError as e:
            return VectorSearchResult(
                query_text=query_text,
                chunks=[],
                status="embedding_failed",
                total_candidates=0,
                threshold_used=threshold,
                message=str(e),
            )

        # Serialize embedding to bytes for sqlite-vec MATCH
        query_bytes = struct.pack(f"{len(query_vector)}f", *query_vector)

        # Execute CTE vector search
        rows = self.conn.execute(
            _VECTOR_SEARCH_SQL,
            (query_bytes, top_k, game_version_scope),
        ).fetchall()

        total_candidates = len(rows)

        # Build RetrievedChunk list with similarity filtering
        chunks: List[RetrievedChunk] = []
        for row in rows:
            distance = float(row[7])
            similarity = 1.0 - distance
            if similarity >= threshold:
                chunks.append(
                    RetrievedChunk(
                        chunk_id=row[0],
                        manifest_id=row[1],
                        heading_hierarchy=row[2],
                        content=row[3],
                        similarity_score=similarity,
                        distance=distance,
                        source_attribution=row[4],
                        topic=row[5],
                        game_version_scope=row[6],
                    )
                )

        status = "success" if chunks else "no_relevant_chunks"

        return VectorSearchResult(
            query_text=query_text,
            chunks=chunks,
            status=status,
            total_candidates=total_candidates,
            threshold_used=threshold,
        )
