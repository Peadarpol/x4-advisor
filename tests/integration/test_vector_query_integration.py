"""Integration tests for VectorQueryEngine against live database and Ollama.

These tests require:
- A populated database at data/db/x4_advisor.db (from M3 ingestion)
- A running Ollama endpoint with qwen3-embedding:0.6b loaded

Skipped gracefully if either prerequisite is unavailable.
"""

from pathlib import Path

import pytest

from x4_advisor.config import get_config, ConfigError
from x4_advisor.embeddings.ollama_embedder import OllamaEmbedder, OllamaEmbeddingError
from x4_advisor.retrieval.vector_query import VectorQueryEngine


def _ollama_available(endpoint: str, model: str) -> bool:
    """Check if Ollama endpoint is reachable and can embed."""
    try:
        embedder = OllamaEmbedder(endpoint=endpoint, model_name=model, timeout_seconds=30.0)
        embedder.embed_text("test")
        return True
    except (OllamaEmbeddingError, Exception):
        return False


@pytest.mark.integration
class TestVectorQueryIntegration:
    """Live integration tests against real database and Ollama."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Skip all tests if prerequisites are not met."""
        try:
            self.config = get_config(validate=False)
        except ConfigError:
            pytest.skip("Config not available")

        self.db_path = self.config.database_path
        if not self.db_path.exists():
            pytest.skip(f"Database not found at {self.db_path}")

        if not _ollama_available(self.config.ollama_endpoint, self.config.embedding_model):
            pytest.skip("Ollama endpoint not available or embedding model not loaded")

        self.embedder = OllamaEmbedder(
            endpoint=self.config.ollama_endpoint,
            model_name=self.config.embedding_model,
        )

    def test_live_vector_search_end_to_end(self) -> None:
        """Searches real ingested community guide chunks and asserts relevant results."""
        with VectorQueryEngine(
            db_path=self.db_path,
            embedder=self.embedder,
            default_threshold=self.config.vector_relevance_threshold,
        ) as engine:
            result = engine.search(
                "How do I use DeadTater for transport ships?",
                top_k=3,
            )

        assert result.status == "success", f"Expected success, got {result.status}: {result.message}"
        assert len(result.chunks) > 0
        assert result.total_candidates >= len(result.chunks)

        # Assert similarity is above configured threshold, not a pinned value
        for chunk in result.chunks:
            assert chunk.similarity_score > self.config.vector_relevance_threshold
            assert chunk.similarity_score <= 1.0
            assert chunk.distance >= 0.0
            assert chunk.game_version_scope == "base_game"
            assert chunk.content  # Non-empty content
            assert chunk.source_attribution  # Has attribution
            assert chunk.manifest_id  # Has manifest linkage

    def test_dissimilar_query_returns_low_similarity(self) -> None:
        """A completely unrelated query returns chunks below threshold or no results."""
        with VectorQueryEngine(
            db_path=self.db_path,
            embedder=self.embedder,
            default_threshold=self.config.vector_relevance_threshold,
        ) as engine:
            result = engine.search(
                "What is the quantum mechanics equation for black hole event horizons?",
                top_k=3,
            )

        # Either no chunks pass the threshold, or any that do have low similarity
        if result.status == "no_relevant_chunks":
            assert result.chunks == []
        else:
            # If somehow some pass, they should be much lower than a relevant query
            for chunk in result.chunks:
                assert chunk.similarity_score < 0.5
