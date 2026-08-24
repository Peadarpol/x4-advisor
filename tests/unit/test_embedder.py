"""Unit tests for OllamaEmbedder client using mocked HTTP responses."""

import io
import json
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from x4_advisor.embeddings.ollama_embedder import OllamaEmbedder, OllamaEmbeddingError


def test_embed_text_success():
    """Verifies successful single embedding generation via /api/embed."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps(
        {"embeddings": [[0.1, 0.2, 0.3]]}
    ).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        embedder = OllamaEmbedder(endpoint="http://localhost:11434")
        vec = embedder.embed_text("Test sentence")
        assert vec == [0.1, 0.2, 0.3]


def test_embed_batch_success():
    """Verifies successful batch embedding generation."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps(
        {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
    ).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        embedder = OllamaEmbedder(endpoint="http://localhost:11434")
        vecs = embedder.embed_batch(["Text 1", "Text 2"])
        assert len(vecs) == 2
        assert vecs[0] == [0.1, 0.2]
        assert vecs[1] == [0.3, 0.4]


def test_embed_empty_batch():
    """Verifies passing empty batch returns empty list without calling API."""
    embedder = OllamaEmbedder()
    assert embedder.embed_batch([]) == []


def test_embed_api_unreachable_raises_error():
    """Verifies connection failure raises OllamaEmbeddingError."""
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        embedder = OllamaEmbedder(endpoint="http://localhost:11434")
        with pytest.raises(OllamaEmbeddingError, match="Failed to connect to Ollama endpoint"):
            embedder.embed_text("Test query")
