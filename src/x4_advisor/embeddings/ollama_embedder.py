"""Ollama vector embedding client shared across ingestion curation and query retrieval."""

import json
import logging
from typing import List, Optional
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


class OllamaEmbeddingError(RuntimeError):
    """Raised when an embedding request to Ollama fails or returns an invalid payload."""

    pass


class OllamaEmbedder:
    """Client for generating vector embeddings via Ollama HTTP API."""

    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model_name: str = "qwen3-embedding:0.6b",
        timeout_seconds: float = 30.0,
        keep_alive: str = "10m",
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive

    def embed_text(self, text: str) -> List[float]:
        """Generates a dense vector embedding for a single text string."""
        results = self.embed_batch([text])
        if not results:
            raise OllamaEmbeddingError("Ollama API returned an empty embedding result.")
        return results[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for a list of text strings."""
        if not texts:
            return []

        url = f"{self.endpoint}/api/embed"
        payload = {
            "model": self.model_name,
            "input": texts,
            "keep_alive": self.keep_alive,
        }
        data_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url=url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                if resp.status != 200:
                    raise OllamaEmbeddingError(
                        f"Ollama API returned HTTP {resp.status}: {resp.read().decode('utf-8')}"
                    )
                res_body = json.loads(resp.read().decode("utf-8"))

        except urllib.error.URLError as e:
            # Fallback to single /api/embeddings endpoint if /api/embed batch endpoint fails or isn't supported
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                return [self._embed_single_legacy(t) for t in texts]
            raise OllamaEmbeddingError(
                f"Failed to connect to Ollama endpoint at '{self.endpoint}': {e}"
            ) from e
        except Exception as e:
            raise OllamaEmbeddingError(f"Error during Ollama embedding call: {e}") from e

        embeddings = res_body.get("embeddings")
        if not embeddings or not isinstance(embeddings, list):
            raise OllamaEmbeddingError(
                f"Invalid payload format from Ollama /api/embed: missing 'embeddings' array."
            )

        return embeddings

    def _embed_single_legacy(self, text: str) -> List[float]:
        """Legacy fallback endpoint POST /api/embeddings for single text string."""
        url = f"{self.endpoint}/api/embeddings"
        payload = {
            "model": self.model_name,
            "prompt": text,
            "keep_alive": self.keep_alive,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                if resp.status != 200:
                    raise OllamaEmbeddingError(
                        f"Ollama API returned HTTP {resp.status}: {resp.read().decode('utf-8')}"
                    )
                res_body = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise OllamaEmbeddingError(f"Failed legacy /api/embeddings request: {e}") from e

        embedding = res_body.get("embedding")
        if not embedding or not isinstance(embedding, list):
            raise OllamaEmbeddingError("Invalid payload format from Ollama /api/embeddings.")
        return embedding
