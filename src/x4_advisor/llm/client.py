"""Ollama HTTP client for router classification and grounded synthesis."""

import json
import logging
import socket
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


class OllamaConnectionError(RuntimeError):
    """Raised when Ollama HTTP endpoint cannot be reached."""

    pass


class OllamaModelNotFoundError(RuntimeError):
    """Raised when the requested model is not found in the Ollama runtime."""

    pass


class OllamaTimeoutError(RuntimeError):
    """Raised when an Ollama API call exceeds its wall-clock timeout budget."""

    pass


class OllamaClient:
    """Client for executing chat completions and tool-calling via Ollama HTTP API."""

    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model_name: str = "gemma4:12b",
        keep_alive: str = "10m",
        timeout_router: float = 15.0,
        timeout_synthesizer: float = 25.0,
    ) -> None:
        """Initializes Ollama HTTP client.

        Note on Timeouts:
            The `timeout_router` (15.0s) and `timeout_synthesizer` (25.0s) values serve
            strictly as socket-level hang-protection circuit breakers to terminate runaway
            or orphaned HTTP requests. They do NOT enforce the end-to-end user SLA (<20s single,
            <30s hybrid), which is measured and asserted separately in integration tests.
        """
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.keep_alive = keep_alive
        self.timeout_router = timeout_router
        self.timeout_synthesizer = timeout_synthesizer

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Executes a chat completion call with optional tool definitions."""
        url = f"{self.endpoint}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
        }

        if tools:
            payload["tools"] = tools

        if options:
            payload["options"] = options

        timeout_sec = timeout if timeout is not None else self.timeout_synthesizer
        return self._post_json(url, payload, timeout_sec)

    def generate(
        self,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Executes a text generation call via /api/generate."""
        url = f"{self.endpoint}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
        }

        if options:
            payload["options"] = options

        timeout_sec = timeout if timeout is not None else self.timeout_synthesizer
        return self._post_json(url, payload, timeout_sec)

    def warmup(self, embedder: Optional[Any] = None) -> None:
        """Pre-loads weights for both the LLM and Embedding models into VRAM."""
        logger.info("Warming up LLM model '%s' in Ollama...", self.model_name)
        try:
            self.chat(
                messages=[{"role": "user", "content": "ping"}],
                options={"num_predict": 1, "num_ctx": 2048},
                timeout=60.0,
            )
            logger.info("LLM model '%s' successfully warmed up.", self.model_name)
        except Exception as e:
            logger.warning("LLM warmup call encountered an error: %s", e)

        if embedder is not None:
            logger.info("Warming up Embedding model '%s' in Ollama...", getattr(embedder, "model_name", "unknown"))
            try:
                embedder.embed_text("warmup")
                logger.info("Embedding model successfully warmed up.")
            except Exception as e:
                logger.warning("Embedding warmup call encountered an error: %s", e)

    def _post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout_sec: float,
    ) -> Dict[str, Any]:
        """Sends POST request to Ollama and parses JSON response with strict wall-clock timeout."""
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data_bytes,
            headers={"Content-Type": "application/json", "User-Agent": "x4-advisor"},
            method="POST",
        )

        start_time = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                elapsed = time.time() - start_time
                if elapsed > timeout_sec:
                    raise OllamaTimeoutError(
                        f"Ollama call to '{url}' exceeded wall-clock timeout of {timeout_sec:.1f}s (took {elapsed:.2f}s)."
                    )

                if resp.status != 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    raise OllamaConnectionError(
                        f"Ollama API returned HTTP {resp.status}: {body}"
                    )

                res_body = json.loads(resp.read().decode("utf-8"))
                return res_body

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            if e.code == 404:
                raise OllamaModelNotFoundError(
                    f"Model '{self.model_name}' was not found by Ollama runtime: {body}"
                ) from e
            raise OllamaConnectionError(
                f"HTTP {e.code} error from Ollama endpoint: {body}"
            ) from e

        except (socket.timeout, TimeoutError) as e:
            elapsed = time.time() - start_time
            raise OllamaTimeoutError(
                f"Ollama call to '{url}' timed out after {elapsed:.2f}s (budget: {timeout_sec:.1f}s): {e}"
            ) from e

        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                elapsed = time.time() - start_time
                raise OllamaTimeoutError(
                    f"Ollama call to '{url}' timed out after {elapsed:.2f}s (budget: {timeout_sec:.1f}s): {e}"
                ) from e
            raise OllamaConnectionError(
                f"Failed to connect to Ollama endpoint at '{self.endpoint}': {e}"
            ) from e

        except json.JSONDecodeError as e:
            raise OllamaConnectionError(
                f"Malformed JSON payload returned from Ollama '{url}': {e}"
            ) from e
