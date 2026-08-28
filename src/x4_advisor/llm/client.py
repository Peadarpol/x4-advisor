"""Ollama HTTP client for router classification and grounded synthesis."""

import http.client
import json
import logging
import socket
import threading
import time
from typing import Any, Dict, List, Optional
import urllib.parse

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


class OllamaCancelledError(RuntimeError):
    """Raised when an in-flight Ollama generation request is cancelled by caller."""

    pass


def _cancel_socket(conn: Optional[http.client.HTTPConnection]) -> None:
    """Closes socket and connection safely on cancellation."""
    if conn is None:
        return
    try:
        sock = getattr(conn, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except (OSError, AttributeError):
                pass
            try:
                conn.sock = None
            except (OSError, AttributeError):
                pass
        conn.close()
    except (OSError, AttributeError):
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
        self.call_history: List[Dict[str, Any]] = []

    def clear_history(self) -> None:
        """Clears accumulated call telemetry history."""
        self.call_history.clear()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        format: Optional[Dict[str, Any] | str] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Executes a chat completion call with optional tool definitions or structured format."""
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

        if format:
            payload["format"] = format

        if options:
            payload["options"] = options

        timeout_sec = timeout if timeout is not None else self.timeout_synthesizer
        res = self._post_json(url, payload, timeout_sec, cancel_event=cancel_event)
        self._record_telemetry("chat", res)
        return res

    def generate(
        self,
        prompt: str,
        format: Optional[Dict[str, Any] | str] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
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

        if format:
            payload["format"] = format

        if options:
            payload["options"] = options

        timeout_sec = timeout if timeout is not None else self.timeout_synthesizer
        res = self._post_json(url, payload, timeout_sec, cancel_event=cancel_event)
        self._record_telemetry("generate", res)
        return res

    def _record_telemetry(self, call_type: str, response: Dict[str, Any]) -> None:
        """Records token counts and execution durations from Ollama response."""
        telemetry = {
            "call_type": call_type,
            "model": response.get("model", self.model_name),
            "prompt_eval_count": response.get("prompt_eval_count", 0),
            "eval_count": response.get("eval_count", 0),
            "prompt_eval_duration_ms": round(response.get("prompt_eval_duration", 0) / 1_000_000, 2),
            "eval_duration_ms": round(response.get("eval_duration", 0) / 1_000_000, 2),
            "total_duration_ms": round(response.get("total_duration", 0) / 1_000_000, 2),
            "load_duration_ms": round(response.get("load_duration", 0) / 1_000_000, 2),
            "done_reason": response.get("done_reason", "stop"),
        }
        self.call_history.append(telemetry)

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
        except OllamaCancelledError:
            raise
        except Exception as e:
            logger.warning("LLM warmup call encountered an error: %s", e)

        if embedder is not None:
            logger.info("Warming up Embedding model '%s' in Ollama...", getattr(embedder, "model_name", "unknown"))
            try:
                embedder.embed_text("warmup")
                logger.info("Embedding model successfully warmed up.")
            except OllamaCancelledError:
                raise
            except Exception as e:
                logger.warning("Embedding warmup call encountered an error: %s", e)

    def _post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout_sec: float,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Sends POST request to Ollama using http.client on worker thread with cancel & timeout protection."""
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path
        if parsed.query:
            path = f"{path}?{parsed.query}"

        if parsed.scheme == "https":
            conn = http.client.HTTPSConnection(host, port, timeout=timeout_sec)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout_sec)

        done_event = threading.Event()
        result_box: List[Dict[str, Any]] = []
        error_box: List[BaseException] = []

        def worker() -> None:
            try:
                data_bytes = json.dumps(payload).encode("utf-8")
                conn.request(
                    "POST",
                    path,
                    body=data_bytes,
                    headers={"Content-Type": "application/json", "User-Agent": "x4-advisor"},
                )
                resp = conn.getresponse()
                status = resp.status
                body = resp.read().decode("utf-8", errors="replace")
                if status == 404:
                    raise OllamaModelNotFoundError(
                        f"Model '{self.model_name}' was not found by Ollama runtime: {body}"
                    )
                if status != 200:
                    raise OllamaConnectionError(
                        f"Ollama API returned HTTP {status}: {body}"
                    )
                try:
                    parsed_json = json.loads(body)
                    result_box.append(parsed_json)
                except json.JSONDecodeError as jde:
                    raise OllamaConnectionError(
                        f"Malformed JSON payload returned from Ollama '{url}': {jde}"
                    ) from jde
            except BaseException as exc:
                error_box.append(exc)
            finally:
                done_event.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        start_time = time.monotonic()
        try:
            while not done_event.wait(0.05):
                if cancel_event and cancel_event.is_set():
                    raise OllamaCancelledError("Request cancelled by caller")
                if time.monotonic() - start_time > timeout_sec:
                    raise OllamaTimeoutError(
                        f"Ollama call to '{url}' exceeded wall-clock timeout of {timeout_sec:.1f}s."
                    )
        except BaseException:
            _cancel_socket(conn)
            raise

        if error_box:
            err = error_box[0]
            if isinstance(err, (OllamaModelNotFoundError, OllamaTimeoutError, OllamaCancelledError, OllamaConnectionError)):
                raise err
            if isinstance(err, (socket.timeout, TimeoutError)):
                raise OllamaTimeoutError(
                    f"Ollama call to '{url}' timed out after {time.monotonic() - start_time:.2f}s (budget: {timeout_sec:.1f}s): {err}"
                ) from err
            if isinstance(err, (ConnectionRefusedError, http.client.RemoteDisconnected, OSError)):
                raise OllamaConnectionError(
                    f"Failed to connect to Ollama endpoint at '{self.endpoint}': {err}"
                ) from err
            raise OllamaConnectionError(f"Unexpected error communicating with Ollama: {err}") from err

        return result_box[0]
