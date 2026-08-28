"""Unit tests for OllamaClient payload construction, options, timeouts, error handling, and cancellation."""

import http.client
import json
import socket
import threading
from unittest.mock import MagicMock, patch

import pytest

from x4_advisor.llm.client import (
    OllamaCancelledError,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)


def test_ollama_client_chat_payload() -> None:
    """Tests that chat payload strictly includes stream: False, think: False, keep_alive, and passed options."""
    client = OllamaClient(
        endpoint="http://localhost:11434",
        model_name="gemma4:12b",
        keep_alive="10m",
        timeout_router=15.0,
        timeout_synthesizer=25.0,
    )

    captured_requests = []

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(
        {"message": {"role": "assistant", "content": "hello"}}
    ).encode("utf-8")

    mock_conn = MagicMock()
    mock_conn.getresponse.return_value = mock_resp

    def mock_request(method, path, body, headers):
        captured_requests.append((method, path, json.loads(body.decode("utf-8")), headers))

    mock_conn.request.side_effect = mock_request

    with patch("http.client.HTTPConnection", return_value=mock_conn):
        messages = [{"role": "user", "content": "What is Cerberus?"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        options = {"num_ctx": 8192, "temperature": 0.0, "seed": 42}

        resp = client.chat(messages=messages, tools=tools, options=options, timeout=5.0)

    assert resp["message"]["content"] == "hello"
    assert len(captured_requests) == 1
    method, path, payload, headers = captured_requests[0]

    assert method == "POST"
    assert path == "/api/chat"
    assert payload["model"] == "gemma4:12b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["keep_alive"] == "10m"
    assert payload["messages"] == messages
    assert payload["tools"] == tools
    assert payload["options"] == {"num_ctx": 8192, "temperature": 0.0, "seed": 42}


def test_ollama_client_generate_payload() -> None:
    """Tests /api/generate payload structure."""
    client = OllamaClient(model_name="gemma4:12b")

    captured_requests = []

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"response": "generated text"}).encode("utf-8")

    mock_conn = MagicMock()
    mock_conn.getresponse.return_value = mock_resp

    def mock_request(method, path, body, headers):
        captured_requests.append((method, path, json.loads(body.decode("utf-8")), headers))

    mock_conn.request.side_effect = mock_request

    with patch("http.client.HTTPConnection", return_value=mock_conn):
        client.generate(prompt="Test prompt", options={"num_predict": 100})

    assert len(captured_requests) == 1
    method, path, payload, headers = captured_requests[0]
    assert path == "/api/generate"
    assert payload["model"] == "gemma4:12b"
    assert payload["prompt"] == "Test prompt"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {"num_predict": 100}


def test_ollama_client_model_not_found() -> None:
    """Tests that HTTP 404 maps to OllamaModelNotFoundError."""
    client = OllamaClient(model_name="missing_model:latest")

    mock_resp = MagicMock()
    mock_resp.status = 404
    mock_resp.read.return_value = b'{"error":"model not found"}'

    mock_conn = MagicMock()
    mock_conn.getresponse.return_value = mock_resp

    with patch("http.client.HTTPConnection", return_value=mock_conn):
        with pytest.raises(OllamaModelNotFoundError, match="was not found by Ollama runtime"):
            client.chat(messages=[{"role": "user", "content": "hi"}])


def test_ollama_client_connection_error() -> None:
    """Tests that network error maps to OllamaConnectionError."""
    client = OllamaClient()

    mock_conn = MagicMock()
    mock_conn.request.side_effect = ConnectionRefusedError("Connection refused")

    with patch("http.client.HTTPConnection", return_value=mock_conn):
        with pytest.raises(OllamaConnectionError, match="Failed to connect to Ollama endpoint"):
            client.chat(messages=[{"role": "user", "content": "hi"}])


def test_ollama_client_timeout_error() -> None:
    """Tests that socket timeout maps to OllamaTimeoutError."""
    client = OllamaClient(timeout_router=2.0)

    mock_conn = MagicMock()
    mock_conn.request.side_effect = socket.timeout("Socket timed out")

    with patch("http.client.HTTPConnection", return_value=mock_conn):
        with pytest.raises(OllamaTimeoutError, match="timed out"):
            client.chat(messages=[{"role": "user", "content": "hi"}], timeout=2.0)


def test_ollama_client_cancellation_seam() -> None:
    """Tests that triggering cancel_event raises OllamaCancelledError and shuts down socket."""
    client = OllamaClient()

    cancel_ev = threading.Event()
    cancel_ev.set()

    mock_sock = MagicMock()
    mock_conn = MagicMock()
    mock_conn.sock = mock_sock

    # Make request block slightly to allow cancel loop to fire
    def slow_request(*args, **kwargs):
        import time
        time.sleep(0.1)

    mock_conn.request.side_effect = slow_request

    with patch("http.client.HTTPConnection", return_value=mock_conn):
        with pytest.raises(OllamaCancelledError, match="Request cancelled by caller"):
            client.chat(messages=[{"role": "user", "content": "hi"}], cancel_event=cancel_ev)

    mock_sock.shutdown.assert_called_once()
    mock_conn.close.assert_called()


def test_ollama_client_warmup() -> None:
    """Tests warmup calling chat on client and embed_text on embedder."""
    client = OllamaClient(model_name="gemma4:12b")
    mock_embedder = MagicMock()
    mock_embedder.model_name = "qwen3-embedding:0.6b"

    with patch.object(client, "chat", return_value={"message": {"content": "pong"}}) as mock_chat:
        client.warmup(embedder=mock_embedder)

    mock_chat.assert_called_once()
    mock_embedder.embed_text.assert_called_once_with("warmup")
