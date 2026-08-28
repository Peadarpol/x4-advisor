"""Unit tests for OllamaClient payload construction, options, timeouts, error handling, and cancellation."""

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


def _make_mock_socket(response_bytes: bytes) -> MagicMock:
    """Creates a mock socket that returns specified response bytes on recv()."""
    mock_sock = MagicMock()
    chunks = [response_bytes[i : i + 4096] for i in range(0, len(response_bytes), 4096)]
    chunks.append(b"")  # EOF
    mock_sock.recv.side_effect = chunks
    return mock_sock


def test_ollama_client_chat_payload() -> None:
    """Tests that chat payload strictly includes stream: False, think: False, keep_alive, and passed options."""
    client = OllamaClient(
        endpoint="http://localhost:11434",
        model_name="gemma4:12b",
        keep_alive="10m",
        timeout_router=15.0,
        timeout_synthesizer=25.0,
    )

    resp_body = json.dumps({"message": {"role": "assistant", "content": "hello"}}).encode("utf-8")
    http_resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n" + resp_body
    mock_sock = _make_mock_socket(http_resp)

    sent_data = bytearray()
    mock_sock.sendall.side_effect = lambda data: sent_data.extend(data)

    with patch("socket.create_connection", return_value=mock_sock):
        messages = [{"role": "user", "content": "What is Cerberus?"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        options = {"num_ctx": 8192, "temperature": 0.0, "seed": 42}

        resp = client.chat(messages=messages, tools=tools, options=options, timeout=5.0)

    assert resp["message"]["content"] == "hello"

    # Extract JSON body from sent HTTP request
    header_end = sent_data.find(b"\r\n\r\n")
    assert header_end != -1
    req_body = json.loads(sent_data[header_end + 4 :].decode("utf-8"))

    assert req_body["model"] == "gemma4:12b"
    assert req_body["stream"] is False
    assert req_body["think"] is False
    assert req_body["keep_alive"] == "10m"
    assert req_body["messages"] == messages
    assert req_body["tools"] == tools
    assert req_body["options"] == {"num_ctx": 8192, "temperature": 0.0, "seed": 42}


def test_ollama_client_generate_payload() -> None:
    """Tests /api/generate payload structure."""
    client = OllamaClient(model_name="gemma4:12b")

    resp_body = json.dumps({"response": "generated text"}).encode("utf-8")
    http_resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n" + resp_body
    mock_sock = _make_mock_socket(http_resp)

    sent_data = bytearray()
    mock_sock.sendall.side_effect = lambda data: sent_data.extend(data)

    with patch("socket.create_connection", return_value=mock_sock):
        client.generate(prompt="Test prompt", options={"num_predict": 100})

    header_end = sent_data.find(b"\r\n\r\n")
    assert header_end != -1
    req_body = json.loads(sent_data[header_end + 4 :].decode("utf-8"))

    assert req_body["model"] == "gemma4:12b"
    assert req_body["prompt"] == "Test prompt"
    assert req_body["stream"] is False
    assert req_body["think"] is False
    assert req_body["options"] == {"num_predict": 100}


def test_ollama_client_model_not_found() -> None:
    """Tests that HTTP 404 maps to OllamaModelNotFoundError."""
    client = OllamaClient(model_name="missing_model:latest")

    http_resp = b"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\r\n{\"error\":\"model not found\"}"
    mock_sock = _make_mock_socket(http_resp)

    with patch("socket.create_connection", return_value=mock_sock):
        with pytest.raises(OllamaModelNotFoundError, match="was not found by Ollama runtime"):
            client.chat(messages=[{"role": "user", "content": "hi"}])


def test_ollama_client_connection_error() -> None:
    """Tests that network error maps to OllamaConnectionError."""
    client = OllamaClient()

    with patch("socket.create_connection", side_effect=ConnectionRefusedError("Connection refused")):
        with pytest.raises(OllamaConnectionError, match="Failed to connect to Ollama endpoint"):
            client.chat(messages=[{"role": "user", "content": "hi"}])


def test_ollama_client_timeout_error() -> None:
    """Tests that connection timeout maps to OllamaTimeoutError."""
    client = OllamaClient(timeout_router=2.0)

    with patch("socket.create_connection", side_effect=socket.timeout("Socket timed out")):
        with pytest.raises(OllamaTimeoutError, match="timed out"):
            client.chat(messages=[{"role": "user", "content": "hi"}], timeout=2.0)


def test_ollama_client_cancellation_seam() -> None:
    """Tests that triggering cancel_event raises OllamaCancelledError and shuts down socket."""
    client = OllamaClient()

    cancel_ev = threading.Event()
    cancel_ev.set()

    mock_sock = MagicMock()

    with patch("socket.create_connection", return_value=mock_sock):
        with pytest.raises(OllamaCancelledError, match="Request cancelled by caller"):
            client.chat(messages=[{"role": "user", "content": "hi"}], cancel_event=cancel_ev)

    mock_sock.shutdown.assert_called_once()
    mock_sock.close.assert_called()


def test_ollama_client_warmup() -> None:
    """Tests warmup calling chat on client and embed_text on embedder."""
    client = OllamaClient(model_name="gemma4:12b")
    mock_embedder = MagicMock()
    mock_embedder.model_name = "qwen3-embedding:0.6b"

    with patch.object(client, "chat", return_value={"message": {"content": "pong"}}) as mock_chat:
        client.warmup(embedder=mock_embedder)

    mock_chat.assert_called_once()
    mock_embedder.embed_text.assert_called_once_with("warmup")
