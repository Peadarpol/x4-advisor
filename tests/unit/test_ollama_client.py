"""Unit tests for OllamaClient payload construction, options, timeouts, and error handling."""

import json
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from x4_advisor.llm.client import (
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

    captured_payloads = []

    def mock_urlopen(req, timeout):
        captured_payloads.append((json.loads(req.data.decode("utf-8")), timeout))
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = json.dumps(
            {"message": {"role": "assistant", "content": "hello"}}
        ).encode("utf-8")
        resp.__enter__.return_value = resp
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        messages = [{"role": "user", "content": "What is Cerberus?"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        options = {"num_ctx": 8192, "temperature": 0.0, "seed": 42}

        resp = client.chat(messages=messages, tools=tools, options=options, timeout=5.0)

    assert resp["message"]["content"] == "hello"
    assert len(captured_payloads) == 1
    payload, timeout_used = captured_payloads[0]

    # Verify top-level parameters
    assert payload["model"] == "gemma4:12b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["keep_alive"] == "10m"
    assert payload["messages"] == messages
    assert payload["tools"] == tools
    assert payload["options"] == {"num_ctx": 8192, "temperature": 0.0, "seed": 42}
    assert timeout_used == 5.0


def test_ollama_client_generate_payload() -> None:
    """Tests /api/generate payload structure."""
    client = OllamaClient(model_name="gemma4:12b")

    captured_payloads = []

    def mock_urlopen(req, timeout):
        captured_payloads.append(json.loads(req.data.decode("utf-8")))
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = json.dumps({"response": "generated text"}).encode("utf-8")
        resp.__enter__.return_value = resp
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        client.generate(prompt="Test prompt", options={"num_predict": 100})

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert payload["model"] == "gemma4:12b"
    assert payload["prompt"] == "Test prompt"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {"num_predict": 100}


def test_ollama_client_model_not_found() -> None:
    """Tests that HTTP 404 maps to OllamaModelNotFoundError."""
    client = OllamaClient(model_name="missing_model:latest")

    http_err = urllib.error.HTTPError(
        url="http://localhost:11434/api/chat",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=None,
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(OllamaModelNotFoundError, match="was not found by Ollama runtime"):
            client.chat(messages=[{"role": "user", "content": "hi"}])


def test_ollama_client_connection_error() -> None:
    """Tests that network error maps to OllamaConnectionError."""
    client = OllamaClient()

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        with pytest.raises(OllamaConnectionError, match="Failed to connect to Ollama endpoint"):
            client.chat(messages=[{"role": "user", "content": "hi"}])


def test_ollama_client_timeout_error() -> None:
    """Tests that socket timeout maps to OllamaTimeoutError."""
    import socket
    client = OllamaClient(timeout_router=2.0)

    with patch("urllib.request.urlopen", side_effect=socket.timeout("Socket timed out")):
        with pytest.raises(OllamaTimeoutError, match="timed out"):
            client.chat(messages=[{"role": "user", "content": "hi"}], timeout=2.0)


def test_ollama_client_warmup() -> None:
    """Tests warmup calling chat on client and embed_text on embedder."""
    client = OllamaClient(model_name="gemma4:12b")
    mock_embedder = MagicMock()
    mock_embedder.model_name = "qwen3-embedding:0.6b"

    with patch.object(client, "chat", return_value={"message": {"content": "pong"}}) as mock_chat:
        client.warmup(embedder=mock_embedder)

    mock_chat.assert_called_once()
    mock_embedder.embed_text.assert_called_once_with("warmup")
