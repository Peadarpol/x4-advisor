"""Unit tests for Milestone M5 configuration, .env loader, and Ollama probe validation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from x4_advisor.config import Config, ConfigError, _load_dotenv_if_present, get_config


def test_load_dotenv_basic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests loading environment variables from a .env file without overriding existing env vars."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Test env file\n"
        "TEST_NEW_KEY=hello_world\n"
        "TEST_EXISTING_KEY=from_dotenv\n"
        "INVALID_LINE_WITHOUT_EQUALS\n"
        "TEST_QUOTED_VAL=\"quoted_string\"\n"
    )

    monkeypatch.setenv("TEST_EXISTING_KEY", "real_environment_value")
    monkeypatch.delenv("TEST_NEW_KEY", raising=False)
    monkeypatch.delenv("TEST_QUOTED_VAL", raising=False)

    _load_dotenv_if_present(env_path=env_file)

    assert os.getenv("TEST_NEW_KEY") == "hello_world"
    assert os.getenv("TEST_EXISTING_KEY") == "real_environment_value"
    assert os.getenv("TEST_QUOTED_VAL") == "quoted_string"


import os


def test_vector_relevance_threshold_default_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests vector_relevance_threshold_is_default property behavior."""
    monkeypatch.delenv("VECTOR_RELEVANCE_THRESHOLD", raising=False)
    config = Config(validate=False)
    assert config.vector_relevance_threshold == 0.40
    assert config.vector_relevance_threshold_is_default is True

    monkeypatch.setenv("VECTOR_RELEVANCE_THRESHOLD", "0.65")
    config2 = Config(validate=False)
    assert config2.vector_relevance_threshold == 0.65
    assert config2.vector_relevance_threshold_is_default is False


def test_validate_m5_config_missing_vars() -> None:
    """Tests that validate_m5_config raises ConfigError if MODEL_NAME or OLLAMA_ENDPOINT is empty."""
    config = Config(validate=False)
    config.model_name = None
    with pytest.raises(ConfigError, match="MODEL_NAME environment variable is not set"):
        config.validate_m5_config(probe_ollama=False)

    config.model_name = "gemma4:12b"
    config.ollama_endpoint = ""
    with pytest.raises(ConfigError, match="OLLAMA_ENDPOINT environment variable is not set"):
        config.validate_m5_config(probe_ollama=False)


def test_validate_m5_config_probe_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests successful reachability probe and exact tag resolution."""
    config = Config(validate=False)
    config.model_name = "gemma4:12b"
    config.ollama_endpoint = "http://localhost:11434"

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(
        {
            "models": [
                {"name": "gemma4:12b-instruct-q4_K_M"},
                {"name": "qwen3-embedding:0.6b"},
            ]
        }
    ).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        config.validate_m5_config(probe_ollama=True)
        # Should resolve prefix match to exact installed tag
        assert config.model_name == "gemma4:12b-instruct-q4_K_M"


def test_validate_m5_config_probe_unreachable() -> None:
    """Tests that reachability probe raises clear ConfigError on network error."""
    config = Config(validate=False)
    config.model_name = "gemma4:12b"
    config.ollama_endpoint = "http://localhost:11434"

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        with pytest.raises(ConfigError, match="Cannot reach Ollama endpoint"):
            config.validate_m5_config(probe_ollama=True)


def test_validate_m5_config_probe_model_not_installed() -> None:
    """Tests that probe lists installed models when requested model is not found."""
    config = Config(validate=False)
    config.model_name = "nonexistent_model:latest"
    config.ollama_endpoint = "http://localhost:11434"

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(
        {"models": [{"name": "qwen3:14b"}, {"name": "granite4.1:8b"}]}
    ).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(ConfigError, match="is not installed in Ollama.*Installed models: \\['qwen3:14b', 'granite4.1:8b'\\]"):
            config.validate_m5_config(probe_ollama=True)
