"""Unit tests for X4 Advisor diagnostics and dataset staleness checking."""

import json
from pathlib import Path
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from x4_advisor.config import Config
from x4_advisor.diagnostics import DiagnosticReport, run_diagnostics
from x4_advisor.storage.schema import EXPECTED_SCHEMA_VERSION


def test_diagnostics_all_healthy(tmp_path: Path) -> None:
    """Tests diagnostics report when Ollama is up, models exist, and DB is valid."""
    db_file = tmp_path / "test_advisor.db"
    conn = sqlite3.connect(db_file)
    conn.execute(
        """
        CREATE TABLE dataset_metadata (
            id INTEGER PRIMARY KEY,
            game_version TEXT,
            build TEXT,
            extraction_timestamp TEXT,
            is_base_game_only INTEGER,
            schema_version TEXT
        );
        """
    )
    conn.execute(
        f"INSERT INTO dataset_metadata VALUES (1, '7.10', '538965', '2026-08-20T10:00:00Z', 1, '{EXPECTED_SCHEMA_VERSION}');"
    )
    for t in ["ships", "wares", "sectors", "sector_resources", "factions", "production_recipes", "knowledge_chunks"]:
        conn.execute(f"CREATE TABLE {t} (id TEXT PRIMARY KEY);")
        conn.execute(f"INSERT INTO {t} VALUES ('test_id');")
    conn.commit()
    conn.close()

    config = Config(validate=False)
    config.database_path_str = str(db_file)
    config.model_name = "gemma4:12b"
    config.embedding_model = "qwen3-embedding:0.6b"

    tags_response = json.dumps({"models": [{"name": "gemma4:12b"}, {"name": "qwen3-embedding:0.6b"}]})
    ps_response = json.dumps({
        "models": [{
            "name": "gemma4:12b",
            "size_vram": 8 * 1024**3,
            "size": 8 * 1024**3,
        }]
    })

    def mock_probe(endpoint, path, timeout_sec=3.0):
        if path == "/api/tags":
            return (200, tags_response, None)
        if path == "/api/ps":
            return (200, ps_response, None)
        return (404, "", None)

    with patch("x4_advisor.diagnostics._probe_http", side_effect=mock_probe):
        report = run_diagnostics(config)

    assert report.success is True
    rendered = report.render(use_color=False)
    assert "HEALTHY" in rendered
    assert "gemma4:12b" in rendered
    assert "All 6 core tables and vector chunks populated" in rendered
    assert "X4: Foundations v7.10" in rendered


def test_diagnostics_ollama_offline(tmp_path: Path) -> None:
    """Tests diagnostics report when Ollama daemon is unreachable."""
    db_file = tmp_path / "test_advisor.db"
    config = Config(validate=False)
    config.database_path_str = str(db_file)

    def mock_probe(endpoint, path, timeout_sec=3.0):
        return (0, "", ConnectionRefusedError("Connection refused"))

    with patch("x4_advisor.diagnostics._probe_http", side_effect=mock_probe):
        report = run_diagnostics(config)

    assert report.success is False
    rendered = report.render(use_color=False)
    assert "UNHEALTHY" in rendered
    assert "Unreachable" in rendered


def test_diagnostics_missing_database(tmp_path: Path) -> None:
    """Tests diagnostics when database file does not exist."""
    db_file = tmp_path / "non_existent.db"
    config = Config(validate=False)
    config.database_path_str = str(db_file)

    tags_response = json.dumps({"models": [{"name": "gemma4:12b"}, {"name": "qwen3-embedding:0.6b"}]})

    with patch("x4_advisor.diagnostics._probe_http", return_value=(200, tags_response, None)):
        report = run_diagnostics(config)

    assert report.success is False
    rendered = report.render(use_color=False)
    assert "Database file not found" in rendered


def test_diagnostics_stale_schema_version(tmp_path: Path) -> None:
    """Tests warning when schema_version differs from EXPECTED_SCHEMA_VERSION."""
    db_file = tmp_path / "test_advisor.db"
    conn = sqlite3.connect(db_file)
    conn.execute(
        """
        CREATE TABLE dataset_metadata (
            id INTEGER PRIMARY KEY,
            game_version TEXT,
            build TEXT,
            extraction_timestamp TEXT,
            is_base_game_only INTEGER,
            schema_version TEXT
        );
        """
    )
    conn.execute("INSERT INTO dataset_metadata VALUES (1, '7.10', '538965', '2026-08-20T10:00:00Z', 1, '0.9.0');")
    for t in ["ships", "wares", "sectors", "sector_resources", "factions", "production_recipes", "knowledge_chunks"]:
        conn.execute(f"CREATE TABLE {t} (id TEXT PRIMARY KEY);")
    conn.commit()
    conn.close()

    config = Config(validate=False)
    config.database_path_str = str(db_file)

    tags_response = json.dumps({"models": [{"name": "gemma4:12b"}, {"name": "qwen3-embedding:0.6b"}]})
    with patch("x4_advisor.diagnostics._probe_http", return_value=(200, tags_response, None)):
        report = run_diagnostics(config)

    rendered = report.render(use_color=False)
    assert "Database schema version '0.9.0' differs from expected '1.1.0'" in rendered
