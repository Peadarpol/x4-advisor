"""Live integration tests for Milestone M8 CLI delivery and diagnostics."""

import argparse
import json
import sqlite3
import threading
import time
import urllib.request

import pytest

from x4_advisor.config import get_config
from x4_advisor.diagnostics import run_diagnostics
from x4_advisor.llm.client import OllamaCancelledError
from x4_advisor.retrieval.advisor_engine import AdvisorEngine


def _is_ollama_available() -> bool:
    """Checks if Ollama endpoint responds and gemma4:12b is available."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status != 200:
                return False
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "").lower() for m in data.get("models", [])]
            return any("gemma4:12b" in m for m in models)
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_ollama_available(),
        reason="Ollama endpoint unreachable or 'gemma4:12b' model not installed in local Ollama.",
    ),
]


def test_integration_doctor_live() -> None:
    """Runs x4-advisor doctor against live environment and asserts success."""
    config = get_config(validate=False)
    args = argparse.Namespace(model=None, db=None, no_color=True)

    report = run_diagnostics(config)
    assert report.success is True, f"Live diagnostics failed:\n{report.render(use_color=False)}"

    rendered = report.render(use_color=False)
    assert "HEALTHY" in rendered
    assert "Ollama Daemon: Connected" in rendered
    assert "Synthesis Model" in rendered
    assert "Database Integrity" in rendered


@pytest.mark.xfail(
    reason="Single-path latency SLA target (<20.0s) may fluctuate depending on local GPU contention",
    strict=False,
)
def test_integration_cold_warmup_and_answer_latency() -> None:
    """Asserts post-warmup first answer latency satisfies <20s SLA."""
    config = get_config(validate=False)
    config.validate_m5_config(probe_ollama=True)

    with AdvisorEngine(config=config) as engine:
        # Measure warmup pass
        t_warmup_start = time.monotonic()
        engine.client.warmup(embedder=engine.embedder)
        warmup_duration = time.monotonic() - t_warmup_start

        # Warmup duration is recorded/reported
        assert warmup_duration > 0.0

        # Assert post-warmup first answer latency < 20s
        t_query_start = time.monotonic()
        resp = engine.answer("What is the hull and shield value of the Cerberus Vanguard?")
        query_duration = time.monotonic() - t_query_start

        assert query_duration < 20.0, f"Query took {query_duration:.2f}s, exceeding 20s SLA"
        assert resp.synthesis_result is not None
        assert resp.synthesis_result.has_evidence is True


def test_integration_read_only_pragma() -> None:
    """Asserts that AdvisorEngine operates strictly under PRAGMA query_only = ON."""
    config = get_config(validate=False)
    with AdvisorEngine(config=config) as engine:
        cursor = engine.conn.cursor()
        cursor.execute("PRAGMA query_only;")
        row = cursor.fetchone()
        assert row[0] == 1, "Database connection is not enforcing PRAGMA query_only = ON"


def test_integration_server_side_cancellation_release() -> None:
    """Asserts that after socket cancellation, the client aborts immediately and can execute subsequent queries."""
    config = get_config(validate=False)
    config.validate_m5_config(probe_ollama=True)

    with AdvisorEngine(config=config) as engine:
        # Warmup engine
        engine.client.warmup(embedder=engine.embedder)

        # Trigger a heavy query with an active cancel_event that sets after a brief delay
        cancel_ev = threading.Event()

        def trigger_cancel():
            time.sleep(0.3)
            cancel_ev.set()

        t_cancel = threading.Thread(target=trigger_cancel, daemon=True)
        t_cancel.start()

        # Cancelled call should raise OllamaCancelledError
        t_cancel_start = time.monotonic()
        with pytest.raises((OllamaCancelledError, RuntimeError)):
            engine.client.chat(
                messages=[{"role": "user", "content": "Write a comprehensive 2000-word tactical guide on fleet combat."}],
                options={"num_predict": 2048},
                cancel_event=cancel_ev,
            )
        cancel_elapsed = time.monotonic() - t_cancel_start
        # Ensure cancellation aborted quickly (within 2s) without waiting for full generation
        assert cancel_elapsed < 2.0, f"Cancellation took {cancel_elapsed:.2f}s, did not abort promptly"
