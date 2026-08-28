"""Unit tests for X4 Advisor CLI argument parsing, output formatting, disambiguation, and REPL commands."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from x4_advisor.cli import (
    ask_command,
    build_parser,
    doctor_command,
    format_advisor_output,
    handle_disambiguation,
    interactive_command,
)
from x4_advisor.diagnostics import DiagnosticReport
from x4_advisor.llm.client import OllamaCancelledError
from x4_advisor.retrieval.models import (
    AbstainReason,
    AdvisorResponse,
    ResolvedEntity,
    RetrievedChunk,
    RouterResult,
    RouteType,
    SynthesisResult,
    VectorSearchResult,
)


def test_cli_parser_subcommands() -> None:
    """Tests that argparse parser correctly handles all subcommands and shared parent options."""
    parser = build_parser()

    # ask subcommand with flags before and after
    args_ask = parser.parse_args(["ask", "--model", "qwen3:14b", "--explain", "--no-warmup", "What", "is", "Cerberus?"])
    assert args_ask.command == "ask"
    assert args_ask.model == "qwen3:14b"
    assert args_ask.explain is True
    assert args_ask.no_warmup is True
    assert args_ask.question == ["What", "is", "Cerberus?"]

    # doctor subcommand
    args_doc = parser.parse_args(["doctor", "--db", "custom.db", "--no-color"])
    assert args_doc.command == "doctor"
    assert args_doc.db == "custom.db"
    assert args_doc.no_color is True

    # interactive subcommand
    args_inter = parser.parse_args(["interactive", "--skip-probe"])
    assert args_inter.command == "interactive"
    assert args_inter.skip_probe is True

    # bare invocation
    args_bare = parser.parse_args([])
    assert args_bare.command is None


def test_format_advisor_output_provenance() -> None:
    """Tests that output formatting renders chunk metadata, scores, and telemetry."""
    chunk = RetrievedChunk(
        chunk_id="chunk_mining_01",
        manifest_id="guide_mining",
        heading_hierarchy="Mining Operations > Solid Resource Extractors",
        content="Mining ships extract ore efficiently.",
        similarity_score=0.825,
        distance=0.175,
        source_attribution="Steam Community Guide",
        topic="mining",
        game_version_scope="7.10+",
    )
    synth = SynthesisResult(
        answer_text="Solid miners extract mineral resources from asteroid fields.",
        has_evidence=True,
        abstain_reason=None,
        evidence_chunk_ids=["chunk_mining_01"],
        notes=["Requires resource probe"],
    )
    vec = VectorSearchResult(
        query_text="mining",
        chunks=[chunk],
        status="success",
        total_candidates=1,
        threshold_used=0.50,
    )
    route = RouterResult(route_type=RouteType.VECTOR)

    resp = AdvisorResponse(
        question="How do solid miners work?",
        route_result=route,
        vector_result=vec,
        synthesis_result=synth,
    )
    # attach timing attributes for explain
    resp.execution_durations = {"router": 0.05, "vector": 0.01, "synthesis": 0.5}
    resp.total_duration_seconds = 0.56

    output = format_advisor_output(resp, explain=True, use_color=False)
    assert "Solid miners extract mineral resources" in output
    assert "Mining Operations > Solid Resource Extractors" in output
    assert "Steam Community Guide (similarity: 0.825)" in output
    assert "[Operational Notes]:" in output
    assert "Requires resource probe" in output
    assert "--- Pipeline Telemetry & Execution Trace ---" in output
    assert "Route Type: VECTOR" in output
    assert "Total Latency: 0.56s" in output


def test_format_advisor_output_abstention_badge() -> None:
    """Tests that abstentions render explicit uppercase badges."""
    route = RouterResult(route_type=RouteType.ABSTAIN, abstain_reason=AbstainReason.OUT_OF_SCOPE_DLC)
    synth = SynthesisResult(
        answer_text="Terran ships are part of the Cradle of Humanity DLC and out of scope.",
        has_evidence=False,
        abstain_reason=AbstainReason.OUT_OF_SCOPE_DLC,
    )
    resp = AdvisorResponse(
        question="What is the Tokyo carrier?",
        route_result=route,
        synthesis_result=synth,
    )

    output = format_advisor_output(resp, use_color=False)
    assert "[ABSTAIN: OUT_OF_SCOPE_DLC]" in output
    assert "Terran ships are part of the Cradle of Humanity DLC" in output


def test_handle_disambiguation_numeric_selection() -> None:
    """Tests that valid numeric selection directly calls engine.answer with pending_route and canonical ID."""
    mock_engine = MagicMock()
    mock_engine.answer.return_value = AdvisorResponse(
        question="What is the cargo of Magnetar?",
        synthesis_result=SynthesisResult(answer_text="Magnetar Vanguard has 8200 m3 cargo capacity.", has_evidence=True),
    )

    candidate1 = ResolvedEntity(id="ship_arg_m_miner_solid_01_a_macro", name="Magnetar Vanguard", entity_type="ship")
    candidate2 = ResolvedEntity(id="ship_arg_m_miner_solid_01_b_macro", name="Magnetar Sentinel", entity_type="ship")

    ambig_resp = AdvisorResponse(
        question="What is the cargo of Magnetar?",
        ambiguous_candidates=[candidate1, candidate2],
        pending_route={"operation": "lookup_entity", "metric": "cargo_capacity"},
    )

    with patch("builtins.input", return_value="1"):
        res = handle_disambiguation(
            engine=mock_engine,
            original_question="What is the cargo of Magnetar?",
            response=ambig_resp,
            use_color=False,
        )

    assert res is not None
    assert res.synthesis_result.answer_text == "Magnetar Vanguard has 8200 m3 cargo capacity."
    mock_engine.answer.assert_called_once_with(
        question="What is the cargo of Magnetar?",
        pending_route={"operation": "lookup_entity", "metric": "cargo_capacity"},
        resolved_entity_id="ship_arg_m_miner_solid_01_a_macro",
    )


def test_handle_disambiguation_cancel() -> None:
    """Tests that entering 'c' or pressing Ctrl+C cancels entity selection and returns None."""
    mock_engine = MagicMock()
    candidate = ResolvedEntity(id="ship_1", name="Ship 1", entity_type="ship")
    ambig_resp = AdvisorResponse(
        question="Q",
        ambiguous_candidates=[candidate],
        pending_route={"operation": "lookup"},
    )

    # Cancel via 'c'
    with patch("builtins.input", return_value="c"):
        res = handle_disambiguation(engine=mock_engine, original_question="Q", response=ambig_resp)
        assert res is None

    # Cancel via KeyboardInterrupt
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        res = handle_disambiguation(engine=mock_engine, original_question="Q", response=ambig_resp)
        assert res is None

    mock_engine.answer.assert_not_called()


def test_ask_command_success() -> None:
    """Tests ask_command execution and returncode 0."""
    args = argparse.Namespace(
        command="ask",
        question=["What", "is", "Cerberus?"],
        model=None,
        db=None,
        skip_probe=True,
        no_warmup=True,
        explain=False,
        no_color=True,
    )

    mock_resp = AdvisorResponse(
        question="What is Cerberus?",
        synthesis_result=SynthesisResult(answer_text="Cerberus is a frigate.", has_evidence=True),
    )

    with patch("x4_advisor.cli.get_config") as mock_cfg, \
         patch("x4_advisor.cli.AdvisorEngine") as mock_engine_cls:

        mock_config = MagicMock()
        mock_cfg.return_value = mock_config
        mock_engine = MagicMock()
        mock_engine.__enter__.return_value = mock_engine
        mock_engine.answer.return_value = mock_resp
        mock_engine_cls.return_value = mock_engine

        ret = ask_command(args)
        assert ret == 0
        mock_engine.answer.assert_called_once_with("What is Cerberus?")


def test_ask_command_cancelled() -> None:
    """Tests ask_command returns code 130 when cancelled by user."""
    args = argparse.Namespace(
        command="ask",
        question=["What", "is", "Cerberus?"],
        model=None,
        db=None,
        skip_probe=True,
        no_warmup=True,
        explain=False,
        no_color=True,
    )

    with patch("x4_advisor.cli.get_config"), \
         patch("x4_advisor.cli.AdvisorEngine") as mock_engine_cls:

        mock_engine = MagicMock()
        mock_engine.__enter__.return_value = mock_engine
        mock_engine.answer.side_effect = OllamaCancelledError("Cancelled")
        mock_engine_cls.return_value = mock_engine

        ret = ask_command(args)
        assert ret == 130


def test_doctor_command() -> None:
    """Tests doctor_command returns 0 on healthy and 1 on failure."""
    args = argparse.Namespace(model=None, db=None, no_color=True)

    with patch("x4_advisor.cli.run_diagnostics") as mock_diag:
        mock_report = DiagnosticReport(timestamp="2026-08-28 00:00:00 UTC", success=True)
        mock_diag.return_value = mock_report
        assert doctor_command(args) == 0

        mock_report.success = False
        assert doctor_command(args) == 1


def test_interactive_command_repl_controls() -> None:
    """Tests interactive REPL handling /help, /doctor, /explain, and /exit."""
    args = argparse.Namespace(
        command="interactive",
        model=None,
        db=None,
        skip_probe=True,
        no_warmup=True,
        explain=False,
        no_color=True,
    )

    inputs = ["/help", "/explain", "/doctor", "/exit"]

    with patch("x4_advisor.cli.get_config"), \
         patch("x4_advisor.cli.AdvisorEngine") as mock_engine_cls, \
         patch("builtins.input", side_effect=inputs), \
         patch("x4_advisor.cli.run_diagnostics") as mock_diag:

        mock_report = DiagnosticReport(timestamp="2026-08-28", success=True)
        mock_diag.return_value = mock_report

        mock_engine = MagicMock()
        mock_engine.__enter__.return_value = mock_engine
        mock_engine_cls.return_value = mock_engine

        ret = interactive_command(args)
        assert ret == 0
        mock_diag.assert_called_once()
        mock_engine.answer.assert_not_called()
