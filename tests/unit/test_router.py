"""Unit tests for LLMRouter grammar-constrained JSON schema classification, parameter validation, and retry logic."""

import json
from unittest.mock import MagicMock

import pytest

from x4_advisor.retrieval.models import AbstainReason, RouteType
from x4_advisor.retrieval.router import LLMRouter


def test_router_structured_fact_lookup() -> None:
    """Tests routing a fact lookup question via structured JSON schema."""
    mock_client = MagicMock()
    mock_client.timeout_router = 15.0
    mock_payload = {
        "route_type": "STRUCTURED",
        "structured": {
            "operation": "lookup_entity",
            "query_name": "Cerberus Vanguard",
            "metric": "none",
            "ship_class": "none",
            "purpose": "none",
            "category": "none",
            "faction": "",
            "resource_id": "none",
            "production_method": "none",
        },
        "vector": {"query_text": ""},
        "abstain_reason": "NONE",
    }
    mock_client.chat.return_value = {
        "message": {
            "role": "assistant",
            "content": json.dumps(mock_payload),
        }
    }

    router = LLMRouter(client=mock_client)
    res = router.route("What is the cargo capacity of the Cerberus Vanguard?")

    assert res.route_type == RouteType.STRUCTURED
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "query_structured_data"
    assert res.tool_calls[0].arguments["query_type"] == "fact_lookup"
    assert res.tool_calls[0].arguments["entity_name"] == "Cerberus Vanguard"
    assert res.abstain_reason is None


def test_router_structured_ranking_with_sort_desc_false() -> None:
    """Tests routing ranking query with sort_desc=False and purpose."""
    mock_client = MagicMock()
    mock_payload = {
        "route_type": "STRUCTURED",
        "structured": {
            "operation": "compare_entities",
            "query_name": "none",
            "metric": "speed",
            "ship_class": "s",
            "purpose": "fight",
            "category": "none",
            "faction": "",
            "resource_id": "none",
            "production_method": "none",
            "sort_desc": False,
            "limit": 5,
        },
        "vector": {"query_text": ""},
        "abstain_reason": "NONE",
    }
    mock_client.chat.return_value = {
        "message": {
            "role": "assistant",
            "content": json.dumps(mock_payload),
        }
    }

    router = LLMRouter(client=mock_client)
    res = router.route("Which S-class fighter is the slowest?")

    assert res.route_type == RouteType.STRUCTURED
    assert res.tool_calls[0].arguments["sort_desc"] is False
    assert res.tool_calls[0].arguments["purpose"] == "fight"
    assert res.tool_calls[0].arguments["ship_class"] == "s"
    assert res.tool_calls[0].arguments["metric"] == "speed"


def test_router_vector_search() -> None:
    """Tests routing strategy/heuristic query to search_knowledge_base."""
    mock_client = MagicMock()
    mock_payload = {
        "route_type": "VECTOR",
        "structured": {"operation": "none"},
        "vector": {"query_text": "effective early game trading routes and tactics"},
        "abstain_reason": "NONE",
    }
    mock_client.chat.return_value = {
        "message": {
            "role": "assistant",
            "content": json.dumps(mock_payload),
        }
    }

    router = LLMRouter(client=mock_client)
    res = router.route("What are good early-game trading strategies?")

    assert res.route_type == RouteType.VECTOR
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "search_knowledge_base"
    assert "early game" in res.tool_calls[0].arguments["query_text"]


def test_router_hybrid_both() -> None:
    """Tests routing a hybrid question to both structured and vector tools."""
    mock_client = MagicMock()
    mock_payload = {
        "route_type": "BOTH",
        "structured": {
            "operation": "production_chain",
            "query_name": "Hull Parts",
            "metric": "none",
            "ship_class": "none",
            "purpose": "none",
            "category": "none",
            "faction": "",
            "resource_id": "none",
            "production_method": "default",
        },
        "vector": {"query_text": "Hull Parts strategic importance universe economy demand"},
        "abstain_reason": "NONE",
    }
    mock_client.chat.return_value = {
        "message": {
            "role": "assistant",
            "content": json.dumps(mock_payload),
        }
    }

    router = LLMRouter(client=mock_client)
    res = router.route("What does Hull Parts production require, and why is it strategically important?")

    assert res.route_type == RouteType.BOTH
    assert len(res.tool_calls) == 2
    names = {tc.name for tc in res.tool_calls}
    assert "query_structured_data" in names
    assert "search_knowledge_base" in names


def test_router_abstain_dlc() -> None:
    """Tests explicit abstention on DLC content."""
    mock_client = MagicMock()
    mock_payload = {
        "route_type": "ABSTAIN",
        "structured": {"operation": "none"},
        "vector": {"query_text": ""},
        "abstain_reason": "OUT_OF_SCOPE_DLC",
    }
    mock_client.chat.return_value = {
        "message": {
            "role": "assistant",
            "content": json.dumps(mock_payload),
        }
    }

    router = LLMRouter(client=mock_client)
    res = router.route("Where can I buy the Syn battleship from the Terran Protectorate?")

    assert res.route_type == RouteType.ABSTAIN
    assert res.abstain_reason == AbstainReason.OUT_OF_SCOPE_DLC


def test_router_parameter_coherence_retry_and_recovery() -> None:
    """Tests that incoherent parameters (e.g. ship metric + ware category) trigger a retry with feedback."""
    mock_client = MagicMock()
    bad_payload = {
        "route_type": "STRUCTURED",
        "structured": {
            "operation": "compare_entities",
            "metric": "cargo_capacity",
            "category": "minerals",
        },
        "vector": {"query_text": ""},
        "abstain_reason": "NONE",
    }
    good_payload = {
        "route_type": "STRUCTURED",
        "structured": {
            "operation": "compare_entities",
            "metric": "volume",
            "category": "minerals",
        },
        "vector": {"query_text": ""},
        "abstain_reason": "NONE",
    }

    mock_client.chat.side_effect = [
        {"message": {"role": "assistant", "content": json.dumps(bad_payload)}},
        {"message": {"role": "assistant", "content": json.dumps(good_payload)}},
    ]

    router = LLMRouter(client=mock_client)
    res = router.route("Which minerals take the most cargo space?")

    assert mock_client.chat.call_count == 2
    assert res.route_type == RouteType.STRUCTURED
    assert res.tool_calls[0].arguments["metric"] == "volume"


def test_router_malformed_tool_call_retry_fallback() -> None:
    """Tests that unrecoverable malformed content falls back to ABSTAIN."""
    mock_client = MagicMock()
    bad_resp = {"message": {"role": "assistant", "content": "not-valid-json{"}}

    mock_client.chat.side_effect = [bad_resp, bad_resp]

    router = LLMRouter(client=mock_client)
    res = router.route("Invalid query")

    assert res.route_type == RouteType.ABSTAIN
    assert res.abstain_reason == AbstainReason.MALFORMED_TOOL_CALL
