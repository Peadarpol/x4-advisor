"""LLM-based tool-calling router classifying queries into structured, vector, hybrid, or abstain routes."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from x4_advisor.llm.client import OllamaClient
from x4_advisor.retrieval.models import (
    AbstainReason,
    RouterResult,
    RouteType,
    ToolCall,
)
from x4_advisor.retrieval.structured_query import (
    ALLOWED_SHIP_METRICS,
    ALLOWED_WARE_METRICS,
)

logger = logging.getLogger(__name__)

# Valid production methods in base game (strictly base-game; no DLC methods)
VALID_PRODUCTION_METHODS = {"default", "teladi", "recycling", "xenon", "processing", "paranid"}

# Valid resource identifiers
VALID_RESOURCES = {"ore", "silicon", "ice", "hydrogen", "helium", "methane", "nividium"}

# Valid ship classes and purposes
VALID_SHIP_CLASSES = {"s", "m", "l", "xl", "ship_s", "ship_m", "ship_l", "ship_xl", "spacesuit"}
VALID_PURPOSES = {"fight", "trade", "mine", "build", "auxiliary", "salvage"}

# Router tool definitions in Ollama function calling format
ROUTER_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_structured_data",
            "description": (
                "Query the structured SQLite game database for exact factual statistics, "
                "ship specifications, ware prices, sector resource yields, multi-tier production chains, "
                "or category listings in base-game X4: Foundations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": [
                            "fact_lookup",
                            "ranking",
                            "sector_yield",
                            "production_chain",
                            "category_listing",
                        ],
                        "description": (
                            "Template type: 'fact_lookup' (T1 single entity stats), 'ranking' (T2 ship/ware comparison), "
                            "'sector_yield' (T2 resource yields by sector), 'production_chain' (T3 recipe tree & materials), "
                            "or 'category_listing' (T4 filter ships by faction/class/purpose or wares by category)."
                        ),
                    },
                    "entity_name": {
                        "type": "string",
                        "description": "Natural language name of ship, ware, sector, or faction (for fact_lookup or production_chain target).",
                    },
                    "metric": {
                        "type": "string",
                        "enum": [
                            "cargo_capacity",
                            "speed",
                            "hull",
                            "shields",
                            "weapon_slots",
                            "turret_slots",
                            "shield_slots",
                            "min_price",
                            "avg_price",
                            "max_price",
                            "volume",
                        ],
                        "description": "Statistical metric to rank by (e.g. 'cargo_capacity', 'speed', 'min_price').",
                    },
                    "ship_class": {
                        "type": "string",
                        "enum": ["s", "m", "l", "xl", "ship_s", "ship_m", "ship_l", "ship_xl", "spacesuit"],
                        "description": "Ship size class filter (e.g. 'l', 'm', 's', 'xl').",
                    },
                    "purpose": {
                        "type": "string",
                        "enum": ["fight", "trade", "mine", "build", "auxiliary", "salvage"],
                        "description": "Ship role/purpose filter.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Ware category filter (e.g. 'minerals', 'energy', 'food', 'weapons', 'hightech').",
                    },
                    "resource_id": {
                        "type": "string",
                        "enum": ["ore", "silicon", "ice", "hydrogen", "helium", "methane", "nividium"],
                        "description": "Resource type for sector yield ranking.",
                    },
                    "production_method": {
                        "type": "string",
                        "enum": ["default", "teladi", "recycling", "xenon", "processing", "paranid"],
                        "description": "Production recipe method for T3 production chain calculation.",
                    },
                    "sort_desc": {
                        "type": "boolean",
                        "description": "True for descending (highest/fastest/largest), False for ascending (lowest/slowest/smallest). Default is True.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows to return (default 5 for rankings, 50 for listings).",
                    },
                    "faction": {
                        "type": "string",
                        "description": "Faction name or ID for category listing filter (e.g. 'argon', 'teladi', 'paranid', 'antigone').",
                    },
                },
                "required": ["query_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the curated unstructured knowledge base for tactical advice, heuristics, "
                "gameplay mechanics, strategy explanations, loadout advice, or 'why' context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "Semantic search query for vector retrieval.",
                    }
                },
                "required": ["query_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "abstain",
            "description": (
                "Explicitly decline to answer if the question concerns out-of-scope DLC expansions "
                "(e.g. Cradle of Humanity / Terran, Split Vendetta, Tides of Avarice, Kingdom End, Timelines) "
                "or non-X4 content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "enum": ["out_of_scope_dlc", "out_of_scope_other"],
                        "description": "Reason for abstaining.",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Optional brief explanation of why the question is out of scope.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
]

ROUTER_SYSTEM_PROMPT = """You are the query routing assistant for X4 Advisor, an expert assistant for the base game of X4: Foundations.
Your task is to classify the user's question and select the appropriate tool(s) to retrieve evidence.

Tool Selection Rules:
1. Use 'query_structured_data' for exact statistics, ship specs, comparisons, production recipes, raw materials, sector resources, or category lists.
2. Use 'search_knowledge_base' for strategic advice, gameplay heuristics, tactics, or explanatory 'why' questions.
3. Call BOTH 'query_structured_data' and 'search_knowledge_base' for hybrid questions requiring both exact data and strategic explanations (e.g. "What does Hull Parts production require, and why is it strategically important?").
4. Call 'abstain' with reason 'out_of_scope_dlc' if the question asks about DLC factions (Terran, Segaris, Split, Boron, Riptide) or DLC-exclusive ships/content.
5. If you cannot determine any matching tool, do not guess parameters.
"""


class LLMRouter:
    """Classifies user questions into structured, vector, hybrid, or abstain routes using Ollama tool-calling."""

    def __init__(
        self,
        client: OllamaClient,
    ) -> None:
        self.client = client

    def route(self, question: str) -> RouterResult:
        """Determines the routing decision for a given user question."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        # Call Ollama with deterministic settings and 8192 context window
        raw_resp = self.client.chat(
            messages=messages,
            tools=ROUTER_TOOLS,
            options={"num_ctx": 8192, "temperature": 0.0, "seed": 42},
            timeout=self.client.timeout_router,
        )

        result, is_valid, error_msg = self._parse_and_validate(raw_resp)
        if is_valid:
            return result

        # Retry once with error feedback
        logger.info("Router tool call validation failed: '%s'. Retrying once...", error_msg)
        retry_messages = list(messages)
        retry_messages.append({"role": "assistant", "content": str(raw_resp.get("message", {}))})
        retry_messages.append(
            {
                "role": "user",
                "content": f"The previous tool call was invalid: {error_msg}. Please correct the tool call or call 'abstain'.",
            }
        )

        try:
            retry_resp = self.client.chat(
                messages=retry_messages,
                tools=ROUTER_TOOLS,
                options={"num_ctx": 8192, "temperature": 0.0, "seed": 42},
                timeout=self.client.timeout_router,
            )
            retry_result, retry_valid, retry_error = self._parse_and_validate(retry_resp)
            if retry_valid:
                return retry_result
            logger.warning("Router retry also failed: '%s'. Defaulting to ABSTAIN.", retry_error)
        except Exception as e:
            logger.warning("Router retry exception: %s. Defaulting to ABSTAIN.", e)

        return RouterResult(
            route_type=RouteType.ABSTAIN,
            tool_calls=[],
            abstain_reason=AbstainReason.MALFORMED_TOOL_CALL,
            raw_response=raw_resp,
        )

    def retry_with_feedback(
        self,
        question: str,
        previous_raw_response: Optional[Dict[str, Any]],
        error_msg: str,
    ) -> RouterResult:
        """Retries routing by feeding execution error (e.g. UnknownFilterValue) back to LLM."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        if previous_raw_response:
            messages.append({"role": "assistant", "content": str(previous_raw_response.get("message", {}))})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"The previous structured query failed with an unknown filter value: {error_msg}. "
                    f"Please correct the tool call parameters using only the valid values or call 'abstain'."
                ),
            }
        )

        try:
            retry_resp = self.client.chat(
                messages=messages,
                tools=ROUTER_TOOLS,
                options={"num_ctx": 8192, "temperature": 0.0, "seed": 42},
                timeout=self.client.timeout_router,
            )
            retry_result, retry_valid, retry_error = self._parse_and_validate(retry_resp)
            if retry_valid:
                return retry_result
            logger.warning("Router retry with feedback failed validation: '%s'", retry_error)
        except Exception as e:
            logger.warning("Router retry with feedback exception: %s", e)

        return RouterResult(
            route_type=RouteType.ABSTAIN,
            tool_calls=[],
            abstain_reason=AbstainReason.MALFORMED_TOOL_CALL,
            raw_response=previous_raw_response,
        )

    def _parse_and_validate(self, response: Dict[str, Any]) -> Tuple[RouterResult, bool, str]:
        """Parses tool calls from Ollama response and validates parameters."""
        message = response.get("message", {})
        tool_calls_raw = message.get("tool_calls", [])

        if not tool_calls_raw:
            # Check if model responded in content indicating refusal or inability
            content = message.get("content", "").strip()
            if "dlc" in content.lower() or "terran" in content.lower() or "split" in content.lower():
                return (
                    RouterResult(
                        route_type=RouteType.ABSTAIN,
                        abstain_reason=AbstainReason.OUT_OF_SCOPE_DLC,
                        raw_response=response,
                    ),
                    True,
                    "",
                )
            return (
                RouterResult(
                    route_type=RouteType.ABSTAIN,
                    abstain_reason=AbstainReason.MALFORMED_TOOL_CALL,
                    raw_response=response,
                ),
                False,
                "No tool calls emitted by the model.",
            )

        parsed_calls: List[ToolCall] = []
        has_structured = False
        has_vector = False
        has_abstain = False
        abstain_reason: Optional[AbstainReason] = None

        for tc in tool_calls_raw:
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", {})

            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except Exception:
                    return (
                        RouterResult(route_type=RouteType.ABSTAIN, raw_response=response),
                        False,
                        f"Malformed JSON arguments in tool call '{name}'.",
                    )

            if not isinstance(args, dict):
                return (
                    RouterResult(route_type=RouteType.ABSTAIN, raw_response=response),
                    False,
                    f"Arguments for tool '{name}' must be an object dictionary.",
                )

            # Validate tool by name
            if name == "query_structured_data":
                val_err = self._validate_structured_args(args)
                if val_err:
                    return (
                        RouterResult(route_type=RouteType.ABSTAIN, raw_response=response),
                        False,
                        val_err,
                    )
                has_structured = True
                parsed_calls.append(ToolCall(name=name, arguments=args))

            elif name == "search_knowledge_base":
                query_text = args.get("query_text", "")
                if not query_text or not str(query_text).strip():
                    return (
                        RouterResult(route_type=RouteType.ABSTAIN, raw_response=response),
                        False,
                        "search_knowledge_base requires non-empty 'query_text'.",
                    )
                has_vector = True
                parsed_calls.append(ToolCall(name=name, arguments=args))

            elif name == "abstain":
                reason_str = args.get("reason", "out_of_scope_other")
                if reason_str == "out_of_scope_dlc":
                    abstain_reason = AbstainReason.OUT_OF_SCOPE_DLC
                else:
                    abstain_reason = AbstainReason.OUT_OF_SCOPE_OTHER
                has_abstain = True
                parsed_calls.append(ToolCall(name=name, arguments=args))

            else:
                return (
                    RouterResult(route_type=RouteType.ABSTAIN, raw_response=response),
                    False,
                    f"Unknown tool call name '{name}'.",
                )

        if has_abstain:
            return (
                RouterResult(
                    route_type=RouteType.ABSTAIN,
                    tool_calls=parsed_calls,
                    abstain_reason=abstain_reason or AbstainReason.OUT_OF_SCOPE_OTHER,
                    raw_response=response,
                ),
                True,
                "",
            )

        if has_structured and has_vector:
            route_type = RouteType.BOTH
        elif has_structured:
            route_type = RouteType.STRUCTURED
        elif has_vector:
            route_type = RouteType.VECTOR
        else:
            route_type = RouteType.ABSTAIN

        return (
            RouterResult(
                route_type=route_type,
                tool_calls=parsed_calls,
                abstain_reason=None,
                raw_response=response,
            ),
            True,
            "",
        )

    def _validate_structured_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validates parameters and cross-parameter coherence for query_structured_data."""
        qtype = args.get("query_type")
        if not qtype:
            return "query_structured_data missing required 'query_type'."

        valid_qtypes = {"fact_lookup", "ranking", "sector_yield", "production_chain", "category_listing"}
        if qtype not in valid_qtypes:
            return f"Invalid query_type '{qtype}'. Allowed: {valid_qtypes}."

        metric = args.get("metric")
        if metric:
            lower_m = str(metric).strip().lower()
            if lower_m not in ALLOWED_SHIP_METRICS and lower_m not in ALLOWED_WARE_METRICS:
                return f"Invalid metric '{metric}'. Allowed ship metrics: {list(ALLOWED_SHIP_METRICS.keys())}, ware metrics: {list(ALLOWED_WARE_METRICS.keys())}."

            # Cross-parameter coherence validation
            if lower_m in ALLOWED_SHIP_METRICS and args.get("category"):
                return f"Incoherent parameters: ship metric '{metric}' cannot be combined with ware category '{args.get('category')}'."
            if lower_m in ALLOWED_WARE_METRICS and (args.get("ship_class") or args.get("purpose")):
                return f"Incoherent parameters: ware metric '{metric}' cannot be combined with ship_class or purpose."

        ship_class = args.get("ship_class")
        if ship_class and str(ship_class).strip().lower() not in VALID_SHIP_CLASSES:
            return f"Invalid ship_class '{ship_class}'. Allowed: {VALID_SHIP_CLASSES}."

        purpose = args.get("purpose")
        if purpose and str(purpose).strip().lower() not in VALID_PURPOSES:
            return f"Invalid purpose '{purpose}'. Allowed: {VALID_PURPOSES}."

        resource_id = args.get("resource_id")
        if resource_id and str(resource_id).strip().lower() not in VALID_RESOURCES:
            return f"Invalid resource_id '{resource_id}'. Allowed: {VALID_RESOURCES}."

        prod_method = args.get("production_method")
        if prod_method and str(prod_method).strip().lower() not in VALID_PRODUCTION_METHODS:
            return f"Invalid production_method '{prod_method}'. Base-game allowed: {VALID_PRODUCTION_METHODS}."

        return None
