"""LLM-based grammar-constrained JSON schema router classifying queries into structured, vector, hybrid, or abstain routes."""

import json
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
from x4_advisor.retrieval.vocabularies import (
    DynamicVocabularies,
    build_router_json_schema,
)

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are the query routing assistant for X4 Advisor, an expert assistant for base-game X4: Foundations.
Analyze the user's question and select the exact route type and retrieval parameters.

Routing Guidelines:
1. STRUCTURED: Exact stats, ship specs, ware prices, sector yields, multi-tier production chains, or category listings.
   - For single entity lookup (e.g. "What is the cargo capacity of Cerberus Vanguard?"): operation='lookup_entity', query_name='Cerberus Vanguard'.
   - For sector stats (e.g. "sunlight rating of Grand Exchange I"): operation='lookup_entity', query_name='Grand Exchange I'.
   - For comparisons/rankings (e.g. "fastest S-class scouts", "top cargo"): operation='compare_entities', metric='speed', ship_class='ship_s', sort_desc=true.
   - For production recipes/materials (e.g. "inputs for Claytronics"): operation='production_chain', query_name='Claytronics', production_method='default'.
   - For category listings (e.g. "all Argon ships", "Medium ships", "refined wares"): operation='list_category'. DO NOT put filter keywords into query_name! Instead use the dedicated fields:
     * When filtering by faction (e.g. Argon, Teladi): faction='argon' (query_name='none').
     * When filtering by ship size (e.g. Medium, S, L, XL): ship_class='ship_m' (query_name='none').
     * When filtering by role/purpose (e.g. combat, trade, mine): purpose='fight' (query_name='none').
     * When filtering by ware category (e.g. refined, hightech): category='refined' (query_name='none').
   - For sector resource yields: operation='sector_yield', resource_id='ore'.
2. VECTOR: Strategic guidance, gameplay mechanics, tactical advice, pilot automation, or explanatory 'why' questions.
   - Set vector.query_text to a concise semantic search query.
3. BOTH (Hybrid): Questions combining exact database specs/recipes WITH strategic advice or economic context.
4. ABSTAIN: Non-X4 questions (NO_EVIDENCE / OUT_OF_SCOPE_OTHER) or questions concerning DLC expansions (Terran, Split, Boron, Avarice, Timelines -> OUT_OF_SCOPE_DLC).

Set unused fields to 'none' or '' as specified by the schema.
"""


class LLMRouter:
    """Classifies user questions into structured, vector, hybrid, or abstain routes using grammar-constrained JSON schema decoding."""

    def __init__(
        self,
        client: OllamaClient,
        vocab: Optional[DynamicVocabularies] = None,
    ) -> None:
        self.client = client
        self.vocab = vocab or DynamicVocabularies()
        self.schema = build_router_json_schema(self.vocab)

    def route(self, question: str) -> RouterResult:
        """Determines the routing decision for a given user question using grammar-constrained decoding."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        # Call Ollama with deterministic settings, 8192 context window, and strict JSON format schema
        raw_resp = self.client.chat(
            messages=messages,
            format=self.schema,
            options={"num_ctx": 8192, "temperature": 0.0, "seed": 42},
            timeout=self.client.timeout_router,
        )

        result, is_valid, error_msg = self._parse_and_validate(raw_resp)
        if is_valid:
            return result

        # Retry once with error feedback
        logger.info("Router schema validation failed: '%s'. Retrying once...", error_msg)
        retry_messages = list(messages)
        retry_messages.append({"role": "assistant", "content": str(raw_resp.get("message", {}).get("content", ""))})
        retry_messages.append(
            {
                "role": "user",
                "content": f"The previous classification was invalid: {error_msg}. Please correct the JSON output.",
            }
        )

        try:
            retry_resp = self.client.chat(
                messages=retry_messages,
                format=self.schema,
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
        """Retries routing by feeding execution error back to LLM."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        if previous_raw_response:
            messages.append({"role": "assistant", "content": str(previous_raw_response.get("message", {}).get("content", ""))})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"The previous structured query failed: {error_msg}. "
                    f"Please correct the JSON parameters using valid values or set route_type='ABSTAIN'."
                ),
            }
        )

        try:
            retry_resp = self.client.chat(
                messages=messages,
                format=self.schema,
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
        """Parses decoded JSON object from Ollama content and constructs normalized RouterResult."""
        message = response.get("message", {})
        content = message.get("content", "")

        if not content or not content.strip():
            return (
                RouterResult(
                    route_type=RouteType.ABSTAIN,
                    abstain_reason=AbstainReason.MALFORMED_TOOL_CALL,
                    raw_response=response,
                ),
                False,
                "Empty content received from model.",
            )

        try:
            payload = json.loads(content)
        except Exception as e:
            return (
                RouterResult(
                    route_type=RouteType.ABSTAIN,
                    abstain_reason=AbstainReason.MALFORMED_TOOL_CALL,
                    raw_response=response,
                ),
                False,
                f"Malformed JSON content: {e}",
            )

        route_type_str = payload.get("route_type", "ABSTAIN")
        struct_block = payload.get("structured", {})
        vector_block = payload.get("vector", {})
        abstain_reason_str = payload.get("abstain_reason", "NONE")

        parsed_calls: List[ToolCall] = []

        if route_type_str == "ABSTAIN":
            if abstain_reason_str == "OUT_OF_SCOPE_DLC":
                reason = AbstainReason.OUT_OF_SCOPE_DLC
            elif abstain_reason_str == "NO_EVIDENCE":
                reason = AbstainReason.NO_EVIDENCE
            else:
                reason = AbstainReason.OUT_OF_SCOPE_OTHER

            parsed_calls.append(
                ToolCall(
                    name="abstain",
                    arguments={"reason": reason.value.lower()},
                )
            )
            return (
                RouterResult(
                    route_type=RouteType.ABSTAIN,
                    tool_calls=parsed_calls,
                    abstain_reason=reason,
                    raw_response=response,
                ),
                True,
                "",
            )

        if route_type_str in ("STRUCTURED", "BOTH"):
            struct_call, val_err = self._translate_structured_block(struct_block)
            if val_err:
                return (
                    RouterResult(route_type=RouteType.ABSTAIN, raw_response=response),
                    False,
                    val_err,
                )
            if struct_call:
                parsed_calls.append(struct_call)

        if route_type_str in ("VECTOR", "BOTH"):
            qtext = vector_block.get("query_text", "")
            if not qtext or not qtext.strip():
                return (
                    RouterResult(route_type=RouteType.ABSTAIN, raw_response=response),
                    False,
                    "VECTOR route requires non-empty vector.query_text.",
                )
            parsed_calls.append(
                ToolCall(
                    name="search_knowledge_base",
                    arguments={"query_text": qtext.strip()},
                )
            )

        if route_type_str == "BOTH":
            rt = RouteType.BOTH
        elif route_type_str == "STRUCTURED":
            rt = RouteType.STRUCTURED
        elif route_type_str == "VECTOR":
            rt = RouteType.VECTOR
        else:
            rt = RouteType.ABSTAIN

        return (
            RouterResult(
                route_type=rt,
                tool_calls=parsed_calls,
                abstain_reason=None,
                raw_response=response,
            ),
            True,
            "",
        )

    def _translate_structured_block(self, block: Dict[str, Any]) -> Tuple[Optional[ToolCall], Optional[str]]:
        """Translates grammar schema structured block into canonical query_structured_data ToolCall."""
        op = block.get("operation", "none")
        if op == "none":
            return None, "Structured operation cannot be 'none' for structured route."

        op_map = {
            "lookup_entity": "fact_lookup",
            "compare_entities": "ranking",
            "production_chain": "production_chain",
            "list_category": "category_listing",
            "sector_yield": "sector_yield",
        }
        query_type = op_map.get(op)
        if not query_type:
            return None, f"Unknown structured operation '{op}'."

        args: Dict[str, Any] = {"query_type": query_type}

        # Map entity name
        qname = block.get("query_name", "")
        if qname and qname != "none":
            args["entity_name"] = qname

        # Map metric
        metric = block.get("metric", "none")
        if metric and metric != "none":
            args["metric"] = metric

        # Map ship class
        sclass = block.get("ship_class", "none")
        if sclass and sclass != "none":
            args["ship_class"] = sclass

        # Map purpose
        purp = block.get("purpose", "none")
        if purp and purp != "none":
            args["purpose"] = purp

        # Map category
        cat = block.get("category", "none")
        if cat and cat != "none":
            args["category"] = cat

        # Map faction
        fac = block.get("faction", "")
        if fac and fac != "none":
            args["faction"] = fac

        # Map resource
        res = block.get("resource_id", "none")
        if res and res != "none":
            args["resource_id"] = res

        # Map production method
        pm = block.get("production_method", "none")
        if pm and pm != "none":
            args["production_method"] = pm

        if "sort_desc" in block:
            args["sort_desc"] = block["sort_desc"]

        if "limit" in block and block["limit"] is not None:
            args["limit"] = block["limit"]

        # Validate coherence
        coherence_err = self._validate_coherence(args)
        if coherence_err:
            return None, coherence_err

        return ToolCall(name="query_structured_data", arguments=args), None

    def _validate_coherence(self, args: Dict[str, Any]) -> Optional[str]:
        """Validates cross-parameter coherence."""
        metric = args.get("metric")
        if metric:
            if metric in ALLOWED_SHIP_METRICS and args.get("category"):
                return f"Incoherent parameters: ship metric '{metric}' cannot be combined with ware category '{args.get('category')}'."
            if metric in ALLOWED_WARE_METRICS and (args.get("ship_class") or args.get("purpose")):
                return f"Incoherent parameters: ware metric '{metric}' cannot be combined with ship_class or purpose."
        return None

    def _validate_structured_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validates structured tool arguments for integrity checking."""
        return self._validate_coherence(args)

