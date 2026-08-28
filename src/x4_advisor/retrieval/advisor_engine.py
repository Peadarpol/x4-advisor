"""Advisor Engine coordinating LLM Router, Structured Query Engine, Vector Query Engine, and Grounded Synthesizer."""

import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from x4_advisor.llm.synthesizer import GroundedSynthesizer
from x4_advisor.embeddings.ollama_embedder import OllamaEmbedder
from x4_advisor.llm.client import OllamaClient
from x4_advisor.retrieval.entity_resolver import EntityResolver
from x4_advisor.retrieval.models import (
    AbstainReason,
    AdvisorResponse,
    AmbiguousEntityResult,
    DatabaseNotReadyError,
    EntityNotFoundResult,
    ResolvedEntity,
    RouterResult,
    RouteType,
    SynthesisResult,
    ToolCall,
    UnknownFilterValue,
    VectorSearchResult,
)
from x4_advisor.retrieval.router import LLMRouter
from x4_advisor.retrieval.structured_query import StructuredQueryEngine
from x4_advisor.retrieval.vector_query import VectorQueryEngine

logger = logging.getLogger(__name__)

CORE_TABLES = [
    "ships",
    "wares",
    "sectors",
    "sector_resources",
    "factions",
    "production_recipes",
]


class AdvisorEngine:
    """End-to-end query coordinator implementing single-turn grounded advisory retrieval."""

    def __init__(
        self,
        config: Optional[Config] = None,
        conn: Optional[sqlite3.Connection] = None,
        db_path: Optional[Path] = None,
        client: Optional[OllamaClient] = None,
        embedder: Optional[OllamaEmbedder] = None,
        router: Optional[LLMRouter] = None,
        synthesizer: Optional[GroundedSynthesizer] = None,
    ) -> None:
        self.config = config or get_config(validate=False)

        # Connection lifecycle management
        if conn is not None:
            self.conn = conn
            self._close_conn_on_exit = False
        else:
            path = db_path or self.config.database_path
            path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(path))
            try:
                import sqlite_vec
                self.conn.enable_load_extension(True)
                sqlite_vec.load(self.conn)
            except Exception as e:
                logger.debug("sqlite-vec load extension skipped: %s", e)
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self._close_conn_on_exit = True

        self.conn.execute("PRAGMA query_only = ON;")

        # Pre-check database readiness across all 6 core tables
        self._verify_database_readiness()

        # Initialize shared components (hang-protection circuit breakers, not SLA enforcement)
        self.client = client or OllamaClient(
            endpoint=self.config.ollama_endpoint,
            model_name=self.config.model_name or "gemma4:12b",
            keep_alive="10m",
            timeout_router=15.0,
            timeout_synthesizer=45.0,
        )

        self.embedder = embedder or OllamaEmbedder(
            endpoint=self.config.ollama_endpoint,
            model_name=self.config.embedding_model,
            timeout_seconds=10.0,
            keep_alive="10m",
        )

        self.structured_engine = StructuredQueryEngine(conn=self.conn)
        self.entity_resolver = EntityResolver(conn=self.conn)

        # VectorQueryEngine wiring with explicit threshold injection
        self.vector_engine = VectorQueryEngine(
            conn=self.conn,
            embedder=self.embedder,
            default_threshold=self.config.vector_relevance_threshold,
        )

        if self.config.vector_relevance_threshold_is_default:
            logger.warning(
                "Using uncalibrated placeholder VECTOR_RELEVANCE_THRESHOLD=%.2f pending M6 empirical calibration.",
                self.config.vector_relevance_threshold,
            )

        self.router = router or LLMRouter(client=self.client)
        if synthesizer is not None:
            self.synthesizer = synthesizer
        else:
            from x4_advisor.llm.synthesizer import GroundedSynthesizer
            self.synthesizer = GroundedSynthesizer(client=self.client)

    def _verify_database_readiness(self) -> None:
        """Verifies that all 6 core tables exist and are populated with records."""
        cursor = self.conn.cursor()
        existing_tables = {
            r[0]
            for r in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        missing = [t for t in CORE_TABLES if t not in existing_tables]
        if missing:
            raise DatabaseNotReadyError(
                f"Database is not initialized or missing core tables: {missing}. "
                "Please run the Milestone M1 ingestion pipeline first."
            )

        # Check that core tables are populated with data
        for table in CORE_TABLES:
            count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count == 0:
                raise DatabaseNotReadyError(
                    f"Core table '{table}' is empty. "
                    "Please run the Milestone M1 ingestion pipeline to populate game data."
                )

    def close(self) -> None:
        """Closes the SQLite database connection if owned by this instance (idempotent)."""
        if self._close_conn_on_exit and self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def __enter__(self) -> "AdvisorEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _normalize_ship_class(self, raw_class: Optional[str]) -> Optional[str]:
        """Normalizes natural ship class names ('s', 'm', 'l', 'xl', 'large') to storage enums ('ship_l')."""
        if not raw_class:
            return None
        clean = str(raw_class).strip().lower()
        mapping = {
            "s": "ship_s",
            "small": "ship_s",
            "ship_s": "ship_s",
            "m": "ship_m",
            "medium": "ship_m",
            "ship_m": "ship_m",
            "l": "ship_l",
            "large": "ship_l",
            "ship_l": "ship_l",
            "xl": "ship_xl",
            "extra_large": "ship_xl",
            "extralarge": "ship_xl",
            "ship_xl": "ship_xl",
            "spacesuit": "spacesuit",
        }
        return mapping.get(clean, clean)

    def answer(
        self,
        question: str,
        pending_route: Optional[RouterResult] = None,
        resolved_entity_id: Optional[str] = None,
    ) -> AdvisorResponse:
        """Executes full single-turn query routing, retrieval, and grounded answer synthesis.

        Args:
            question: Natural language question from the user.
            pending_route: Continuation token representing the pending routing decision
                when resuming from an ambiguous entity clarification.
            resolved_entity_id: User-selected entity ID settling an ambiguous entity match.
        """
        notes: List[str] = []
        structured_result: Optional[Any] = None
        vector_result: Optional[VectorSearchResult] = None
        ambiguous_candidates: Optional[List[ResolvedEntity]] = None

        # ---------------------------------------------------------------------
        # Resumed Disambiguation Execution (Single-turn narrow exception)
        # ---------------------------------------------------------------------
        if resolved_entity_id and str(resolved_entity_id).strip():
            clean_id = str(resolved_entity_id).strip()
            route_result = pending_route or RouterResult(
                route_type=RouteType.STRUCTURED,
                tool_calls=[ToolCall(name="query_structured_data", arguments={"query_type": "fact_lookup", "entity_name": clean_id})],
            )

            # Re-execute pending structured query using the resolved ID directly
            primary_call = next(
                (tc for tc in route_result.tool_calls if tc.name == "query_structured_data"),
                None,
            )
            qtype = primary_call.arguments.get("query_type", "fact_lookup") if primary_call else "fact_lookup"

            if qtype == "production_chain":
                method = primary_call.arguments.get("production_method", "default") if primary_call else "default"
                structured_result = self.structured_engine.query_t3_production_chain(clean_id, method=method)
            else:
                structured_result = self.structured_engine.query_t1_fact_lookup(clean_id)

            # If the original route was BOTH, also run the companion vector search
            if route_result.route_type == RouteType.BOTH:
                vec_call = next((tc for tc in route_result.tool_calls if tc.name == "search_knowledge_base"), None)
                v_query = vec_call.arguments.get("query_text", question) if vec_call else question
                vector_result = self.vector_engine.search(v_query)

            synth_timeout = 30.0 if route_result.route_type == RouteType.BOTH else 25.0
            synth_res = self.synthesizer.synthesize(
                question=question,
                structured_result=structured_result,
                vector_result=vector_result,
                notes=notes,
                timeout=synth_timeout,
            )

            return AdvisorResponse(
                question=question,
                route_result=route_result,
                structured_result=structured_result,
                vector_result=vector_result,
                synthesis_result=synth_res,
                ambiguous_candidates=None,
                pending_route=None,
            )

        # ---------------------------------------------------------------------
        # Standard Initial Routing Step
        # ---------------------------------------------------------------------
        route_result = self.router.route(question)
        structured_result: Optional[Any] = None
        vector_result: Optional[VectorSearchResult] = None
        unresolved_filter_err: Optional[UnknownFilterValue] = None

        if route_result.route_type == RouteType.ABSTAIN:
            synth_res = self.synthesizer.synthesize(
                question=question,
                abstain_reason=route_result.abstain_reason or AbstainReason.OUT_OF_SCOPE_OTHER,
                notes=notes,
                timeout=12.0,
            )
            return AdvisorResponse(
                question=question,
                route_result=route_result,
                synthesis_result=synth_res,
            )

        # ---------------------------------------------------------------------
        # Execute Structured Retrieval (if STRUCTURED or BOTH)
        # ---------------------------------------------------------------------
        if route_result.route_type in (RouteType.STRUCTURED, RouteType.BOTH):
            struct_call = next((tc for tc in route_result.tool_calls if tc.name == "query_structured_data"), None)
            if struct_call:
                args = struct_call.arguments
                def _execute_structured_args(call_args: Dict[str, Any]) -> Tuple[Optional[Any], Optional[List[ResolvedEntity]]]:
                    qtype = call_args.get("query_type", "fact_lookup")
                    if qtype == "fact_lookup":
                        entity_name = call_args.get("entity_name", "")
                        resolved = self.entity_resolver.resolve_entity(entity_name)
                        if isinstance(resolved, AmbiguousEntityResult):
                            return None, resolved.candidates
                        elif isinstance(resolved, ResolvedEntity):
                            return self.structured_engine.query_t1_fact_lookup(resolved.id), None
                        return None, None

                    elif qtype == "ranking":
                        raw_class = call_args.get("ship_class")
                        norm_class = self._normalize_ship_class(raw_class)
                        category = call_args.get("category")
                        cat_or_class = norm_class or category
                        return self.structured_engine.query_t2_ranking(
                            category_or_class=cat_or_class,
                            metric=call_args.get("metric", "cargo_capacity"),
                            purpose=call_args.get("purpose"),
                            sort_desc=call_args.get("sort_desc", True),
                            limit=call_args.get("limit", 5),
                        ), None

                    elif qtype == "sector_yield":
                        return self.structured_engine.query_t2_sector_yield_ranking(
                            resource_id=call_args.get("resource_id", "ore"),
                            limit=call_args.get("limit", 5),
                        ), None

                    elif qtype == "production_chain":
                        target_name = call_args.get("entity_name", "")
                        resolved = self.entity_resolver.resolve_entity(target_name, entity_types=["ware"])
                        if isinstance(resolved, AmbiguousEntityResult):
                            return None, resolved.candidates
                        elif isinstance(resolved, ResolvedEntity):
                            return self.structured_engine.query_t3_production_chain(
                                ware_id=resolved.id,
                                method=call_args.get("production_method", "default"),
                            ), None
                        return None, None

                    elif qtype == "category_listing":
                        filter_type = "category"
                        filter_val = call_args.get("category", "")
                        if call_args.get("faction"):
                            filter_type = "faction"
                            filter_val = call_args.get("faction", "")
                        elif call_args.get("ship_class"):
                            filter_type = "ship_class"
                            filter_val = self._normalize_ship_class(call_args.get("ship_class")) or ""
                        elif call_args.get("purpose"):
                            filter_type = "purpose"
                            filter_val = call_args.get("purpose", "")

                        return self.structured_engine.query_t4_category_listing(
                            filter_type=filter_type,
                            filter_value=filter_val,
                            limit=call_args.get("limit", 50),
                        ), None
                    return None, None

                unresolved_filter_err: Optional[UnknownFilterValue] = None
                try:
                    s_res, amb_res = _execute_structured_args(args)
                    if amb_res:
                        ambiguous_candidates = amb_res
                        synth_res = self.synthesizer.synthesize(
                            question=question,
                            ambiguous_candidates=ambiguous_candidates,
                        )
                        return AdvisorResponse(
                            question=question,
                            route_result=route_result,
                            synthesis_result=synth_res,
                            ambiguous_candidates=ambiguous_candidates,
                            pending_route=route_result,
                        )
                    structured_result = s_res
                except UnknownFilterValue as e:
                    logger.info("Structured query raised UnknownFilterValue: %s. Attempting single retry with router feedback...", e)
                    # Feed the error and valid values back into router
                    retry_route = self.router.retry_with_feedback(
                        question=question,
                        previous_raw_response=route_result.raw_response,
                        error_msg=str(e),
                    )
                    retry_struct_call = next((tc for tc in retry_route.tool_calls if tc.name == "query_structured_data"), None)
                    if retry_struct_call:
                        try:
                            s_res, amb_res = _execute_structured_args(retry_struct_call.arguments)
                            if amb_res:
                                ambiguous_candidates = amb_res
                                synth_res = self.synthesizer.synthesize(
                                    question=question,
                                    ambiguous_candidates=ambiguous_candidates,
                                )
                                return AdvisorResponse(
                                    question=question,
                                    route_result=retry_route,
                                    synthesis_result=synth_res,
                                    ambiguous_candidates=ambiguous_candidates,
                                    pending_route=retry_route,
                                )
                            structured_result = s_res
                            route_result = retry_route
                        except UnknownFilterValue as e2:
                            logger.warning("Router retry structured query also raised UnknownFilterValue: %s", e2)
                            unresolved_filter_err = e2
                            notes.append(str(e2))
                    else:
                        unresolved_filter_err = e
                        notes.append(str(e))
                        if retry_route.route_type == RouteType.ABSTAIN:
                            route_result = retry_route

        # ---------------------------------------------------------------------
        # Execute Vector Retrieval (if VECTOR or BOTH)
        # ---------------------------------------------------------------------
        if route_result.route_type in (RouteType.VECTOR, RouteType.BOTH):
            vec_call = next((tc for tc in route_result.tool_calls if tc.name == "search_knowledge_base"), None)
            query_text = vec_call.arguments.get("query_text", question) if vec_call else question
            vector_result = self.vector_engine.search(query_text)
            if vector_result and vector_result.status == "database_not_ready":
                raise DatabaseNotReadyError(f"Vector database not ready: {vector_result.error_message}")

        # ---------------------------------------------------------------------
        # Per-Route Evidence & Abstention Assessment
        # ---------------------------------------------------------------------
        abstain_reason: Optional[AbstainReason] = None

        if route_result.route_type == RouteType.STRUCTURED:
            if structured_result is None or (
                hasattr(structured_result, "items") and not structured_result.items
            ):
                abstain_reason = AbstainReason.NO_EVIDENCE

        elif route_result.route_type == RouteType.VECTOR:
            if not vector_result or not vector_result.chunks or vector_result.status == "no_relevant_chunks":
                abstain_reason = AbstainReason.NO_EVIDENCE

        elif route_result.route_type == RouteType.BOTH:
            struct_empty = structured_result is None or (
                hasattr(structured_result, "items") and not structured_result.items
            )
            vector_empty = not vector_result or not vector_result.chunks or vector_result.status == "no_relevant_chunks"

            if struct_empty and vector_empty:
                abstain_reason = AbstainReason.NO_EVIDENCE
            elif vector_empty:
                notes.append("Partial retrieval: vector strategy search yielded no matching context above threshold.")
            elif struct_empty:
                notes.append("Partial retrieval: structured game lookup returned no records.")

        # ---------------------------------------------------------------------
        # Unresolved Filter Value Direct Response
        # ---------------------------------------------------------------------
        if unresolved_filter_err and structured_result is None:
            synth_res = SynthesisResult(
                answer_text=(
                    f"The requested {unresolved_filter_err.field} '{unresolved_filter_err.attempted_value}' was not recognized "
                    f"in the database. Valid options are: {', '.join(unresolved_filter_err.valid_values)}."
                ),
                has_evidence=False,
                abstain_reason=AbstainReason.NO_EVIDENCE,
                evidence_chunk_ids=[],
                was_method_fallback=False,
                notes=notes,
            )
            return AdvisorResponse(
                question=question,
                route_result=route_result,
                synthesis_result=synth_res,
            )

        # ---------------------------------------------------------------------
        # Grounded Synthesis (synth_timeout is hang-protection only)
        # ---------------------------------------------------------------------
        synth_timeout = 30.0 if route_result.route_type == RouteType.BOTH else 25.0
        synth_res = self.synthesizer.synthesize(
            question=question,
            structured_result=structured_result,
            vector_result=vector_result,
            abstain_reason=abstain_reason,
            notes=notes,
            timeout=synth_timeout,
        )

        return AdvisorResponse(
            question=question,
            route_result=route_result,
            structured_result=structured_result,
            vector_result=vector_result,
            synthesis_result=synth_res,
            ambiguous_candidates=None,
            pending_route=None,
        )
