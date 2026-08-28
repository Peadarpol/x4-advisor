"""Milestone M7 Benchmark Evaluation Runner & Local Model Bake-Off."""

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from x4_advisor.config import Config, get_config
from x4_advisor.embeddings.ollama_embedder import OllamaEmbedder
from x4_advisor.grounding.grounding_verifier import GroundingVerifier
from x4_advisor.grounding.taxonomy import ClaimClass, GroundingReport
from x4_advisor.llm.client import OllamaClient
from x4_advisor.llm.synthesizer import GroundedSynthesizer
from x4_advisor.retrieval.advisor_engine import AdvisorEngine
from x4_advisor.retrieval.models import AbstainReason, RouteType
from x4_advisor.retrieval.router import LLMRouter
from x4_advisor.storage.db import get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_benchmark")

MODELS_TO_EVALUATE = ["gemma4:12b", "granite4.1:8b", "qwen3:14b"]


def _compare_structured_tool_args(actual_args: Dict[str, Any], expected_args: Dict[str, Any]) -> bool:
    """Semantically compares actual tool arguments against expected tool arguments."""
    exp_qtype = expected_args.get("query_type")
    act_qtype = actual_args.get("query_type")
    if exp_qtype and exp_qtype != act_qtype:
        return False

    # Check entity_name (allow case-insensitive match or substring)
    exp_ent = expected_args.get("entity_name")
    act_ent = actual_args.get("entity_name")
    if exp_ent:
        if not act_ent or exp_ent.lower() not in act_ent.lower() and act_ent.lower() not in exp_ent.lower():
            return False

    # Check metric
    exp_metric = expected_args.get("metric")
    act_metric = actual_args.get("metric")
    if exp_metric and exp_metric != act_metric:
        return False

    # Check closed-set filters
    for field in ("ship_class", "purpose", "category", "faction", "production_method", "resource_id"):
        exp_val = expected_args.get(field)
        act_val = actual_args.get(field)
        if exp_val and exp_val != act_val:
            return False

    return True


def _serialize_structured_result(sr: Any) -> Any:
    """Serializes heterogeneous structured query results into JSON-serializable list of row dicts."""
    if sr is None:
        return []
    if hasattr(sr, "data"):  # SingleEntityResult
        return [{"entity_id": getattr(sr, "entity_id", ""), "entity_name": getattr(sr, "entity_name", ""), "data": sr.data}]
    if hasattr(sr, "items"):  # RankingResult, CategoryListResult
        items = getattr(sr, "items", [])
        return [
            {
                "id": getattr(it, "id", it.get("id", "") if isinstance(it, dict) else ""),
                "name": getattr(it, "name", it.get("name", "") if isinstance(it, dict) else ""),
                "value": getattr(it, "value", it.get("value", None) if isinstance(it, dict) else None),
                "metric": getattr(it, "metric_name", it.get("metric_name", "") if isinstance(it, dict) else ""),
                "unit": getattr(it, "unit", it.get("unit", "") if isinstance(it, dict) else ""),
            } if not isinstance(it, dict) else it
            for it in items
        ]
    if hasattr(sr, "total_raw_materials"):  # ProductionChainResult
        return [{
            "target_ware_id": getattr(sr, "target_ware_id", ""),
            "target_ware_name": getattr(sr, "target_ware_name", ""),
            "method": getattr(sr, "method", ""),
            "output_amount": getattr(sr, "output_amount", 0),
            "production_time": getattr(sr, "production_time", 0.0),
            "total_raw_materials": getattr(sr, "total_raw_materials", {}),
        }]
    if isinstance(sr, (list, tuple)):
        return list(sr)
    if isinstance(sr, dict):
        return [sr]
    return [str(sr)]


def _get_structured_row_count(sr: Any) -> int:
    """Returns row/item count for heterogeneous structured query results."""
    if sr is None:
        return 0
    if hasattr(sr, "data"):
        return 1
    if hasattr(sr, "items"):
        return len(getattr(sr, "items", []))
    if hasattr(sr, "total_raw_materials"):
        return len(getattr(sr, "total_raw_materials", {}))
    if isinstance(sr, (list, tuple, dict)):
        return len(sr)
    return 1


def _evaluate_tool_calls_semantic(
    actual_tool_calls: List[Dict[str, Any]],
    expected_tool_calls: List[Dict[str, Any]],
) -> bool:
    """Verifies that all expected tool calls were emitted with compatible arguments."""
    if not expected_tool_calls:
        return True

    for exp_tc in expected_tool_calls:
        exp_name = exp_tc.get("name")
        exp_args = exp_tc.get("arguments", {})

        matched = False
        for act_tc in actual_tool_calls:
            act_name = act_tc.get("name")
            act_args = act_tc.get("arguments", {})

            if exp_name != act_name:
                continue

            if exp_name == "query_structured_data":
                if _compare_structured_tool_args(act_args, exp_args):
                    matched = True
                    break
            elif exp_name == "search_knowledge_base":
                if "query_text" in act_args and str(act_args["query_text"]).strip():
                    matched = True
                    break
            elif exp_name == "abstain":
                exp_reason = str(exp_args.get("reason", "")).lower()
                act_reason = str(act_args.get("reason", "")).lower()
                if exp_reason == act_reason or (exp_reason == "no_evidence" and act_reason in ("no_evidence", "out_of_scope_other")):
                    matched = True
                    break

        if not matched:
            return False

    return True


def evaluate_model_on_corpus(
    model_name: str,
    corpus: List[Dict[str, Any]],
    config: Config,
    conn: sqlite3.Connection,
) -> Dict[str, Any]:
    """Runs all 36 evaluation cases for a single model and calculates comprehensive grounding metrics."""
    logger.info(f"\n========================================================")
    logger.info(f"   STARTING EVALUATION RUN FOR MODEL: {model_name}      ")
    logger.info(f"========================================================")

    client = OllamaClient(
        endpoint=config.ollama_endpoint,
        model_name=model_name,
        keep_alive="10m",
        timeout_router=30.0,
        timeout_synthesizer=60.0,
    )
    embedder = OllamaEmbedder(
        endpoint=config.ollama_endpoint,
        model_name=config.embedding_model,
    )
    router = LLMRouter(client=client)
    synthesizer = GroundedSynthesizer(client=client)

    engine = AdvisorEngine(
        config=config,
        conn=conn,
        client=client,
        embedder=embedder,
        router=router,
        synthesizer=synthesizer,
    )

    # Warmup model in VRAM
    client.warmup(embedder=embedder)

    verifier = GroundingVerifier()

    results = []
    single_path_latencies = []
    hybrid_latencies = []
    all_latencies = []

    route_correct_count = 0
    grounded_cases_count = 0
    total_claims = 0
    total_facts = 0
    total_inferences = 0
    total_advice = 0
    total_unsupported = 0
    total_contradicted = 0

    structured_cases_total = 0
    structured_cases_passed = 0
    abstain_cases_total = 0
    abstain_cases_passed = 0

    total_expected_chunks = 0
    total_retrieved_expected_chunks = 0
    total_retrieved_chunks = 0

    for idx, case in enumerate(corpus, start=1):
        case_id = case["case_id"]
        category = case["category"]
        question = case["question"]
        expected_route = case["expected_route"]
        expected_abstention = case.get("expected_abstention")
        expected_facts = case.get("expected_structured_facts")
        prohibited = case.get("prohibited_unsupported_claims")
        expected_tool_calls = case.get("expected_tool_calls", [])
        expected_chunk_ids = case.get("expected_chunk_ids", [])

        logger.info(f"[{idx}/{len(corpus)}] ({case_id} - {category}): {question}")
        client.clear_history()
        t0 = time.time()

        try:
            response = engine.answer(question)
            duration = time.time() - t0
            all_latencies.append(duration)

            actual_route = response.route_result.route_type.value
            if actual_route == "BOTH" or expected_route == "BOTH":
                hybrid_latencies.append(duration)
            else:
                single_path_latencies.append(duration)

            synth_res = response.synthesis_result
            answer_text = synth_res.answer_text
            actual_abstain = synth_res.abstain_reason.value if synth_res.abstain_reason else None

            # Route accuracy check
            route_passed = (actual_route == expected_route)
            if expected_route == "ABSTAIN" and synth_res.abstain_reason:
                route_passed = True
            if route_passed:
                route_correct_count += 1

            # Abstention check (F1 / R4.3: Stated evaluation scoring leniency for NO_EVIDENCE)
            abstain_passed = True
            if expected_abstention:
                abstain_cases_total += 1
                if expected_abstention == "OUT_OF_SCOPE_DLC":
                    abstain_passed = (actual_abstain == "OUT_OF_SCOPE_DLC")
                elif expected_abstention == "NO_EVIDENCE":
                    abstain_passed = (str(actual_abstain).upper() in ("NO_EVIDENCE", "OUT_OF_SCOPE_OTHER"))
                if abstain_passed:
                    abstain_cases_passed += 1

            # Tool call semantic evaluation (F22)
            actual_tool_calls_dicts = [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in response.route_result.tool_calls
            ]
            tool_calls_match = _evaluate_tool_calls_semantic(actual_tool_calls_dicts, expected_tool_calls)

            # Grounding verification
            report: GroundingReport = verifier.verify_answer(
                answer_text=answer_text,
                structured_data=response.structured_result,
                vector_chunks=(response.vector_result.chunks if response.vector_result else None),
                expected_facts=expected_facts,
                prohibited_claims=prohibited,
            )

            # F6: Do not grant free is_grounded on failed routes
            if synth_res.abstain_reason is not None and abstain_passed and route_passed:
                report.is_grounded = True
            elif response.ambiguous_candidates and route_passed:
                report.is_grounded = True

            total_claims += report.total_claims
            total_facts += report.facts_count
            total_inferences += report.inferences_count
            total_advice += report.advice_count
            total_unsupported += report.unsupported_count
            total_contradicted += report.contradicted_count

            # Structured accuracy tracking (wired with tool calls match)
            if category in ("T1_FACT_LOOKUP", "T2_COMPARISON", "T3_PRODUCTION_CHAIN", "T4_CATEGORY_LISTING", "STRUCTURED_VS_COMMUNITY_CONFLICT", "SUPPORTED_INFERENCE"):
                structured_cases_total += 1
                if route_passed and tool_calls_match and report.is_grounded and report.contradicted_count == 0:
                    structured_cases_passed += 1

            case_passed = route_passed and tool_calls_match and report.is_grounded and abstain_passed
            if case_passed:
                grounded_cases_count += 1

            # Retrieval quality scoring (O4)
            retrieved_chunk_ids = [c.chunk_id for c in response.vector_result.chunks] if response.vector_result else []
            retrieved_chunk_objs = [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "similarity": round(getattr(c, "similarity_score", getattr(c, "similarity", 0.0)), 4)
                }
                for c in (response.vector_result.chunks if response.vector_result else [])
            ]
            if expected_chunk_ids:
                total_expected_chunks += len(expected_chunk_ids)
                matched_chunks = set(expected_chunk_ids).intersection(set(retrieved_chunk_ids))
                total_retrieved_expected_chunks += len(matched_chunks)
                total_retrieved_chunks += len(retrieved_chunk_ids)
                chunk_recall = round(len(matched_chunks) / len(expected_chunk_ids), 4)
                chunk_precision = round(len(matched_chunks) / len(retrieved_chunk_ids), 4) if retrieved_chunk_ids else 0.0
            else:
                chunk_recall = 1.0
                chunk_precision = 1.0

            # Retrieval outcome metadata (for M7.1 replay manifest)
            max_sim = max([getattr(c, "similarity_score", getattr(c, "similarity", 0.0)) for c in response.vector_result.chunks], default=0.0) if response.vector_result else 0.0
            retrieval_outcome = {
                "row_count": _get_structured_row_count(response.structured_result),
                "chunk_count": len(response.vector_result.chunks) if response.vector_result else 0,
                "max_similarity": round(max_sim, 4),
                "threshold": config.vector_relevance_threshold,
            }

            # Serialized structured rows
            serialized_rows = _serialize_structured_result(response.structured_result)

            # Capture LLM call telemetry
            llm_telemetry = list(client.call_history)

            logger.info(
                f"   -> Result: route={actual_route} (exp={expected_route}), "
                f"tools_match={tool_calls_match}, grounded={report.is_grounded}, "
                f"contra={report.contradicted_count}, unsupp={report.unsupported_count}, "
                f"time={duration:.2f}s"
            )

            results.append({
                "case_id": case_id,
                "category": category,
                "question": question,
                "expected_route": expected_route,
                "actual_route": actual_route,
                "route_passed": route_passed,
                "expected_tool_calls": expected_tool_calls,
                "emitted_tool_calls": actual_tool_calls_dicts,
                "tool_calls_match": tool_calls_match,
                "case_passed": case_passed,
                "is_grounded": report.is_grounded,
                "claims_count": report.total_claims,
                "facts_count": report.facts_count,
                "inferences_count": report.inferences_count,
                "advice_count": report.advice_count,
                "unsupported_count": report.unsupported_count,
                "contradicted_count": report.contradicted_count,
                "latency_seconds": round(duration, 2),
                "answer_text": answer_text,
                "structured_result": serialized_rows,
                "retrieved_chunks": retrieved_chunk_objs,
                "retrieval_outcome": retrieval_outcome,
                "expected_chunk_ids": expected_chunk_ids,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "chunk_recall": chunk_recall,
                "chunk_precision": chunk_precision,
                "llm_telemetry": llm_telemetry,
            })

        except Exception as e:
            logger.error(f"Execution failed on case {case_id}: {e}", exc_info=True)
            results.append({
                "case_id": case_id,
                "category": category,
                "error": str(e),
                "case_passed": False,
            })

    # Summary calculations
    total_cases = len(corpus)
    route_accuracy = route_correct_count / total_cases if total_cases > 0 else 0.0
    overall_pass_rate = grounded_cases_count / total_cases if total_cases > 0 else 0.0
    structured_precision = structured_cases_passed / structured_cases_total if structured_cases_total > 0 else 0.0
    abstain_accuracy = abstain_cases_passed / abstain_cases_total if abstain_cases_total > 0 else 0.0
    ucr = total_unsupported / total_claims if total_claims > 0 else 0.0

    def _calc_stats(lats: List[float]) -> Tuple[float, float, float]:
        if not lats:
            return 0.0, 0.0, 0.0
        m = sum(lats) / len(lats)
        s = sorted(lats)
        p90 = s[int(0.90 * len(s))]
        mx = max(lats)
        return round(m, 2), round(p90, 2), round(mx, 2)

    all_mean, all_p90, all_max = _calc_stats(all_latencies)
    sp_mean, sp_p90, sp_max = _calc_stats(single_path_latencies)
    hy_mean, hy_p90, hy_max = _calc_stats(hybrid_latencies)

    overall_chunk_recall = round(total_retrieved_expected_chunks / total_expected_chunks, 4) if total_expected_chunks > 0 else 1.0
    overall_chunk_precision = round(total_retrieved_expected_chunks / total_retrieved_chunks, 4) if total_retrieved_chunks > 0 else 1.0

    advice_rate = total_advice / total_claims if total_claims > 0 else 0.0

    # Formal 7-Gate Contract Verification (M7 / SPEC-001 §15)
    gate_passed = (
        (ucr <= 0.03) and
        (total_contradicted == 0) and
        (abstain_accuracy == 1.0) and
        (structured_precision >= 0.90) and
        (route_accuracy >= 0.90) and
        (overall_pass_rate >= 0.85) and
        (sp_p90 < 20.0) and
        (hy_p90 < 30.0) and
        (advice_rate <= 0.15)
    )

    summary = {
        "model_name": model_name,
        "total_cases": total_cases,
        "overall_pass_rate": round(overall_pass_rate, 4),
        "route_accuracy": round(route_accuracy, 4),
        "structured_precision": round(structured_precision, 4),
        "abstain_accuracy": round(abstain_accuracy, 4),
        "total_claims": total_claims,
        "facts_count": total_facts,
        "inferences_count": total_inferences,
        "advice_count": total_advice,
        "advice_rate": round(advice_rate, 4),
        "unsupported_count": total_unsupported,
        "contradicted_count": total_contradicted,
        "unsupported_claim_rate": round(ucr, 4),
        "latency_all": {"mean": all_mean, "p90": all_p90, "max": all_max},
        "latency_single_path": {"mean": sp_mean, "p90": sp_p90, "max": sp_max},
        "latency_hybrid": {"mean": hy_mean, "p90": hy_p90, "max": hy_max},
        "retrieval_chunk_recall": overall_chunk_recall,
        "retrieval_chunk_precision": overall_chunk_precision,
        "gate_passed": gate_passed,
        "case_details": results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run M7 Grounding Benchmark across candidate models.")
    parser.add_argument("--models", nargs="+", default=MODELS_TO_EVALUATE, help="Ollama models to evaluate.")
    parser.add_argument("--output-prefix", type=str, default="m7_0_baseline", help="Prefix for results JSON and report MD.")
    args = parser.parse_args()

    corpus_path = Path("tests/fixtures/eval_corpus.json")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    config = get_config(validate=False)
    conn = get_connection(config.database_path)

    all_model_summaries = {}

    for model_name in args.models:
        summary = evaluate_model_on_corpus(model_name, corpus, config, conn)
        all_model_summaries[model_name] = summary

    conn.close()

    # Save JSON summary (Immutable artifact pattern)
    out_json = Path(f"docs/eval/{args.output_prefix}_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_model_summaries, f, indent=2)

    # Generate Markdown Report
    report_lines = [
        f"# Model Grounding Evaluation Report ({args.output_prefix})",
        "",
        "## Summary Results Table",
        "",
        "| Model Name | Pass Rate | Route Acc | Struct Prec | Abstain Acc | UCR | Contra | Single P90 | Hybrid P90 | Chunk Recall | Gate Passed |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for model_name, s in all_model_summaries.items():
        gate_str = "✅ PASS" if s["gate_passed"] else "❌ FAIL"
        report_lines.append(
            f"| `{model_name}` | {s['overall_pass_rate']:.1%} | {s['route_accuracy']:.1%} | "
            f"{s['structured_precision']:.1%} | {s['abstain_accuracy']:.1%} | {s['unsupported_claim_rate']:.1%} ({s['unsupported_count']}/{s['total_claims']}) | "
            f"{s['contradicted_count']} | {s['latency_single_path']['p90']}s | {s['latency_hybrid']['p90']}s | "
            f"{s['retrieval_chunk_recall']:.1%} | {gate_str} |"
        )

    report_lines.extend([
        "",
        "## Layer 2 Qualification Gate Contract (SPEC-001 §15)",
        "- **Unsupported Claim Rate (UCR):** <= 3.0% (Mandatory Hard Gate)",
        "- **Zero Contradictions Invariant:** 0 contradicted claims (Mandatory Hard Gate)",
        "- **Abstention Accuracy:** 100.0% (Mandatory Hard Gate; N=4 total, N=2 DLC)",
        "- **Structured Precision:** >= 90.0% (Quality Gate)",
        "- **Routing Accuracy:** >= 90.0% (Quality Gate)",
        "- **Overall Ground Truth Pass Rate:** >= 85.0% (Quality Gate)",
        "- **Latency SLAs:** Single-path P90 < 20.0s, Hybrid P90 < 30.0s",
        "- **Modality Distribution:** ADVICE <= 15% of total claims overall",
        "",
        "## Per-Model Breakdown",
    ])

    for model_name, s in all_model_summaries.items():
        report_lines.extend([
            f"### Model: `{model_name}`",
            f"- **Total Claims Extracted:** {s['total_claims']}",
            f"- **Modality Breakdown:** Facts: {s['facts_count']} | Supported Inferences: {s['inferences_count']} | Advice: {s['advice_count']} ({s['advice_rate']:.1%})",
            f"- **Unsupported Claims:** {s['unsupported_count']} | **Contradicted Claims:** {s['contradicted_count']}",
            f"- **Latency (Single-Path):** Mean: {s['latency_single_path']['mean']}s | P90: {s['latency_single_path']['p90']}s | Max: {s['latency_single_path']['max']}s",
            f"- **Latency (Hybrid):** Mean: {s['latency_hybrid']['mean']}s | P90: {s['latency_hybrid']['p90']}s | Max: {s['latency_hybrid']['max']}s",
            f"- **Retrieval Chunk Recall / Precision:** {s['retrieval_chunk_recall']:.1%} / {s['retrieval_chunk_precision']:.1%}",
            "",
            "#### Category Breakdown",
            "| Category | Passed / Total | Claims | Facts | Infer | Adv | Unsupp | Contra |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        cat_stats = defaultdict(lambda: {"total": 0, "facts": 0, "inferences": 0, "advice": 0, "unsupported": 0, "contradicted": 0, "cases": 0, "cases_passed": 0})
        for c in s["case_details"]:
            cat = c["category"]
            cat_stats[cat]["cases"] += 1
            if c.get("case_passed"):
                cat_stats[cat]["cases_passed"] += 1
            cat_stats[cat]["total"] += c.get("claims_count", 0)
            cat_stats[cat]["facts"] += c.get("facts_count", 0)
            cat_stats[cat]["inferences"] += c.get("inferences_count", 0)
            cat_stats[cat]["advice"] += c.get("advice_count", 0)
            cat_stats[cat]["unsupported"] += c.get("unsupported_count", 0)
            cat_stats[cat]["contradicted"] += c.get("contradicted_count", 0)

        for cat, cs in sorted(cat_stats.items()):
            report_lines.append(
                f"| `{cat}` | {cs['cases_passed']}/{cs['cases']} | {cs['total']} | {cs['facts']} | {cs['inferences']} | {cs['advice']} | {cs['unsupported']} | {cs['contradicted']} |"
            )
        report_lines.append("")

    out_md = Path(f"docs/eval/{args.output_prefix}_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    logger.info(f"Saved evaluation JSON to {out_json} and Markdown report to {out_md}")


if __name__ == "__main__":
    main()
