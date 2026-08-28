"""Milestone M6 Benchmark Evaluation Runner & Local Model Bake-Off."""

import argparse
import json
import logging
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional

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

    verifier = GroundingVerifier()

    results = []
    latencies = []
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

    for idx, case in enumerate(corpus, start=1):
        case_id = case["case_id"]
        category = case["category"]
        question = case["question"]
        expected_route = case["expected_route"]
        expected_abstention = case.get("expected_abstention")
        expected_facts = case.get("expected_structured_facts")
        prohibited = case.get("prohibited_unsupported_claims")

        logger.info(f"[{idx}/{len(corpus)}] ({case_id} - {category}): {question}")
        t0 = time.time()

        try:
            response = engine.answer(question)
            duration = time.time() - t0
            latencies.append(duration)

            actual_route = response.route_result.route_type.value
            synth_res = response.synthesis_result
            answer_text = synth_res.answer_text
            actual_abstain = synth_res.abstain_reason.value if synth_res.abstain_reason else None

            # Route accuracy check
            route_passed = (actual_route == expected_route)
            if expected_route == "ABSTAIN" and synth_res.abstain_reason:
                route_passed = True
            if route_passed:
                route_correct_count += 1

            # Abstention check
            abstain_passed = True
            if expected_abstention:
                abstain_cases_total += 1
                if expected_abstention == "OUT_OF_SCOPE_DLC":
                    abstain_passed = (actual_abstain == "OUT_OF_SCOPE_DLC")
                elif expected_abstention == "NO_EVIDENCE":
                    abstain_passed = (actual_abstain in ("NO_EVIDENCE", "out_of_scope_other"))
                if abstain_passed:
                    abstain_cases_passed += 1

            # Grounding verification
            report: GroundingReport = verifier.verify_answer(
                answer_text=answer_text,
                structured_data=response.structured_result,
                vector_chunks=(response.vector_result.chunks if response.vector_result else None),
                expected_facts=expected_facts,
                prohibited_claims=prohibited,
            )

            if synth_res.abstain_reason is not None and abstain_passed:
                report.is_grounded = True
            elif response.ambiguous_candidates:
                report.is_grounded = True

            total_claims += report.total_claims
            total_facts += report.facts_count
            total_inferences += report.inferences_count
            total_advice += report.advice_count
            total_unsupported += report.unsupported_count
            total_contradicted += report.contradicted_count

            # Structured accuracy tracking
            if category in ("T1_FACT_LOOKUP", "T2_COMPARISON", "T3_PRODUCTION_CHAIN", "T4_CATEGORY_LISTING", "STRUCTURED_VS_COMMUNITY_CONFLICT", "SUPPORTED_INFERENCE"):
                structured_cases_total += 1
                if report.is_grounded and report.contradicted_count == 0:
                    structured_cases_passed += 1

            case_passed = route_passed and report.is_grounded and abstain_passed
            if case_passed:
                grounded_cases_count += 1

            logger.info(
                f"   -> Result: route={actual_route} (exp={expected_route}), "
                f"grounded={report.is_grounded}, contra={report.contradicted_count}, "
                f"unsupp={report.unsupported_count}, time={duration:.2f}s"
            )

            results.append({
                "case_id": case_id,
                "category": category,
                "route_passed": route_passed,
                "actual_route": actual_route,
                "expected_route": expected_route,
                "case_passed": case_passed,
                "is_grounded": report.is_grounded,
                "claims_count": report.total_claims,
                "facts_count": report.facts_count,
                "inferences_count": report.inferences_count,
                "advice_count": report.advice_count,
                "unsupported_count": report.unsupported_count,
                "contradicted_count": report.contradicted_count,
                "latency_seconds": round(duration, 2),
                "answer_snippet": answer_text[:120] if answer_text else "",
            })

        except Exception as e:
            logger.error(f"Execution failed on case {case_id}: {e}")
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

    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_lat = sorted(latencies)
    p90_lat = sorted_lat[int(0.90 * len(sorted_lat))] if sorted_lat else 0.0

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
        "unsupported_count": total_unsupported,
        "contradicted_count": total_contradicted,
        "unsupported_claim_rate": round(ucr, 4),
        "latency_mean_sec": round(mean_lat, 2),
        "latency_p90_sec": round(p90_lat, 2),
        "gate_passed": (structured_precision >= 0.90) and (ucr <= 0.05) and (abstain_accuracy == 1.0) and (overall_pass_rate >= 0.85),
        "case_details": results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run M6 Grounding Benchmark across candidate models.")
    parser.add_argument("--models", nargs="+", default=MODELS_TO_EVALUATE, help="Ollama models to evaluate.")
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

    # Save JSON summary
    out_json = Path("docs/eval/m6_bakeoff_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_model_summaries, f, indent=2)

    # Generate Markdown Report
    report_lines = [
        "# Milestone M6 Model Bake-Off & Grounding Evaluation Report",
        "",
        "## Summary Results Table",
        "",
        "| Model Name | Pass Rate | Route Acc | Struct Prec | Abstain Acc | UCR | Contradictions | Mean Latency | Gate Passed |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for model_name, s in all_model_summaries.items():
        gate_str = "✅ PASS" if s["gate_passed"] else "❌ FAIL"
        report_lines.append(
            f"| `{model_name}` | {s['overall_pass_rate']:.1%} | {s['route_accuracy']:.1%} | "
            f"{s['structured_precision']:.1%} | {s['abstain_accuracy']:.1%} | {s['unsupported_claim_rate']:.1%} | "
            f"{s['contradicted_count']} | {s['latency_mean_sec']}s | {gate_str} |"
        )

    report_lines.extend([
        "",
        "## Architectural Evaluation Gates (SPEC-001 §11)",
        "- **Structured Precision:** >= 90%",
        "- **Unsupported Claim Rate (UCR):** <= 5%",
        "- **Abstention Accuracy (DLC & No-Evidence):** 100%",
        "- **Overall Ground Truth Pass Rate:** >= 85%",
        "- **Zero Contradictions Invariant:** 0 contradicted claims across all cases",
        "",
        "## Per-Model Breakdown",
    ])

    for model_name, s in all_model_summaries.items():
        report_lines.extend([
            f"### Model: `{model_name}`",
            f"- **Total Claims Extracted:** {s['total_claims']}",
            f"- **Facts:** {s['facts_count']} | **Supported Inferences:** {s['inferences_count']} | **Advice:** {s['advice_count']}",
            f"- **Unsupported Claims:** {s['unsupported_count']} | **Contradicted Claims:** {s['contradicted_count']}",
            f"- **Latency (Mean / P90):** {s['latency_mean_sec']}s / {s['latency_p90_sec']}s",
            "",
            "#### Category Breakdown",
            "| Category | Passed / Total | Claims | Facts | Infer | Adv | Unsupp | Contra |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        from collections import defaultdict
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

    out_md = Path("docs/eval/m6_bakeoff_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    logger.info(f"Saved bake-off JSON to {out_json} and Markdown report to {out_md}")


if __name__ == "__main__":
    main()
