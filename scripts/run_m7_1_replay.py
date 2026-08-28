"""Milestone M7.1 Baseline Replay Runner.

Re-scores the persisted M7.0 baseline answers through the hardened sentence-localized
GroundingVerifier without executing any GPU model inference.
"""

from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from x4_advisor.grounding.grounding_verifier import GroundingVerifier
from x4_advisor.grounding.taxonomy import GroundingReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m7_1_replay")


def replay_baseline(input_path: Path, output_prefix: str) -> Dict[str, Any]:
    """Replays persisted baseline cases through the new GroundingVerifier."""
    with open(input_path, "r", encoding="utf-8") as f:
        m7_0_data = json.load(f)

    corpus_path = Path("tests/fixtures/eval_corpus.json")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    corpus_by_id = {c["case_id"]: c for c in corpus}

    verifier = GroundingVerifier()
    rescored_models = {}

    for model_name, summary_0 in m7_0_data.items():
        logger.info(f"Re-scoring baseline for model: {model_name}")

        results = []
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

        for case_detail in summary_0["case_details"]:
            cid = case_detail["case_id"]
            corpus_case = corpus_by_id.get(cid, {})

            answer_text = case_detail.get("answer_text", "")
            structured_data = case_detail.get("structured_result")
            retrieved_chunks = case_detail.get("retrieved_chunks")
            retrieval_outcome = case_detail.get("retrieval_outcome")

            expected_route = case_detail.get("expected_route")
            actual_route = case_detail.get("actual_route")
            route_passed = case_detail.get("route_passed", False)
            if route_passed:
                route_correct_count += 1

            expected_abstention = corpus_case.get("expected_abstention")
            expected_facts = corpus_case.get("expected_structured_facts")
            prohibited = corpus_case.get("prohibited_unsupported_claims")

            # Re-score through hardened verifier
            report: GroundingReport = verifier.verify_answer(
                answer_text=answer_text,
                structured_data=structured_data,
                vector_chunks=retrieved_chunks,
                expected_facts=expected_facts,
                prohibited_claims=prohibited,
                retrieval_outcome=retrieval_outcome,
            )

            # Abstention evaluation
            abstain_passed = True
            if expected_abstention:
                abstain_cases_total += 1
                # Check if case was correctly refused
                if expected_abstention == "OUT_OF_SCOPE_DLC":
                    abstain_passed = (actual_route == "ABSTAIN" and (
                        any("dlc" in str(tc.get("arguments", {})).lower() for tc in case_detail.get("emitted_tool_calls", []))
                        or "out_of_scope_dlc" in str(case_detail.get("emitted_tool_calls", [])).lower()
                    ))
                elif expected_abstention == "NO_EVIDENCE":
                    # Stated leniency: accept router-level refusal OR grounded negative disclaimer
                    has_tool_refusal = any(
                        r in str(tc.get("arguments", {})).lower()
                        for tc in case_detail.get("emitted_tool_calls", [])
                        for r in ("no_evidence", "out_of_scope_other")
                    )
                    has_grounded_disclaimer = any(
                        c.classification.value == "FACT" and any(w in c.text.lower() for w in ["does not contain", "no information", "no evidence"])
                        for c in report.claims
                    )
                    abstain_passed = has_tool_refusal or has_grounded_disclaimer

                if abstain_passed:
                    abstain_cases_passed += 1

            if actual_route == "ABSTAIN" and abstain_passed and route_passed:
                report.is_grounded = True

            total_claims += report.total_claims
            total_facts += report.facts_count
            total_inferences += report.inferences_count
            total_advice += report.advice_count
            total_unsupported += report.unsupported_count
            total_contradicted += report.contradicted_count

            # Structured accuracy tracking
            tool_calls_match = case_detail.get("tool_calls_match", False)
            cat = case_detail.get("category", "")
            if cat in ("T1_FACT_LOOKUP", "T2_COMPARISON", "T3_PRODUCTION_CHAIN", "T4_CATEGORY_LISTING", "STRUCTURED_VS_COMMUNITY_CONFLICT", "SUPPORTED_INFERENCE"):
                structured_cases_total += 1
                if route_passed and tool_calls_match and report.is_grounded and report.contradicted_count == 0:
                    structured_cases_passed += 1

            case_passed = route_passed and tool_calls_match and report.is_grounded and abstain_passed
            if case_passed:
                grounded_cases_count += 1

            # Update case record
            updated_case = dict(case_detail)
            updated_case["is_grounded"] = report.is_grounded
            updated_case["claims_count"] = report.total_claims
            updated_case["facts_count"] = report.facts_count
            updated_case["inferences_count"] = report.inferences_count
            updated_case["advice_count"] = report.advice_count
            updated_case["unsupported_count"] = report.unsupported_count
            updated_case["contradicted_count"] = report.contradicted_count
            updated_case["case_passed"] = case_passed
            results.append(updated_case)

        total_cases = len(results)
        route_accuracy = route_correct_count / total_cases if total_cases > 0 else 0.0
        overall_pass_rate = grounded_cases_count / total_cases if total_cases > 0 else 0.0
        structured_precision = structured_cases_passed / structured_cases_total if structured_cases_total > 0 else 0.0
        abstain_accuracy = abstain_cases_passed / abstain_cases_total if abstain_cases_total > 0 else 0.0
        ucr = total_unsupported / total_claims if total_claims > 0 else 0.0
        advice_rate = total_advice / total_claims if total_claims > 0 else 0.0

        sp_p90 = summary_0["latency_single_path"]["p90"]
        hy_p90 = summary_0["latency_hybrid"]["p90"]

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

        summary_rescored = {
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
            "latency_all": summary_0["latency_all"],
            "latency_single_path": summary_0["latency_single_path"],
            "latency_hybrid": summary_0["latency_hybrid"],
            "retrieval_chunk_recall": summary_0["retrieval_chunk_recall"],
            "retrieval_chunk_precision": summary_0["retrieval_chunk_precision"],
            "gate_passed": gate_passed,
            "case_details": results,
        }
        rescored_models[model_name] = summary_rescored

    # Save JSON results
    out_json = Path(f"docs/eval/{output_prefix}_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rescored_models, f, indent=2)

    # Generate Markdown Report
    report_lines = [
        f"# Model Grounding Baseline Replay Report ({output_prefix})",
        "",
        "## Summary Results Table (M7.1 Rescored Verifier)",
        "",
        "| Model Name | Pass Rate | Route Acc | Struct Prec | Abstain Acc | UCR | Contra | Single P90 | Hybrid P90 | Chunk Recall | Gate Passed |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for model_name, s in rescored_models.items():
        gate_str = "✅ PASS" if s["gate_passed"] else "❌ FAIL"
        report_lines.append(
            f"| `{model_name}` | {s['overall_pass_rate']:.1%} | {s['route_accuracy']:.1%} | "
            f"{s['structured_precision']:.1%} | {s['abstain_accuracy']:.1%} | {s['unsupported_claim_rate']:.1%} ({s['unsupported_count']}/{s['total_claims']}) | "
            f"{s['contradicted_count']} | {s['latency_single_path']['p90']}s | {s['latency_hybrid']['p90']}s | "
            f"{s['retrieval_chunk_recall']:.1%} | {gate_str} |"
        )

    report_lines.extend([
        "",
        "## Instrument Drift Comparison: M7.0 Baseline vs M7.1 Rescored",
        "",
        "| Model Name | Metric | M7.0 Baseline | M7.1 Rescored | Delta (Instrument Drift) |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for model_name in rescored_models.keys():
        s0 = m7_0_data[model_name]
        s1 = rescored_models[model_name]
        report_lines.extend([
            f"| `{model_name}` | Pass Rate | {s0['overall_pass_rate']:.1%} | {s1['overall_pass_rate']:.1%} | {s1['overall_pass_rate'] - s0['overall_pass_rate']:+.1%} |",
            f"| `{model_name}` | UCR | {s0['unsupported_claim_rate']:.1%} ({s0['unsupported_count']}/{s0['total_claims']}) | {s1['unsupported_claim_rate']:.1%} ({s1['unsupported_count']}/{s1['total_claims']}) | {s1['unsupported_claim_rate'] - s0['unsupported_claim_rate']:+.1%} |",
            f"| `{model_name}` | Contradictions | {s0['contradicted_count']} | {s1['contradicted_count']} | {s1['contradicted_count'] - s0['contradicted_count']:+d} |",
            f"| `{model_name}` | Abstain Accuracy | {s0['abstain_accuracy']:.1%} | {s1['abstain_accuracy']:.1%} | {s1['abstain_accuracy'] - s0['abstain_accuracy']:+.1%} |",
        ])

    out_md = Path(f"docs/eval/{output_prefix}_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    logger.info(f"Saved rescored replay JSON to {out_json} and report to {out_md}")
    return rescored_models


if __name__ == "__main__":
    replay_baseline(
        input_path=Path("docs/eval/m7_0_baseline_results.json"),
        output_prefix="m7_1_baseline_rescored",
    )
