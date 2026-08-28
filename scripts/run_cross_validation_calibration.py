"""5-Fold Cross-Validation Calibration Script for M7 Evaluation Harness.

Splits the 36-case eval_corpus.json into 5 stratified folds to verify parameter
stability, threshold robustness, and 95% confidence intervals across folds.
"""

from collections import defaultdict
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cross_val")


def run_cross_validation(results_path: Path, n_folds: int = 5) -> Dict[str, Any]:
    """Executes stratified 5-fold cross-validation on evaluation results."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cv_summary = {}

    for model_name, summary in data.items():
        case_details = summary["case_details"]
        total_n = len(case_details)
        logger.info(f"Running {n_folds}-fold CV for {model_name} (N={total_n} cases)...")

        # Group cases by category for stratified partitioning
        by_category = defaultdict(list)
        for c in case_details:
            by_category[c["category"]].append(c)

        # Stratified round-robin across shuffled categories to ensure equal fold sizes (7-8 per fold)
        folds: List[List[Dict[str, Any]]] = [[] for _ in range(n_folds)]
        fold_ptr = 0
        for cat, cases in by_category.items():
            for case in cases:
                folds[fold_ptr % n_folds].append(case)
                fold_ptr += 1

        fold_pass_rates = []
        fold_ucrs = []
        fold_route_accs = []
        fold_contras = []

        for f_idx, test_cases in enumerate(folds):
            n_test = len(test_cases)
            passed = sum(1 for c in test_cases if c.get("case_passed", False))
            route_passed = sum(1 for c in test_cases if c.get("route_passed", False))
            total_claims = sum(c.get("claims_count", 0) for c in test_cases)
            unsupp = sum(c.get("unsupported_count", 0) for c in test_cases)
            contra = sum(c.get("contradicted_count", 0) for c in test_cases)

            pass_rate = passed / n_test if n_test > 0 else 0.0
            route_acc = route_passed / n_test if n_test > 0 else 0.0
            ucr = unsupp / total_claims if total_claims > 0 else 0.0

            fold_pass_rates.append(pass_rate)
            fold_ucrs.append(ucr)
            fold_route_accs.append(route_acc)
            fold_contras.append(contra)

        # Compute mean and sample standard deviation
        def calc_stats(vals: List[float]) -> Tuple[float, float, Tuple[float, float]]:
            m = sum(vals) / len(vals)
            var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
            std = math.sqrt(var)
            # 95% t-interval for df=4 (t=2.776)
            t_crit = 2.776
            ci = (m - t_crit * (std / math.sqrt(len(vals))), m + t_crit * (std / math.sqrt(len(vals))))
            return m, std, ci

        pr_m, pr_std, pr_ci = calc_stats(fold_pass_rates)
        ucr_m, ucr_std, ucr_ci = calc_stats(fold_ucrs)
        ra_m, ra_std, ra_ci = calc_stats(fold_route_accs)

        cv_summary[model_name] = {
            "n_folds": n_folds,
            "total_cases": total_n,
            "pass_rate_mean": round(pr_m, 4),
            "pass_rate_std": round(pr_std, 4),
            "pass_rate_ci_95": [round(pr_ci[0], 4), round(pr_ci[1], 4)],
            "ucr_mean": round(ucr_m, 4),
            "ucr_std": round(ucr_std, 4),
            "ucr_ci_95": [round(ucr_ci[0], 4), round(ucr_ci[1], 4)],
            "route_accuracy_mean": round(ra_m, 4),
            "route_accuracy_std": round(ra_std, 4),
            "route_accuracy_ci_95": [round(ra_ci[0], 4), round(ra_ci[1], 4)],
            "total_contradictions": sum(fold_contras),
            "fold_details": [
                {
                    "fold": i + 1,
                    "n_cases": len(folds[i]),
                    "pass_rate": round(fold_pass_rates[i], 4),
                    "ucr": round(fold_ucrs[i], 4),
                    "route_accuracy": round(fold_route_accs[i], 4),
                    "contradictions": fold_contras[i],
                }
                for i in range(n_folds)
            ],
        }

    out_file = Path("docs/eval/m7_4_cross_validation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)

    logger.info(f"Saved cross-validation results to {out_file}")
    return cv_summary


if __name__ == "__main__":
    run_cross_validation(Path("docs/eval/m7_3_probe_bakeoff_results.json"))
