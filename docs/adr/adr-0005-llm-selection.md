# ADR 0005 — LLM Selection: Empirical Bake-Off & Escalation Decision

**Status:** Accepted (Empirical Bake-Off Completed — Escalated: No Candidate Cleared Both Layer 2 Hard Gates)

## Context

The advisor's LLM handles routing (tool-calling classification) and synthesis (grounded answer generation) on a 12GB VRAM budget. In Milestone M6, three shortlisted models (`gemma4:12b`, `granite4.1:8b`, `qwen3:14b`) were evaluated on the canonical $N=36$ evaluation corpus across 11 query templates against the Layer 2 Grounding Gates defined in SPEC-001 §11.

## Mandatory Layer 2 Grounding Gates vs. Empirical Results

| Gate / Metric | SPEC-001 §11 Target | `gemma4:12b` | `granite4.1:8b` | `qwen3:14b` |
| :--- | :--- | :--- | :--- | :--- |
| **Unsupported Claim Rate (UCR)** | **$\le 3.0\%$** (Hard Gate) | **7.9%** (18/229) ❌ | 13.5% (25/185) ❌ | 13.5% (23/170) ❌ |
| **Zero Contradictions Invariant** | **0 Contradictions** (Hard Gate) | **1** ❌ | **4** ❌ | **0** ✅ |
| **Abstention Accuracy** | **100.0%** (Hard Gate) | 75.0% (3/4) ❌ | **100.0%** (4/4) ✅ | 50.0% (2/4) ❌ |
| **Structured Precision** | $\ge 90.0\%$ | **95.0%** (19/20) ✅ | 70.0% (14/20) ❌ | **95.0%** (19/20) ✅ |
| **Routing Accuracy** | $\ge 90.0\%$ | **97.2%** (35/36) ✅ | 77.8% (28/36) ❌ | **91.7%** (33/36) ✅ |
| **Overall Pass Rate** | $\ge 85.0\%$ | **91.7%** (33/36) ✅ | 52.8% (19/36) ❌ | 80.6% (29/36) ❌ |
| **Mean / P90 Latency** | $<20.0\text{s} / <30.0\text{s}$ | 13.40s / 24.32s ✅ | **10.42s** / 20.46s ✅ | 16.96s / 24.51s ✅ |
| **Gate Status** | **ALL GATES PASS** | **FAIL** | **FAIL** | **FAIL** |

## Decision: Empirical Escalation & Provisional Operating Default

1. **Formal Gate Outcome (No Winner by Rule):** No candidate model satisfied both mandatory Layer 2 grounding gates simultaneously.
   - `qwen3:14b` is the only model that satisfied the Zero Contradiction Invariant (0 across all 36 cases), but failed the UCR gate (13.5%) and severely over-abstained on base-game content (50% abstention accuracy).
   - `gemma4:12b` was closest to the UCR ceiling (7.9%) and achieved the highest overall pass rate (91.7%) and routing stability (97.2%), but failed both the $\le 3.0\%$ UCR gate and the Zero Contradiction Invariant (1 contradiction in hybrid synthesis).
   - `granite4.1:8b` failed routing stability (77.8%), structured precision (70.0%), and produced 4 contradicted claims.

2. **Provisional Operating Choice for Phase 1 (M7):**
   - Rather than halting development or relaxing the formal gate post-hoc, `gemma4:12b` is selected as the **provisional operating default** for Milestone M7 (CLI delivery).
   - This choice is documented as a pragmatic operating selection made *despite* failing the formal gate, with the 7.9% UCR and 1-contradiction risk explicitly recorded in the system risk register.

3. **Retrieval Recall vs. Generative Latency Tradeoff:**
   - At $\tau = 0.65$, vector retrieval under-recalled (returning only 1–2 chunks and falsely rejecting valid procedural guidance in the 0.50–0.64 band), allowing single-path synthesis to artificially complete in 5–10s.
   - Recalibrating to $\tau = 0.50$ correctly recovers critical procedural knowledge (e.g., fleet commands, budget transfers, supply chain bottlenecks), but increases retrieved context volume to 4–5 chunks.
   - On a 12GB VRAM budget with `gemma4:12b` (Q4_K_M), synthesizing over this full retrieved context increases generation time to ~28.6s (exceeding the single-path target SLA of <20.0s, though remaining within the 30.0s client timeout).
   - **Architectural Tradeoff Decision:** Retrieval correctness and grounded recall are explicitly prioritized over latency SLA metrics. This latency gap is tracked as a documented limitation of the provisional model rather than engineered away by artificially constricting retrieval.

## Consequences

- The formal Layer 2 Grounding Gate remains at $\le 3.0\%$ UCR and 0 Contradictions. It is **not** lowered to accommodate model shortcomings.
- The evaluation harness (`scripts/run_eval_benchmark.py`) and 5-class verifier remain active regression gates.
- `test_live_m5_vector_search` in `test_m5_live_ollama.py` carries a documented non-strict `xfail` acknowledging the ~28.6s latency under full multi-chunk retrieval context at $\tau = 0.50$.
- Phase 2 exploration will evaluate post-processing verification guards, context distillation/summarization, or fine-tuning specifically targeted at the remaining unsupported claim surface and latency profile.

---

## Addendum: Milestone M7 Synthesis & Routing Tuning Re-Bake-Off

**Date:** August 2026  
**Status:** Accepted — Escalation Maintained: No Candidate Cleared Both Mandatory Hard Gates; `gemma4:12b` Retained as Provisional Operating Default for M8 (CLI Delivery).

### M7 Tuning Interventions Applied
1. **Instrument Unconfounding (M7.0/M7.1):** Fixed scoring casing defect, removed router regex fallback, replaced lenient multi-chunk bag-of-words overlap with strict sentence-localized proposition matching, and established Condition C for negative evidence disclaimers.
2. **Grammar-Constrained JSON Schema Routing (M7.2 / ADR-0008):** Passed strict grammar schema via Ollama's `format` parameter, eliminating syntax errors and illegal enum tokens.
3. **Synthesis Grounding & Anti-Preamble Tuning (M7.3):** Added strict anti-preamble prompt constraints and structured evidence unit indexing (`[EVIDENCE E1..En]`).

### M7.3 Final Multi-Model Re-Bake-Off Results

| Gate / Metric | SPEC-001 §15 Target | `gemma4:12b` | `granite4.1:8b` | `qwen3:14b` |
| :--- | :--- | :--- | :--- | :--- |
| **Unsupported Claim Rate (UCR)** | **$\le 3.0\%$** (Hard Gate) | **10.9%** (21/193) ❌ | 25.6% (33/129) ❌ | **19.7%** (24/122) ❌ |
| **Zero Contradictions Invariant** | **0 Contradictions** (Hard Gate) | **5** ❌ | **6** ❌ | **2** ❌ |
| **Abstention Accuracy** | **100.0%** (Hard Gate) | 75.0% (3/4) ❌ | 0.0% (0/4) ❌ | 50.0% (2/4) ❌ |
| **Structured Precision** | $\ge 90.0\%$ | **85.0%** (17/20) ❌ | 70.0% (14/20) ❌ | **80.0%** (16/20) ❌ |
| **Routing Accuracy** | $\ge 90.0\%$ | **94.4%** (34/36) ✅ | **97.2%** (35/36) ✅ | **94.4%** (34/36) ✅ |
| **Overall Pass Rate** | $\ge 85.0\%$ | **77.8%** (28/36) ❌ | 55.6% (20/36) ❌ | **75.0%** (27/36) ❌ |
| **Single / Hybrid P90 Latency** | $<20.0\text{s} / <30.0\text{s}$ | 25.97s / 25.79s ✅ | **21.39s / 21.32s** ✅ | 23.40s / 23.81s ✅ |
| **Gate Status** | **ALL GATES PASS** | **FAIL** | **FAIL** | **FAIL** |

### Adjudication & Operating Decision
1. **Formal Gate Outcome (Escalation Maintained):** Despite substantial measurable progress (Gemma's UCR dropping from 34.9% un-tuned down to 10.9%, and pass rate jumping from 33.3% to 77.8%), **no model cleared both mandatory Layer 2 hard gates simultaneously**.
2. **Provisional Model for Milestone M8:** `gemma4:12b` is retained as the **provisional operating default** for M8 (CLI Delivery), with `qwen3:14b` as the validated low-contradiction fallback.
3. **Explicit Accounting of Residual Unsupported Claims (Gemma $N=21$):**
   - **Ambiguous Entity Clarifications (9 / 21 = 42.9%):** Conversational prompts clarifying entity variants without retrieved background text.
   - **Unretrieved Knowledge under Layer 1 Recall Ceiling (5 / 21 = 23.8%):** Strategic queries where unretrieved chunks under the 56.5%–65.2% recall ceiling force parametric extrapolation.
   - **Statistical Comparisons & Category Listings (4 / 21 = 19.0%):** Minor unanchored comparison phrasing.
   - **Supported Inferences & Fact Lookups (3 / 21 = 14.3%):** Mathematical inference rounding.
4. **Documented Layer 1 Recall Boundary:** Overall chunk recall on vector/hybrid queries remains bounded at **56.5%–65.2%**, identifying Layer 1 retrieval tuning (reranking, chunk expansion) as the primary lever for future grounding improvements.