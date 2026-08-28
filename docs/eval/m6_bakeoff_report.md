# Milestone M6 Model Bake-Off & Grounding Evaluation Report

## Summary Results Table

| Model Name | Pass Rate | Route Acc | Struct Prec | Abstain Acc | UCR | Contradictions | Mean Latency | Gate Passed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gemma4:12b` | 91.7% | 97.2% | 95.0% | 75.0% | 7.9% | 1 | 13.4s | ❌ FAIL |
| `granite4.1:8b` | 52.8% | 77.8% | 70.0% | 100.0% | 13.5% | 4 | 10.42s | ❌ FAIL |
| `qwen3:14b` | 80.6% | 91.7% | 95.0% | 50.0% | 13.5% | 0 | 16.96s | ❌ FAIL |

## Architectural Evaluation Gates (SPEC-001 §11)
- **Structured Precision:** >= 90%
- **Unsupported Claim Rate (UCR):** <= 5%
- **Abstention Accuracy (DLC & No-Evidence):** 100%
- **Overall Ground Truth Pass Rate:** >= 85%
- **Zero Contradictions Invariant:** 0 contradicted claims across all cases

## Per-Model Breakdown
### Model: `gemma4:12b`
- **Total Claims Extracted:** 229
- **Facts:** 207 | **Supported Inferences:** 1 | **Advice:** 2
- **Unsupported Claims:** 18 | **Contradicted Claims:** 1
- **Latency (Mean / P90):** 13.4s / 24.32s

#### Category Breakdown
| Category | Passed / Total | Claims | Facts | Infer | Adv | Unsupp | Contra |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `AMBIGUOUS_ENTITY` | 2/2 | 9 | 0 | 0 | 0 | 9 | 0 |
| `HYBRID_BOTH` | 3/4 | 13 | 11 | 1 | 0 | 0 | 1 |
| `NO_EVIDENCE` | 1/2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `OUT_OF_SCOPE_DLC` | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `STRUCTURED_VS_COMMUNITY_CONFLICT` | 2/2 | 2 | 2 | 0 | 0 | 0 | 0 |
| `SUPPORTED_INFERENCE` | 2/2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `T1_FACT_LOOKUP` | 4/4 | 7 | 6 | 0 | 0 | 1 | 0 |
| `T2_COMPARISON` | 4/4 | 12 | 10 | 0 | 0 | 2 | 0 |
| `T3_PRODUCTION_CHAIN` | 4/4 | 12 | 11 | 0 | 0 | 1 | 0 |
| `T4_CATEGORY_LISTING` | 3/4 | 146 | 141 | 0 | 1 | 4 | 0 |
| `VECTOR_STRATEGY` | 6/6 | 27 | 25 | 0 | 1 | 1 | 0 |

### Model: `granite4.1:8b`
- **Total Claims Extracted:** 185
- **Facts:** 148 | **Supported Inferences:** 6 | **Advice:** 2
- **Unsupported Claims:** 25 | **Contradicted Claims:** 4
- **Latency (Mean / P90):** 10.42s / 20.46s

#### Category Breakdown
| Category | Passed / Total | Claims | Facts | Infer | Adv | Unsupp | Contra |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `AMBIGUOUS_ENTITY` | 1/2 | 4 | 0 | 0 | 0 | 4 | 0 |
| `HYBRID_BOTH` | 1/4 | 26 | 17 | 2 | 1 | 6 | 0 |
| `NO_EVIDENCE` | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `OUT_OF_SCOPE_DLC` | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `STRUCTURED_VS_COMMUNITY_CONFLICT` | 0/2 | 6 | 4 | 0 | 0 | 0 | 2 |
| `SUPPORTED_INFERENCE` | 0/2 | 4 | 3 | 0 | 0 | 0 | 1 |
| `T1_FACT_LOOKUP` | 2/4 | 4 | 2 | 1 | 0 | 0 | 1 |
| `T2_COMPARISON` | 1/4 | 2 | 1 | 1 | 0 | 0 | 0 |
| `T3_PRODUCTION_CHAIN` | 3/4 | 22 | 18 | 0 | 0 | 4 | 0 |
| `T4_CATEGORY_LISTING` | 3/4 | 96 | 90 | 1 | 0 | 5 | 0 |
| `VECTOR_STRATEGY` | 4/6 | 21 | 13 | 1 | 1 | 6 | 0 |

### Model: `qwen3:14b`
- **Total Claims Extracted:** 170
- **Facts:** 143 | **Supported Inferences:** 3 | **Advice:** 1
- **Unsupported Claims:** 23 | **Contradicted Claims:** 0
- **Latency (Mean / P90):** 16.96s / 24.51s

#### Category Breakdown
| Category | Passed / Total | Claims | Facts | Infer | Adv | Unsupp | Contra |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `AMBIGUOUS_ENTITY` | 2/2 | 9 | 0 | 0 | 0 | 9 | 0 |
| `HYBRID_BOTH` | 2/4 | 12 | 4 | 1 | 1 | 6 | 0 |
| `NO_EVIDENCE` | 0/2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `OUT_OF_SCOPE_DLC` | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `STRUCTURED_VS_COMMUNITY_CONFLICT` | 2/2 | 2 | 2 | 0 | 0 | 0 | 0 |
| `SUPPORTED_INFERENCE` | 2/2 | 3 | 3 | 0 | 0 | 0 | 0 |
| `T1_FACT_LOOKUP` | 3/4 | 4 | 2 | 1 | 0 | 1 | 0 |
| `T2_COMPARISON` | 4/4 | 7 | 4 | 1 | 0 | 2 | 0 |
| `T3_PRODUCTION_CHAIN` | 4/4 | 9 | 9 | 0 | 0 | 0 | 0 |
| `T4_CATEGORY_LISTING` | 3/4 | 104 | 102 | 0 | 0 | 2 | 0 |
| `VECTOR_STRATEGY` | 5/6 | 20 | 17 | 0 | 0 | 3 | 0 |

