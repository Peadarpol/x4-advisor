# Model Grounding Evaluation Report (m7_3_probe_bakeoff)

## Summary Results Table

| Model Name | Pass Rate | Route Acc | Struct Prec | Abstain Acc | UCR | Contra | Single P90 | Hybrid P90 | Chunk Recall | Gate Passed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gemma4:12b` | 77.8% | 94.4% | 85.0% | 75.0% | 10.9% (21/193) | 5 | 25.97s | 25.79s | 65.2% | ❌ FAIL |
| `granite4.1:8b` | 55.6% | 97.2% | 70.0% | 0.0% | 25.6% (33/129) | 6 | 21.39s | 21.32s | 52.2% | ❌ FAIL |
| `qwen3:14b` | 75.0% | 94.4% | 80.0% | 50.0% | 19.7% (24/122) | 2 | 23.4s | 23.81s | 56.5% | ❌ FAIL |

## Layer 2 Qualification Gate Contract (SPEC-001 §15)
- **Unsupported Claim Rate (UCR):** <= 3.0% (Mandatory Hard Gate)
- **Zero Contradictions Invariant:** 0 contradicted claims (Mandatory Hard Gate)
- **Abstention Accuracy:** 100.0% (Mandatory Hard Gate; N=4 total, N=2 DLC)
- **Structured Precision:** >= 90.0% (Quality Gate)
- **Routing Accuracy:** >= 90.0% (Quality Gate)
- **Overall Ground Truth Pass Rate:** >= 85.0% (Quality Gate)
- **Latency SLAs:** Single-path P90 < 20.0s, Hybrid P90 < 30.0s
- **Modality Distribution:** ADVICE <= 15% of total claims overall

## Per-Model Breakdown
### Model: `gemma4:12b`
- **Total Claims Extracted:** 193
- **Modality Breakdown:** Facts: 158 | Supported Inferences: 2 | Advice: 7 (3.6%)
- **Unsupported Claims:** 21 | **Contradicted Claims:** 5
- **Latency (Single-Path):** Mean: 16.14s | P90: 25.97s | Max: 27.69s
- **Latency (Hybrid):** Mean: 24.48s | P90: 25.79s | Max: 25.79s
- **Retrieval Chunk Recall / Precision:** 65.2% / 30.0%

#### Category Breakdown
| Category | Passed / Total | Claims | Facts | Infer | Adv | Unsupp | Contra |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `AMBIGUOUS_ENTITY` | 2/2 | 9 | 0 | 0 | 0 | 9 | 0 |
| `HYBRID_BOTH` | 2/4 | 23 | 17 | 1 | 3 | 0 | 2 |
| `NO_EVIDENCE` | 0/2 | 1 | 0 | 0 | 0 | 0 | 1 |
| `OUT_OF_SCOPE_DLC` | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `STRUCTURED_VS_COMMUNITY_CONFLICT` | 1/2 | 2 | 2 | 0 | 0 | 0 | 0 |
| `SUPPORTED_INFERENCE` | 2/2 | 3 | 2 | 0 | 0 | 1 | 0 |
| `T1_FACT_LOOKUP` | 4/4 | 7 | 6 | 0 | 0 | 1 | 0 |
| `T2_COMPARISON` | 2/4 | 10 | 7 | 0 | 0 | 2 | 1 |
| `T3_PRODUCTION_CHAIN` | 4/4 | 14 | 13 | 0 | 0 | 1 | 0 |
| `T4_CATEGORY_LISTING` | 4/4 | 73 | 71 | 0 | 0 | 2 | 0 |
| `VECTOR_STRATEGY` | 5/6 | 51 | 40 | 1 | 4 | 5 | 1 |

### Model: `granite4.1:8b`
- **Total Claims Extracted:** 129
- **Modality Breakdown:** Facts: 80 | Supported Inferences: 7 | Advice: 3 (2.3%)
- **Unsupported Claims:** 33 | **Contradicted Claims:** 6
- **Latency (Single-Path):** Mean: 12.63s | P90: 21.39s | Max: 22.76s
- **Latency (Hybrid):** Mean: 20.94s | P90: 21.32s | Max: 21.32s
- **Retrieval Chunk Recall / Precision:** 52.2% / 24.5%

#### Category Breakdown
| Category | Passed / Total | Claims | Facts | Infer | Adv | Unsupp | Contra |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `AMBIGUOUS_ENTITY` | 2/2 | 9 | 0 | 0 | 0 | 9 | 0 |
| `HYBRID_BOTH` | 2/4 | 24 | 15 | 1 | 1 | 6 | 1 |
| `NO_EVIDENCE` | 0/2 | 2 | 0 | 1 | 0 | 0 | 1 |
| `OUT_OF_SCOPE_DLC` | 0/2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `STRUCTURED_VS_COMMUNITY_CONFLICT` | 2/2 | 3 | 3 | 0 | 0 | 0 | 0 |
| `SUPPORTED_INFERENCE` | 1/2 | 5 | 3 | 0 | 0 | 1 | 1 |
| `T1_FACT_LOOKUP` | 4/4 | 7 | 7 | 0 | 0 | 0 | 0 |
| `T2_COMPARISON` | 1/4 | 7 | 3 | 2 | 0 | 1 | 1 |
| `T3_PRODUCTION_CHAIN` | 3/4 | 20 | 16 | 0 | 0 | 4 | 0 |
| `T4_CATEGORY_LISTING` | 3/4 | 5 | 0 | 0 | 0 | 5 | 0 |
| `VECTOR_STRATEGY` | 2/6 | 47 | 33 | 3 | 2 | 7 | 2 |

### Model: `qwen3:14b`
- **Total Claims Extracted:** 122
- **Modality Breakdown:** Facts: 88 | Supported Inferences: 3 | Advice: 5 (4.1%)
- **Unsupported Claims:** 24 | **Contradicted Claims:** 2
- **Latency (Single-Path):** Mean: 15.61s | P90: 23.4s | Max: 34.02s
- **Latency (Hybrid):** Mean: 23.04s | P90: 23.81s | Max: 23.81s
- **Retrieval Chunk Recall / Precision:** 56.5% / 26.0%

#### Category Breakdown
| Category | Passed / Total | Claims | Facts | Infer | Adv | Unsupp | Contra |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `AMBIGUOUS_ENTITY` | 2/2 | 9 | 0 | 0 | 0 | 9 | 0 |
| `HYBRID_BOTH` | 3/4 | 13 | 6 | 1 | 3 | 2 | 1 |
| `NO_EVIDENCE` | 0/2 | 4 | 0 | 0 | 0 | 4 | 0 |
| `OUT_OF_SCOPE_DLC` | 1/2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `STRUCTURED_VS_COMMUNITY_CONFLICT` | 2/2 | 2 | 2 | 0 | 0 | 0 | 0 |
| `SUPPORTED_INFERENCE` | 2/2 | 3 | 2 | 0 | 0 | 1 | 0 |
| `T1_FACT_LOOKUP` | 4/4 | 4 | 4 | 0 | 0 | 0 | 0 |
| `T2_COMPARISON` | 1/4 | 5 | 2 | 1 | 0 | 1 | 1 |
| `T3_PRODUCTION_CHAIN` | 4/4 | 9 | 8 | 0 | 0 | 1 | 0 |
| `T4_CATEGORY_LISTING` | 3/4 | 52 | 49 | 0 | 0 | 3 | 0 |
| `VECTOR_STRATEGY` | 5/6 | 21 | 15 | 1 | 2 | 3 | 0 |

