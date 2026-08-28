# Model Grounding Evaluation Report (m7_0_baseline)

## Summary Results Table

| Model Name | Pass Rate | Route Acc | Struct Prec | Abstain Acc | UCR | Contra | Single P90 | Hybrid P90 | Chunk Recall | Gate Passed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gemma4:12b` | 58.3% | 88.9% | 58.8% | 50.0% | 6.6% (18/275) | 1 | 33.55s | 31.94s | 56.5% | ❌ FAIL |
| `granite4.1:8b` | 25.0% | 77.8% | 20.0% | 50.0% | 15.0% (45/300) | 5 | 25.98s | 24.34s | 56.5% | ❌ FAIL |
| `qwen3:14b` | 52.8% | 91.7% | 50.0% | 100.0% | 10.8% (21/194) | 0 | 28.23s | 25.4s | 56.5% | ❌ FAIL |

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
- **Total Claims Extracted:** 275
- **Modality Breakdown:** Facts: 246 | Supported Inferences: 4 | Advice: 6 (2.2%)
- **Unsupported Claims:** 18 | **Contradicted Claims:** 1
- **Latency (Single-Path):** Mean: 22.49s | P90: 33.55s | Max: 40.41s
- **Latency (Hybrid):** Mean: 31.32s | P90: 31.94s | Max: 31.94s
- **Retrieval Chunk Recall / Precision:** 56.5% / 26.0%

#### Category Breakdown
| Category | Passed / Total | Claims | Facts | Infer | Adv | Unsupp | Contra |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `AMBIGUOUS_ENTITY` | 2/2 | 9 | 0 | 0 | 0 | 9 | 0 |
| `HYBRID_BOTH` | 1/4 | 30 | 23 | 1 | 3 | 2 | 1 |
| `NO_EVIDENCE` | 0/2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `OUT_OF_SCOPE_DLC` | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `STRUCTURED_VS_COMMUNITY_CONFLICT` | 1/2 | 3 | 1 | 0 | 0 | 2 | 0 |
| `SUPPORTED_INFERENCE` | 1/2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `T1_FACT_LOOKUP` | 2/4 | 2 | 2 | 0 | 0 | 0 | 0 |
| `T2_COMPARISON` | 1/4 | 11 | 9 | 1 | 0 | 1 | 0 |
| `T3_PRODUCTION_CHAIN` | 1/4 | 12 | 11 | 0 | 0 | 1 | 0 |
| `T4_CATEGORY_LISTING` | 4/4 | 142 | 141 | 0 | 1 | 0 | 0 |
| `VECTOR_STRATEGY` | 6/6 | 64 | 57 | 2 | 2 | 3 | 0 |

### Model: `granite4.1:8b`
- **Total Claims Extracted:** 300
- **Modality Breakdown:** Facts: 240 | Supported Inferences: 7 | Advice: 3 (1.0%)
- **Unsupported Claims:** 45 | **Contradicted Claims:** 5
- **Latency (Single-Path):** Mean: 16.92s | P90: 25.98s | Max: 29.4s
- **Latency (Hybrid):** Mean: 19.66s | P90: 24.34s | Max: 24.34s
- **Retrieval Chunk Recall / Precision:** 56.5% / 29.5%

#### Category Breakdown
| Category | Passed / Total | Claims | Facts | Infer | Adv | Unsupp | Contra |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `AMBIGUOUS_ENTITY` | 1/2 | 4 | 0 | 0 | 0 | 4 | 0 |
| `HYBRID_BOTH` | 0/4 | 64 | 45 | 1 | 1 | 16 | 1 |
| `NO_EVIDENCE` | 0/2 | 6 | 2 | 2 | 0 | 2 | 0 |
| `OUT_OF_SCOPE_DLC` | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `STRUCTURED_VS_COMMUNITY_CONFLICT` | 0/2 | 6 | 4 | 0 | 0 | 0 | 2 |
| `SUPPORTED_INFERENCE` | 0/2 | 4 | 2 | 0 | 0 | 1 | 1 |
| `T1_FACT_LOOKUP` | 0/4 | 4 | 2 | 1 | 0 | 0 | 1 |
| `T2_COMPARISON` | 0/4 | 2 | 1 | 1 | 0 | 0 | 0 |
| `T3_PRODUCTION_CHAIN` | 2/4 | 21 | 17 | 1 | 0 | 3 | 0 |
| `T4_CATEGORY_LISTING` | 2/4 | 121 | 116 | 0 | 0 | 5 | 0 |
| `VECTOR_STRATEGY` | 2/6 | 68 | 51 | 1 | 2 | 14 | 0 |

### Model: `qwen3:14b`
- **Total Claims Extracted:** 194
- **Modality Breakdown:** Facts: 162 | Supported Inferences: 5 | Advice: 6 (3.1%)
- **Unsupported Claims:** 21 | **Contradicted Claims:** 0
- **Latency (Single-Path):** Mean: 16.08s | P90: 28.23s | Max: 35.65s
- **Latency (Hybrid):** Mean: 24.47s | P90: 25.4s | Max: 25.4s
- **Retrieval Chunk Recall / Precision:** 56.5% / 26.0%

#### Category Breakdown
| Category | Passed / Total | Claims | Facts | Infer | Adv | Unsupp | Contra |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `AMBIGUOUS_ENTITY` | 2/2 | 9 | 0 | 0 | 0 | 9 | 0 |
| `HYBRID_BOTH` | 1/4 | 16 | 8 | 1 | 4 | 3 | 0 |
| `NO_EVIDENCE` | 0/2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `OUT_OF_SCOPE_DLC` | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `STRUCTURED_VS_COMMUNITY_CONFLICT` | 2/2 | 2 | 2 | 0 | 0 | 0 | 0 |
| `SUPPORTED_INFERENCE` | 1/2 | 3 | 3 | 0 | 0 | 0 | 0 |
| `T1_FACT_LOOKUP` | 2/4 | 4 | 2 | 1 | 0 | 1 | 0 |
| `T2_COMPARISON` | 0/4 | 7 | 4 | 1 | 0 | 2 | 0 |
| `T3_PRODUCTION_CHAIN` | 2/4 | 9 | 9 | 0 | 0 | 0 | 0 |
| `T4_CATEGORY_LISTING` | 3/4 | 104 | 102 | 0 | 0 | 2 | 0 |
| `VECTOR_STRATEGY` | 4/6 | 40 | 32 | 2 | 2 | 4 | 0 |

