# Model Grounding Evaluation Report (m7_1_probe_gemma_minted)

## Summary Results Table

| Model Name | Pass Rate | Route Acc | Struct Prec | Abstain Acc | UCR | Contra | Single P90 | Hybrid P90 | Chunk Recall | Gate Passed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gemma4:12b` | 52.8% | 97.2% | 60.0% | 50.0% | 8.8% (25/284) | 4 | 25.41s | 26.7s | 56.5% | ❌ FAIL |

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
- **Total Claims Extracted:** 284
- **Modality Breakdown:** Facts: 244 | Supported Inferences: 4 | Advice: 7 (2.5%)
- **Unsupported Claims:** 25 | **Contradicted Claims:** 4
- **Latency (Single-Path):** Mean: 14.91s | P90: 25.41s | Max: 26.95s
- **Latency (Hybrid):** Mean: 25.18s | P90: 26.7s | Max: 26.7s
- **Retrieval Chunk Recall / Precision:** 56.5% / 26.0%

#### Category Breakdown
| Category | Passed / Total | Claims | Facts | Infer | Adv | Unsupp | Contra |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `AMBIGUOUS_ENTITY` | 2/2 | 9 | 0 | 0 | 0 | 9 | 0 |
| `HYBRID_BOTH` | 0/4 | 32 | 21 | 2 | 3 | 3 | 3 |
| `NO_EVIDENCE` | 0/2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `OUT_OF_SCOPE_DLC` | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `STRUCTURED_VS_COMMUNITY_CONFLICT` | 2/2 | 2 | 2 | 0 | 0 | 0 | 0 |
| `SUPPORTED_INFERENCE` | 1/2 | 1 | 1 | 0 | 0 | 0 | 0 |
| `T1_FACT_LOOKUP` | 4/4 | 7 | 6 | 0 | 0 | 1 | 0 |
| `T2_COMPARISON` | 1/4 | 10 | 8 | 0 | 0 | 2 | 0 |
| `T3_PRODUCTION_CHAIN` | 1/4 | 12 | 11 | 0 | 0 | 1 | 0 |
| `T4_CATEGORY_LISTING` | 3/4 | 142 | 140 | 0 | 0 | 2 | 0 |
| `VECTOR_STRATEGY` | 3/6 | 68 | 54 | 2 | 4 | 7 | 1 |

