# Model Grounding Evaluation Report (m7_2_probe_granite)

## Summary Results Table

| Model Name | Pass Rate | Route Acc | Struct Prec | Abstain Acc | UCR | Contra | Single P90 | Hybrid P90 | Chunk Recall | Gate Passed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `granite4.1:8b` | 19.4% | 94.4% | 20.0% | 25.0% | 32.1% (88/274) | 16 | 21.71s | 23.39s | 47.8% | ❌ FAIL |

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
### Model: `granite4.1:8b`
- **Total Claims Extracted:** 274
- **Modality Breakdown:** Facts: 160 | Supported Inferences: 9 | Advice: 1 (0.4%)
- **Unsupported Claims:** 88 | **Contradicted Claims:** 16
- **Latency (Single-Path):** Mean: 13.39s | P90: 21.71s | Max: 33.67s
- **Latency (Hybrid):** Mean: 22.59s | P90: 23.39s | Max: 23.39s
- **Retrieval Chunk Recall / Precision:** 47.8% / 23.9%

#### Category Breakdown
| Category | Passed / Total | Claims | Facts | Infer | Adv | Unsupp | Contra |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `AMBIGUOUS_ENTITY` | 2/2 | 9 | 0 | 0 | 0 | 9 | 0 |
| `HYBRID_BOTH` | 0/4 | 65 | 32 | 0 | 0 | 25 | 8 |
| `NO_EVIDENCE` | 0/2 | 3 | 0 | 0 | 0 | 1 | 2 |
| `OUT_OF_SCOPE_DLC` | 0/2 | 6 | 2 | 0 | 0 | 4 | 0 |
| `STRUCTURED_VS_COMMUNITY_CONFLICT` | 1/2 | 6 | 5 | 0 | 0 | 0 | 1 |
| `SUPPORTED_INFERENCE` | 1/2 | 11 | 7 | 0 | 0 | 4 | 0 |
| `T1_FACT_LOOKUP` | 2/4 | 8 | 3 | 1 | 0 | 1 | 3 |
| `T2_COMPARISON` | 0/4 | 20 | 9 | 2 | 0 | 9 | 0 |
| `T3_PRODUCTION_CHAIN` | 0/4 | 23 | 16 | 0 | 0 | 7 | 0 |
| `T4_CATEGORY_LISTING` | 0/4 | 50 | 41 | 1 | 0 | 8 | 0 |
| `VECTOR_STRATEGY` | 1/6 | 73 | 45 | 5 | 1 | 20 | 2 |

