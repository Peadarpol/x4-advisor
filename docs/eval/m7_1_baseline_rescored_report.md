# Model Grounding Baseline Replay Report (m7_1_baseline_rescored)

## Summary Results Table (M7.1 Rescored Verifier)

| Model Name | Pass Rate | Route Acc | Struct Prec | Abstain Acc | UCR | Contra | Single P90 | Hybrid P90 | Chunk Recall | Gate Passed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gemma4:12b` | 33.3% | 88.9% | 45.0% | 75.0% | 34.9% (96/275) | 1 | 33.55s | 31.94s | 56.5% | ❌ FAIL |
| `granite4.1:8b` | 22.2% | 77.8% | 20.0% | 100.0% | 40.7% (122/300) | 6 | 25.98s | 24.34s | 56.5% | ❌ FAIL |
| `qwen3:14b` | 33.3% | 91.7% | 50.0% | 100.0% | 32.0% (62/194) | 2 | 28.23s | 25.4s | 56.5% | ❌ FAIL |

## Instrument Drift Comparison: M7.0 Baseline vs M7.1 Rescored

| Model Name | Metric | M7.0 Baseline | M7.1 Rescored | Delta (Instrument Drift) |
| :--- | :--- | :--- | :--- | :--- |
| `gemma4:12b` | Pass Rate | 58.3% | 33.3% | -25.0% |
| `gemma4:12b` | UCR | 6.6% (18/275) | 34.9% (96/275) | +28.4% |
| `gemma4:12b` | Contradictions | 1 | 1 | +0 |
| `gemma4:12b` | Abstain Accuracy | 50.0% | 75.0% | +25.0% |
| `granite4.1:8b` | Pass Rate | 25.0% | 22.2% | -2.8% |
| `granite4.1:8b` | UCR | 15.0% (45/300) | 40.7% (122/300) | +25.7% |
| `granite4.1:8b` | Contradictions | 5 | 6 | +1 |
| `granite4.1:8b` | Abstain Accuracy | 50.0% | 100.0% | +50.0% |
| `qwen3:14b` | Pass Rate | 52.8% | 33.3% | -19.5% |
| `qwen3:14b` | UCR | 10.8% (21/194) | 32.0% (62/194) | +21.1% |
| `qwen3:14b` | Contradictions | 0 | 2 | +2 |
| `qwen3:14b` | Abstain Accuracy | 100.0% | 100.0% | +0.0% |
