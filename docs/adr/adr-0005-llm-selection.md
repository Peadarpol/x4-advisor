# ADR 0005 — LLM Selection: Shortlist and Deferred Empirical Choice

**Status:** Accepted (shortlist and elimination), final selection deferred to SPEC-001 M6

## Context

The advisor's LLM handles both routing (tool-calling classification) and synthesis (grounded answer generation), on a 12GB VRAM budget. Candidates considered: Qwen3.6, Gemma 4 12B, Granite 4.1 8B, Qwen3 14B, Phi-4 14B, DeepSeek-R1 14B.

## Decision

**Eliminated:** Qwen3.6 — no variant below 27B exists, and 27B's real quantized footprint (~19GB at Q4_K_M) doesn't fit 12GB without compromises severe enough to defeat the purpose. Phi-4 14B — best-in-class MMLU for its size, but capped at 16K context, which disqualifies it for a RAG workload. DeepSeek-R1 14B — its distinguishing feature (visible chain-of-thought reasoning) adds latency this task's straightforward retrieval-then-synthesize workload doesn't need.

**Shortlisted:** Gemma 4 12B (provisional default), Granite 4.1 8B (purpose-built for tool-calling/RAG/classification without reasoning overhead), Qwen3 14B (mature control, field-evidence of successful local RAG use).

**Final selection is explicitly deferred** to empirical testing using SPEC-001's own retrieval-grounding evaluation harness (M6) — not decided from desk research or published benchmarks alone, since no published evidence directly measures the property that actually matters here (faithful, low-hallucination synthesis given retrieved context, plus reliable tool-calling for routing).

## Rationale

VRAM figures were independently verified against real GGUF file listings (not estimated), and the elimination/shortlist reasoning is defensible on that basis. But general capability benchmarks (MMLU, GPQA) don't measure the specific property this application needs, and no dataset was found directly comparing groundedness/routing-reliability across these specific candidates. Building the evaluation harness this project already needs (per the grounding contract, SPEC-001 §6) and running the shortlist through it is more informative than further desk research.

## Consequences

- Model choice isn't finalized until SPEC-001 M6 exists — code should be written model-agnostically (config-driven model name) rather than hardcoding an assumption
- The evaluation harness built for this decision is the same one used for the ongoing offline grounding gate — one piece of infrastructure serves both purposes