# ADR 0004 — Local Inference Runtime: Ollama

**Status:** Accepted

## Context

Options for running local LLM/embedding inference: Ollama, llama.cpp directly, or vLLM. This project needs programmatic API access from Python for a single user, on a single RTX 3080 Ti (12GB VRAM).

## Decision

Use **Ollama**.

## Rationale

vLLM's core advantage — 2.3x–20x higher throughput under concurrent load via PagedAttention and continuous batching — solves a multi-user serving problem this project doesn't have; it also demands real DevOps investment (CUDA/quantization format work, Linux-first support) disproportionate to a solo project already carrying a full new-territory RAG build. llama.cpp directly is the right move only once Ollama's abstractions are actually in the way (an unsupported model architecture, needing fine KV-cache control) — neither applies today, and since Ollama is built on llama.cpp, nothing is lost by starting here. Ollama's OpenAI-compatible API and native tool-calling support (confirmed for both Gemma 4 and Qwen-family models) directly serve this project's router design.

## Consequences

- Single-request-at-a-time serving model is fine — there is no concurrency to serve
- Native tool-calling is available for the router without hand-rolled text classification
- If a future need genuinely requires llama.cpp's lower-level control, migration cost is low since the underlying engine is shared