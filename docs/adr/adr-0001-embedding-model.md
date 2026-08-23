# ADR 0001 — Embedding Model: Qwen3-Embedding-0.6B

**Status:** Accepted

## Context

The hybrid retrieval architecture needs an embedding model for the unstructured knowledge layer — embedding curated content at ingestion time and embedding user queries at retrieval time. The project already runs LLM inference through Ollama with Qwen-family models under consideration, so ecosystem consistency was a factor alongside raw embedding quality.

## Decision

Use **Qwen3-Embedding-0.6B**, served via Ollama, rather than a generic option like `nomic-embed-text`.

## Rationale

Comparisons of local embedding options showed Qwen3-Embedding-0.6B offering strong quality-per-VRAM (competitive MTEB scores at a small footprint), Apache-2.0 licensed, and Ollama-native. Given the project already standardizes on Qwen-family tooling elsewhere and the corpus (base-game ship/ware/strategy text) doesn't need multilingual or hybrid-search capability that a heavier option like BGE-M3 would offer, there was no reason to introduce a second model family purely for embeddings.

## Consequences

- One less dependency to manage — embeddings and (potentially) generation share the Qwen ecosystem
- Re-embedding the entire corpus is required if this choice changes later (embeddings from different models aren't interchangeable) — a cost worth knowing about before the corpus grows large