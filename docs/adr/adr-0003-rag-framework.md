# ADR 0003 — RAG Framework: Custom, Not LangChain/LlamaIndex

**Status:** Accepted

## Context

RAG applications commonly reach for a framework (LangChain, LlamaIndex) to handle chunking, embedding orchestration, and retrieval. This project's stated purpose includes learning RAG mechanics directly, not just producing a working advisor.

## Decision

Build the retrieval pipeline (chunking, embedding calls, the router, structured/vector query dispatch) in plain Python, without a RAG orchestration framework.

## Rationale

Frameworks earn their cost at team scale or high query volume, where their abstractions save real coordination effort. For a single-user, single-corpus project, they mostly hide the exact mechanics this project exists to understand — using one would work against the learning objective, not just be unnecessary overhead. A custom pipeline is also easier to debug and reason about at this scale: every step is visible Python code, not a framework's internal behavior.

## Consequences

- More code to write directly (chunking logic, retrieval routing) that a framework would otherwise provide
- Full visibility and control over every step — directly serves the project's learning purpose
- If the project's needs ever outgrow this (unlikely at the stated scale), adopting a framework later is possible without having built anything framework-specific to unwind