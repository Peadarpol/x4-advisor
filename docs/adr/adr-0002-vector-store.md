# ADR 0002 — Vector Store: sqlite-vec

**Status:** Accepted

## Context

The unstructured knowledge layer needs vector similarity search. Options considered: a dedicated vector database server (Chroma), an embedded SQLite extension (sqlite-vec), or a cloud-managed vector store.

## Decision

Use **sqlite-vec**, in the same SQLite file as the structured relational tables.

## Rationale

For a single-operator, single-corpus project, an embedded store wins on every dimension that costs time: no server process to run, and "deploying" a new index is copying one file. A dedicated server like Chroma pays off once multiple collections or complex metadata filtering across many corpora are needed — not the case here. sqlite-vec also supports metadata columns and filtering directly in KNN queries, which is what enforces the base-game-only scope at the vector layer (not just at extraction time).

## Consequences

- **sqlite-vec is pre-1.0** and its own documentation warns of possible breaking changes. The version in use must be pinned explicitly (in `pyproject.toml`), with an upgrade procedure documented before bumping it — not left floating on "latest."
- One database file serves both structured and vector data — simpler operationally, but means the whole file is the unit of backup/versioning, not two independent systems