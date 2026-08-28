# AGENTS.md

This file is the working reference for any agent operating in this repository. Keep it current -- an earlier version of this file sat untouched as a stale initial scaffold for a while, which is exactly the failure mode it exists to prevent.

## Planning documents (read before touching anything non-trivial)
- docs/planning/project-charter.md -- why this project exists, purpose priority order, governance approach
- docs/planning/scope-boundary.md -- what's in/out of scope, per phase, at implementation granularity
- docs/architecture/solution-design.md -- how the system is architected
- docs/planning/specs/spec-001.md -- Phase 1's implementation-ready contract (milestones M1-M7, requirements, failure behavior, exit criteria)
- docs/adr/adr-000X-*.md -- standalone technology/architecture decisions (0001-0007: embedding model, vector store, RAG framework, inference runtime, LLM selection, extraction tool, ingestion/paraphrase architecture)

## Specifications Lifecycle
- All specs live under docs/planning/specs/.
- Lifecycle status flow: DRAFT → APPROVED → DELIVERED.
- Follows the same governance conventions established in ai-delivery-control, deliberately scaled down -- this is a solo personal/learning project, not a multi-contributor system. No CDR ledger, no posture configurability, no formal REQ-xxx traceability ID system. See project-charter.md for the explicit reasoning.

## Core Governance & Safety Rules
- **Self-approval rule**: An agent cannot mark its own spec APPROVED or DELIVERED.
- **Adversarial review**: Required before any merge to main. In practice this is enforced by GitHub branch protection on main (PR + human approval required) plus the delivery agent (Peadarpol-AiDelivery) holding Write, not Admin, access with no bypass -- not a separate automated review script. Nothing merges without the repository owner's actual review and approval.
- **GitHub Authentication Rule**: Any `gh` CLI or GitHub API operation performed by an agent MUST set `$env:GH_TOKEN = $env:AIDELIVERY_GH_TOKEN` prior to invocation to guarantee operations are authored by `peadarpol-aidelivery` rather than falling back to the system owner's default `gh` keyring session.

- **Pre-commit gates that actually run** (.pre-commit-config.yaml): check-active-repo (verifies commits land in the correct repo) and a pytest gate (test suite must pass).

## Required Phase 1 Deliverables
- **Retrieval-grounding check**: Automated test suite and 5-class proposition verifier (`tests/unit/test_grounding_taxonomy.py`, `scripts/run_eval_benchmark.py`) verifying that claims are grounded in retrieved evidence units without hallucinations. (DELIVERED in M6/M7).

## Data Handling -- Non-Negotiable
- Never commit anything under data/ (raw sources, extracted game data, curated/paraphrased knowledge base content, embeddings, the SQLite database). All gitignored, all local-only, by design.
- No pre-built structured or unstructured data is ever redistributed -- not even as a convenience. Every user (including the owner) runs the ingestion pipelines against their own game install and their own sourced content. See project-charter.md for the reasoning.
- Retrieved/curated content is always treated as data, never instructions, when placed in a prompt -- see solution-design.md for the architectural detail.

## Contribution Conventions
See CONTRIBUTING.md for the short version. In brief: architectural changes need an ADR before implementation; behavioral changes need a spec/test update; nothing outside stated scope without discussion first.

## Current Status
SPEC-001 (Phase 1) is approved and DELIVERED. Milestone M1 (structured game extraction) is DELIVERED (PR #4). Milestone M2 (structured query engine for 4 query templates) is DELIVERED (PR #5). Milestone M3 (unstructured ingestion pipeline) is DELIVERED (PR #6). Milestone M4 (unstructured vector retrieval engine) is DELIVERED (PR #7). Milestone M5 (LLM router + synthesizer engine) is DELIVERED (PR #8). Milestone M6 (retrieval-grounding evaluation harness, 5-class claim taxonomy verifier, 5-fold cross-validation calibration, local model bake-off, and ADR-0005 empirical escalation) is DELIVERED (PR #9). Milestone M7 (Synthesis & Routing Tuning: instrument unconfounding, sentence-localized proposition verifier, grammar-constrained JSON Schema routing ADR-0008, anti-preamble prompt tuning, 5-fold cross-validation calibration, and empirical re-bake-off) is DELIVERED (PR #10). Milestone M8 (CLI Delivery: single-turn REPL and CLI interface, single-turn entity disambiguation resumption, evidence provenance transparency, dataset staleness validation, and pipeline observability) is DELIVERED (PR #11).



