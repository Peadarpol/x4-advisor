# AGENTS.md

## Specifications Lifecycle
- All specs live under docs/planning/specs/.
- Lifecycle status flow: DRAFT → APPROVED → DELIVERED.
- Follows the same governance conventions established in i-delivery-control.

## Core Governance & Safety Rules
- **Self-approval rule**: An agent cannot mark its own spec APPROVED or DELIVERED.
- **Adversarial review**: Required before any merge to main.

## Required Phase 1 Deliverables (TODO)
- **Retrieval-grounding check**: An automated test verifying that anything the synthesizer claims is "grounded" traces back to real retrieved content, not fabricated.
  - *Note*: This test does not exist yet. Nothing should be treated as fully gated until it does.

## Dependency Notes
- No traceability-check script or self-approval-enforcement script exist yet.
- Both scripts depend on SPEC-001 existing first. Do not stub these prior to SPEC-001.
