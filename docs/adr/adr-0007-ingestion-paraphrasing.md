# ADR 0007 — Ingestion Paraphrase Architecture: Claim-First, Verify-Second

**Status:** Accepted

## Context

Curated wiki/community content is paraphrased by an LLM before entering the knowledge base (never stored verbatim, for copyright reasons). The original design was "LLM paraphrase draft → human reads it against the source → approve or edit." This has a real weakness: a reviewer skimming fluent prose can miss a subtle factual drift buried in otherwise-correct wording, especially changes in certainty rather than fact (e.g., "often recommended" silently becoming "is the best").

## Decision

Redesign the pipeline so the **claim set**, not the paraphrase, is the authoritative intermediate representation:

source → extract typed claims (entity/predicate/object/unit/qualifier) → generate paraphrase *from* the claim set (not from raw source prose directly) → re-extract claims from the generated paraphrase → automated comparison → human review only of flagged discrepancies.

The comparison checks entity/numeric/unit preservation and, specifically, **epistemic drift**: polarity ("does not" → "does"), quantifiers ("often" → "always"), modality ("can" → "will"), and attribution ("the guide recommends" → "X is objectively best").

## Rationale

There's solid research support for fact-aware generation and claim-level verification as an approach to reducing factual drift in paraphrase/summarization tasks, though no controlled study establishes a specific quantitative improvement for this exact task — this is adopted as a sound engineering control, not a proven percentage reduction. The practical benefit is changing the human reviewer's task from "spot anything wrong in a paragraph" (doesn't scale, easy to miss subtle drift) to "resolve the specific claims the automated comparison couldn't confidently match" (scales with corpus size, and targets exactly the failure mode most likely to slip past a skim-read).

## Consequences

- `source_manifest` needs additional curation-status states (`claims_extracted`, `flagged_review`) beyond simple draft/approved
- Content that's already a short, discrete factual statement can skip full paraphrase generation entirely and be stored as a structured fact directly — not everything needs to go through the full pipeline
- This is real, non-trivial pipeline logic to build (claim extraction and comparison), not just a prompt change — should be scoped accordingly in whichever spec covers the unstructured ingestion milestone (SPEC-001 M3)
- Which LLM performs claim extraction/paraphrase generation remains a separate, still-open decision (local runtime model vs. a stronger model used only for this build-time task)