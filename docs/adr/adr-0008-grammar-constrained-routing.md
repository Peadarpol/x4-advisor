# ADR 0008 — Grammar-Constrained JSON Schema Routing via Inference Format Decoding

**Status:** Accepted

## Context

In Milestone M6/M7.0, query routing relied on native LLM tool-calling (via Ollama's `tools` parameter). Under unconstrained tool-calling, smaller open-weights models (such as Granite 8B and Qwen 14B) frequently emitted malformed tool arguments, hallucinated parameter combinations (e.g. `category: 'ships'` with `metric: 'cargo_capacity'`), or invented out-of-scope values (e.g. `production_method: 'terran'`, `metric: 'price'`), causing structured precision to crater (20%–58.8%).

Furthermore, tool-calling frameworks rely on heuristic JSON decoding where syntax errors or out-of-vocabulary enums require fragile retry loops that degrade response latency.

## Decision

Transition the query router from unconstrained tool-calling to **grammar-constrained JSON Schema decoding** via Ollama's `format` parameter.

1. **Single Source of Truth Vocabularies:** Define code-derived constants (`VALID_OPERATIONS`, `ALLOWED_METRICS`, `VALID_PURPOSES`, `VALID_SHIP_CLASSES`) and database-derived sets (categories, resources, production methods) in `src/x4_advisor/retrieval/vocabularies.py`.
2. **Grammar Enforcement:** Pass the unified JSON Schema to Ollama with `format=schema`, ensuring the LLM's token logits are strictly constrained at the decoder level to valid schema tokens and canonical enums.
3. **Internal Normalization & Validation:** Map the schema structure to query engine methods (`fact_lookup`, `ranking`, `production_chain`, `category_listing`, `sector_yield`), while retaining Python-level coherence validation as a secondary defense.

## Rationale

- **Zero Syntax Errors:** Grammar-constrained decoding mathematically prevents malformed JSON or illegal enum values at token generation time.
- **Deterministic Coherence:** Models cannot combine incompatible parameters or invent out-of-scope DLC terms.
- **Latency Reduction:** Eliminates tool parsing errors and retry cycles, reducing router latency overhead.
- **Architectural Symmetry:** Provides a uniform decoding structure for single-path, hybrid (`BOTH`), and abstention routing.

## Consequences

- Router output is now a typed JSON object instead of a list of tool call dictionaries.
- Schema enums are continuously qualified against the live database allowlists via `tests/unit/test_router_schema_contract.py`.
- Router prompt instructions are simplified to selecting schema fields rather than documenting tool definitions.
