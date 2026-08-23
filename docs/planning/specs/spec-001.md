# SPEC-001 — Phase 1 Foundation

**Status:** Approved
**References:** `project-charter.md` (why/governance), `scope-boundary.md` §1 (what's in/out), `solution-design.md` (how)

This spec turns Phase 1's charter and scope decisions into an implementation-ready contract. It doesn't re-explain decisions already made elsewhere — where something has a charter section, scope-boundary section, or ADR, this spec references it and states only what's new: concrete, testable requirements.

---

## 1. Build milestones

Matches the charter's stated build order, broken into checkable increments rather than one large undifferentiated "Phase 1" effort:

- **M1 — Structured extraction:** `x4cat` pipeline runs against a real X4 installation, produces validated records (§3) in the domain schema, in SQLite.
- **M2 — Structured query engine:** the four query templates (`scope-boundary.md` §1.2) work against the extracted data, testable directly (script/test harness), no chat UI or LLM router yet.
- **M3 — Unstructured ingestion pipeline:** source registry → claim extraction → paraphrase-from-claims → claim verification → chunk → embed, populating `knowledge_chunks`.
- **M4 — Vector retrieval:** query embedding + sqlite-vec similarity search against the M3 corpus.
- **M5 — Router + synthesis:** LLM tool-calling router selects structured/vector/both/abstain; synthesizer generates answers from retrieved evidence.
- **M6 — Grounding (Layers 1–3):** deterministic runtime checks, offline evaluation gate, selective runtime verification (`solution-design.md` §7).
- **M7 — Chat interface:** Streamlit, single-turn, ties M1–M6 together into what a user actually interacts with.

Each milestone should be independently testable before the next starts — this is the practical meaning of the charter's "each layer working end-to-end before the next is added."

---

## 2. Data correctness (distinct from grounding — M1 requirement)

Grounding verifies an LLM answer matches retrieved data; it says nothing about whether that data was correct on extraction. Before any record enters SQLite:

- Cargo capacity, speed ≥ 0 (and speed > 0 for anything that isn't a station)
- Production recipe input/output quantities > 0
- Every `ware_id` referenced by a recipe or ship actually exists in the `wares` table
- Every `faction_id` referenced by a sector actually exists in the `factions` table
- Ship `class` is within an allowed enum, not free text
- Internal IDs are unique (no two records claim the same `x4cat`-derived ID)
- Required fields (name, at minimum) are not null

**Identity:** the game's own internal macro/ware ID is the primary key for every entity, never the display name — names aren't reliable identifiers (localization, renames, near-duplicates across variants).

**Failure handling:** a record failing any invariant is skipped and logged in a visible ingestion report, not silently dropped and not an aborting failure for the whole run (`solution-design.md` §3.2).

## 3. DLC boundary (M1 requirement)

- Extraction reads only root-level `.cat`/`.dat` archives; `extensions/ego_dlc_*` is never read
- No DLC content enters the structured database or the curated knowledge base, by construction, not by downstream filtering

## 4. Query templates (M2 requirement)

The four templates from `scope-boundary.md` §1.2 (T1 fact lookup, T2 comparison/ranking, T3 production chain traversal, T4 category listing) are implemented as fixed, parameterized queries — never LLM-generated SQL. A question that fits none of them is not a template implementation gap to patch reactively; it routes to the knowledge-base path (M5).

**Entity name resolution** (`structured_query.py`): the router extracts natural-language entity names from questions, but records are keyed by internal ID (§2), not display name. Resolution order: case-insensitive exact match against `display_name`, then partial/substring match if no exact match exists. An ambiguous partial match (more than one candidate) is a distinct, explicit outcome — not a silent pick of the first or best-scoring result (§8).

## 5. Router and tool boundary (M5 requirement)

- Router uses native tool-calling (Ollama), not free-text classification, per `solution-design.md` §5
- **Ollama API calls must explicitly set `"think": false`** for both router and synthesizer calls. Gemma 4 (and other hybrid-thinking models) default to producing a visible chain-of-thought reasoning trace via Ollama, which adds unnecessary latency for this task's straightforward classification/synthesis workload. Confirmed via direct testing against `gemma4:12b` on the target hardware: omitting `think: false` produces a multi-step reasoning trace before the answer; setting it explicitly returns a clean, fast response (~0.4s total duration in testing) with no thinking trace. This must be set explicitly in every API call — do not rely on a default, and do not rely on `ollama run`'s interactive CLI behavior, which is for human use only and does not reflect what the application code should do.
- Four outcomes: structured query, vector search, both, abstain
- **Retrieved content is data, never instructions** — a source containing something resembling an instruction to the model must not be able to alter routing, tool permissions, or output policy. This is a structural separation in how retrieved content is placed in the prompt, not a hope that curation alone prevents it.
- Tools are read-only at the connection level (no write path exists to misuse); unknown tool calls are rejected; parameters are validated before use; result counts are capped

## 6. Grounding contract (M6 requirement)

Every claim in a synthesized answer is classified as:

| Class | Meaning | Permitted in the final answer? |
|---|---|---|
| FACT | Directly stated by retrieved evidence | Yes |
| SUPPORTED_INFERENCE | Logically follows from evidence, not verbatim | Yes, and should read as inference, not as a bare fact |
| ADVICE | An explicit recommendation/opinion | Yes, explicitly framed as such, never presented as a factual claim |
| UNSUPPORTED | Not established by any retrieved evidence | No — triggers Layer 3 verification or abstention |
| CONTRADICTED | Evidence contradicts the claim | No — same as above |

This is what makes "grounded" a checkable requirement instead of an aspiration. The offline evaluation gate (M6, `scope-boundary.md` §1.6) scores against this table directly.

## 7. Abstention (M5/M6 requirement)

Three distinct reasons, surfaced differently to the user, not collapsed into one "I don't know":

- **No evidence** — structured and vector retrieval both returned nothing usable
- **Out of scope** — the question concerns DLC content or another named non-goal
- **Conflicting evidence** — structured and unstructured sources disagree, or two unstructured sources disagree

## 8. Failure behavior (M5–M7 requirement)

| Condition | Required behavior |
|---|---|
| Ollama unreachable | Clear operational error, not a hang or generic exception |
| Model not loaded | Setup guidance pointing at the missing prerequisite |
| Database doesn't exist yet | Explain that ingestion (M1) hasn't run, not a raw SQL error |
| Database stale (game version mismatch) | Warning, not a silent wrong answer (dataset versioning per `solution-design.md` §4) |
| Structured query returns nothing | "No matching data found," not a fabricated answer |
| Entity name matches more than one record ambiguously | Ask the user which one they meant, and accept their answer — settle it, don't guess, per explicit preference over silently picking a candidate to keep the interaction brief. This is the one deliberate, narrow exception to single-turn (`scope-boundary.md` §1.4) — scoped to resolving this one parameter, not general conversation memory. |
| Vector search returns only low-similarity results | Treated as no-evidence abstention (§7), not passed to the synthesizer as if relevant |
| Router emits a malformed/invalid tool call | Rejected, retried once, then abstain — never silently ignored |
| A save-related or DLC entity appears (Phase 2 territory, noted for consistency) | Never silently omitted from a count — "9 of 10 supported, 1 unrecognized," not a quietly wrong total |

## 9. Configuration and startup contract

For Phase 1, required configuration is minimal:

```
X4_INSTALL_PATH
OLLAMA_ENDPOINT
MODEL_NAME
EMBEDDING_MODEL
DATABASE_PATH
VECTOR_RELEVANCE_THRESHOLD
```

- **`VECTOR_RELEVANCE_THRESHOLD` is a special case:** unlike the others, it's not a value the user sets — it's determined empirically against the evaluation corpus (§11) during M6, then set as the default. Don't hardcode a number because it "sounds about right"; the threshold that separates "relevant enough to use" from "no-evidence abstention" (§7, §8) needs to come from actually measuring against real cases.

- **Where defined:** `.env`, per `.env.example` in the repository root
- **Defaults:** `OLLAMA_ENDPOINT` defaults to Ollama's standard local address; the others have no safe default (an install path or database path guessed wrong is worse than an explicit required value) and must be set explicitly
- **Startup validation:** the application checks required configuration is present and points at something real (the install path exists and looks like an X4 installation, the database path either exists or ingestion clearly hasn't run yet, Ollama actually responds at the configured endpoint) before doing anything else
- **Fail fast:** invalid or missing configuration produces a clear, specific error naming exactly what's wrong and where to fix it, at startup — never a delayed failure partway through answering a question, and never a silent fallback to a guessed value

## 10. Non-functional requirements

- No network access required at runtime (fully local/offline)
- Warm-request latency (model already loaded): under 20 seconds single-path, under 30 seconds hybrid — measured on the actual target hardware, not estimated (`scope-boundary.md` §1.5)
- Single-turn only — no conversation history passed to router or synthesizer (`scope-boundary.md` §1.4)
- Structured database is read-only during query execution — no write path from the query engine

## 11. Evaluation corpus (M6 requirement)

Minimum 30 question/expected-answer pairs (`scope-boundary.md` §1.6). Coverage should reasonably span, not exhaustively enumerate:
- All four query templates, plus at least one knowledge-base-only and one hybrid question
- At least one deliberate no-evidence case and one out-of-scope (DLC) case, to verify abstention actually fires
- At least one case per grounding class (§6) so the offline gate has something of each type to score
- **At least one "inference laundering" case** — evidence supports a narrower claim than the answer asserts (e.g., evidence gives one ship's cargo capacity; a bad answer claims that makes it "the best choice for large-scale trading" as if that followed logically, when it's actually ADVICE, not SUPPORTED_INFERENCE). This specific boundary is the most likely place for the claim taxonomy to leak.
- **At least one structured-vs-community conflict case** — a question where structured data and a community source would give different answers, to verify the source-authority hierarchy (§5) actually wins in practice, not just in the prompt text

**Ground truth format**, not just a question/answer pair — each case needs enough structure that scoring isn't circular (the judge model shouldn't be the only thing deciding whether an answer was right):

```
case_id, question, expected_route, expected_entities,
expected_facts, allowed_inferences, expected_abstention,
expected_evidence_ids
```

This is a floor to clear M6/Phase 1, not a target corpus size to stop at — it grows afterward as real usage surfaces gaps.

## 12. Dependencies requiring their own ADR before or during this spec

- LLM selection: empirical bake-off between Gemma 4 12B, Granite 4.1 8B, and Qwen3 14B (control) using this spec's own evaluation harness (§10) — not decided from desk research (charter §7, already established)
- `sqlite-vec` version pin + upgrade procedure
- Golden extraction fixtures for `x4cat` (a handful of known ship/ware/recipe records with expected output, so an `x4cat` version bump is tested against known-good results rather than trusted blindly)

## 13. Exit criteria

Per `scope-boundary.md` §1.9, restated here as the acceptance gate for this spec specifically: all four query templates return correct, grounded answers against the evaluation corpus; the 30-question floor passes the offline grounding gate; the full loop runs via Streamlit, single-turn, within the latency targets; a third party could clone and run both pipelines against their own inputs (secondary, not gating, per the charter's revised priority ordering).