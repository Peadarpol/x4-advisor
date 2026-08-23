# Solution Design — X4 Advisor

**Status:** Approved
**Companion documents:** `project-charter.md` (why/what), `scope-boundary.md` (Phase 1 in/out boundaries, to be drafted separately)

This document describes *how* X4 Advisor is built. It assumes the charter's decisions as given and does not re-justify them; where a decision has a dedicated ADR, this document references it rather than re-explaining the reasoning.

---

## 1. Architecture Overview

X4 Advisor is a hybrid retrieval system, not pure RAG. The core insight driving this shape: most high-value X4 questions are structured-data lookups (ship stats, ware prices, production chains) with exact, deterministic answers, while a smaller set of questions (strategy, heuristics, "why" explanations) genuinely require retrieval over unstructured community knowledge.

The system has two distinct stages that run at different times, not one continuous pipeline. Stage 1 runs offline, ahead of time, whenever the knowledge base needs building or refreshing. Stage 2 runs at query time, every time the user asks a question, and depends entirely on Stage 1 having already populated the data store.

### Stage 1 — Offline ingestion (builds the data store; see §3 for full detail)

```
Game files (.cat/.dat)              Curated wiki/community sources
         │                                      │
         ▼                                      ▼
   x4cat extraction                    source_registry (trusted)
         │                                      │
         ▼                                      ▼
   normalized structured               raw capture → claim extraction
   records                             → paraphrase → claim verification
         │                                      │
         ▼                                      ▼
   structured tables ─────────────────► shared SQLite file ◄──── knowledge_chunks
   (ships, wares,                       (§4 Data Model)          (sqlite-vec, embedded)
    production, factions,
    sectors)
```

This stage is run by whoever is populating the knowledge base (the owner, or — per the charter's revised priority ordering — anyone else willing to run it themselves against their own game install and their own sourced content). It is not part of what a user interacts with; it produces the populated database that Stage 2 then queries.

### Stage 2 — Runtime query (this is what the rest of §1 describes)

```
User Question
      │
      ▼
┌─────────────────┐
│  Router (LLM,    │  Native tool-calling: selects
│  tool-calling)   │  query_structured_data / search_knowledge_base / both
└────────┬─────────┘
         │
    ┌────┴─────────────────────┐
    │                          │
    ▼                          ▼
┌──────────────┐        ┌────────────────┐
│ SQL Query     │        │ Vector Search   │
│ (SQLite)      │        │ (sqlite-vec)    │
└───────┬───────┘        └────────┬────────┘
        │                         │
        └───────────┬─────────────┘
                     ▼
          ┌─────────────────────┐
          │  LLM Synthesizer     │
          │  (grounded answer     │
          │   generation)         │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Grounding Check     │
          │  (claim-level         │
          │   entailment verify)  │
          └──────────┬──────────┘
                     │
                     ▼
              Answer to User
```

The router and synthesizer are the same underlying LLM, called twice with different prompts/tool definitions — not two separate models. Stage 2 assumes Stage 1 has already run; if it hasn't (a fresh clone with no database yet), this is one of the error conditions named in §7's error-handling list, not a state Stage 2 needs to handle gracefully on its own.

**Phase 1 MVP note:** the Stage 2 diagram above shows the full end-state architecture. Per the charter's stated build order (structured extraction → unstructured ingestion → routing → synthesis → interface), the earliest working version has no router and no vector path at all — it's structured queries only, direct to the synthesizer. The router and vector-search branch are added once that simpler loop works end-to-end, not built simultaneously with it.

## 2. Component Breakdown

Mapped to the folders already scaffolded in the repository:

- **`ingestion/`** — see §3 for the full pipeline design; briefly, `game_data_extractor.py` handles structured game-file extraction and `wiki_content.py` handles the manually-curated, LLM-assisted-paraphrase content pipeline
  - `chunker.py`: heading-aware *and* size-bounded semantic chunking (splits oversized sections rather than treating "semantic" and "bounded" as mutually exclusive), with a fallback to paragraph-level splitting with overlap for source material that lacks clear structural markers — structured data never passes through this
- **`embeddings/`** — wraps the embedding model (Qwen3-Embedding-0.6B via Ollama) for the unstructured ingestion path and for embedding user queries at retrieval time
- **`storage/`** — owns the SQLite schema: relational tables for ships/wares/factions/production-chains, and the sqlite-vec virtual table for embedded chunks, in the same database file
- **`retrieval/`**
  - `router.py`: the tool-calling router — defines the available "tools" (structured query, vector search) and interprets the LLM's tool-call decision
  - `structured_query.py`: translates a routed structured request into a parameterized SQL query against the relational tables — not raw text-to-SQL generation by the LLM, to avoid injection risk and unpredictable query shapes; the LLM selects *which* structured lookup to run from a small, fixed set of query templates, it does not write arbitrary SQL
  - `vector_query.py`: embeds the query and runs similarity search against sqlite-vec
- **`llm/synthesizer.py`** — the second LLM call: takes the user's question plus whatever was retrieved (structured rows, vector chunks, or both) and generates the final natural-language answer, explicitly instructed to use only the supplied material
- **`ui/app.py`** — the chat interface (Streamlit for Phase 1, per the charter)
- **`save_parser/`, `voice/`** — out of scope for this document; Phase 2 and Phase 3 respectively, to be designed when their specs are drafted

## 3. Ingestion Pipeline

This section didn't exist in earlier drafts of this document — ingestion was described as two component files with a one-line summary each. Given how much has actually been decided since, it earns a proper section of its own.

### 3.1 Source discovery (pre-pipeline, not yet executed)

Sources are found, not just processed. Discovery runs across three independent channels before anything is captured:
- The owner's own manual research (Google searches, personal judgment)
- Claude performing web searches for candidate sources
- A research brief handed to ChatGPT, specifically instructed to only report sources it can confirm actually exist and are accessible, rather than guess at plausible-sounding URLs

Candidates are recorded in a `source_registry` table — separate from, and earlier-stage than, the `source_manifest` that tracks content actually in the pipeline:

```
source_registry
---------------
source_id, url, title, proposed_by (peter_manual | claude_search | chatgpt_brief),
category (wiki | forum_guide | steam_guide | other), topic_tags,
proposed_date, status (proposed → trusted → rejected → superseded),
trust_rationale, reviewed_by, reviewed_date, notes,
content_date, last_checked, superseded_by
```

A source being `trusted` doesn't mean permanently current — `content_date` and `last_checked` let staleness be assessed later (a 2021 guide may be fine or badly outdated; this at least makes that assessable rather than invisible), and `superseded_by` links forward when a better source replaces an older one rather than just marking the old one `superseded` with no trail to what replaced it.

Corroboration across channels (multiple channels independently surfacing the same source) is a useful trust signal, but the `trusted` designation is always a deliberate human decision, never automatic. Only `trusted` entries graduate into `source_manifest` and become eligible for raw capture. **This mechanism is designed but not yet run** — no sources have been discovered or vetted as of this document's current state.

### 3.2 Structured data pipeline (game files)

```
.cat/.dat archives (root-level only)
        │
        ▼
    x4cat (primary extraction/indexing dependency)
        │
        ▼
normalized intermediate representation  ← isolates schema from XML format changes
        │
        ▼
validation (open — see below)
        │
        ▼
domain schema → SQLite
```

- **Extraction tool: resolved.** `x4cat` is the primary dependency — verified MIT-licensed, actively released (checked directly against the repository), explicitly tested against current X4 versions, with CLI and indexing/search tooling well-suited to a Python pipeline. Egosoft's own **X Catalog Tool** is kept as a reference/fallback extractor for troubleshooting format questions, not a pipeline dependency. **X4FProjector** — despite genuinely excellent semantic coverage of exactly the categories this project needs (ships, wares, production, equipment) — is used only as an independent cross-check oracle during schema development, not a runtime dependency: its own maintenance signals (low commit count, no confirmed current-version compatibility) don't clear the bar for a foundational dependency, but comparing its output against this project's own parser during development is a cheap, valuable way to catch parsing bugs. This project's own normalization layer stays independent of `x4cat`'s internal data model regardless — `x4cat` is a well-vetted way to get the raw game data out, not a source of this project's own domain schema.
- **Validation strategy: two distinct concerns, not one.** "Does extraction succeed at all" (malformed/incomplete records — recommend skip-and-log with a visible report, per above) is different from "is the extracted data actually correct" — domain-invariant checks (cargo capacity ≥ 0, referenced ware/faction IDs actually exist, no self-referencing production recipes, ship class within an allowed set) that catch extraction bugs before they become silently-wrong answers. **This is not the same problem the grounding check solves** — grounding verifies the LLM's answer matches retrieved data; it says nothing about whether that retrieved data was correct in the first place. A small set of domain invariants, checked at ingestion time, closes this gap cheaply.
- **Entity identity: use the game's own internal ID, not display name, as the primary key.** Ship/ware names aren't reliable identifiers (renames, localization, near-duplicates) — `x4cat`'s indexed output exposes internal macro/ware IDs, which should be the actual join key, with display name carried as an attribute, not the identity.
- **Re-ingestion triggering: resolved.** The pipeline compares the stored `dataset_metadata` game version against the currently-installed version at startup/query time and surfaces a staleness warning. It does **not** auto-re-extract silently — re-running extraction stays a deliberate, manual action. This closes the open question from earlier planning about what actually *uses* the version-tracking field.
- **No redistribution:** the extracted database is never shipped pre-built, for anyone, including as a convenience to the owner's own future self on a different machine — extraction always runs against the actual installed game, so the data always matches the actual game version in use. This is a direct consequence of the versioning risk above, not just the charter's separate redistribution principle, though both point the same direction.

### 3.3 Unstructured content pipeline (wiki/community sources)

```
trusted entries in source_registry
        │
        ▼
raw capture → data/raw/ (gitignored, never committed, never redistributed)
        │
        ▼
extract typed claims (entity/predicate/object/unit/qualifier/provenance span)
        │
        ▼
generate paraphrase FROM the claim set, not from raw prose directly
        │
        ▼
re-extract claims from the generated paraphrase
        │
        ▼
automated claim-level comparison → PASS / FLAG
        │
        ▼
human review (only flagged claims, not full-text re-reading)
        │
        ▼
chunk (heading-aware + size-bounded, paragraph-fallback for unstructured sources)
        │
        ▼
embed → incremental insert into knowledge_chunks
```

- **Paraphrasing mechanism, revised:** the earlier "paraphrase → human reads everything" design has a real weakness — a reviewer skimming fluent prose can miss a subtle factual drift buried in otherwise-correct wording. The redesigned pipeline makes the **claim set**, not the paraphrase, the authoritative intermediate representation: source content is first decomposed into typed claims (e.g., `{subject: "Mercury Vanguard", predicate: "cargo_capacity", object: 3000, unit: "m³"}`), the paraphrase is generated *from* that claim set rather than directly from the source prose, and claims are re-extracted from the generated paraphrase and automatically compared against the originals.
- **What the automated comparison checks, specifically:** entity preservation, numeric/unit preservation, and — the category most likely to slip past a human skim — **epistemic drift**: polarity ("does not" becoming "does"), quantifiers ("often/usually" becoming "always"), modality ("can/may" becoming "will/must"), and attribution ("the guide recommends X" becoming "X is objectively best"). No numerical fact needs to change for meaning to shift materially, and this is exactly the failure mode a fluent-prose comparison misses.
- **What this changes for the human reviewer:** the review task changes from "spot anything wrong in this paragraph" to "resolve the specific claims the automated comparison couldn't confidently match" — a much narrower, more scalable task, and one that scales with corpus size far better than re-reading full paraphrases against full sources every time.
- **Not everything needs full paraphrasing.** Content that's already a short, discrete factual statement can be stored as a structured fact directly rather than run through paraphrase generation at all; genuine strategic/explanatory prose is where paraphrasing actually earns its place. This also means numeric values in the final knowledge base can be templated in from the validated claim record rather than regenerated in prose by the LLM each time — removing an entire class of "the model restated a number slightly wrong" risk.
- **No proven quantitative claim attached to this design.** There's solid research support for fact-aware generation and claim-level verification as an architectural approach to reducing factual drift, but no controlled evidence establishing a specific measured improvement over simpler paraphrasing for this kind of task. This is adopted as a sound engineering control, not because it's been proven to reduce drift by some specific percentage — worth being honest about that distinction rather than overclaiming.
- **Which LLM does the paraphrasing: still open.** The claim-first redesign doesn't resolve this — the same choice remains, just applied at the "generate paraphrase from claims" step instead of "paraphrase raw source": the same local model the advisor runs at inference time, or a stronger model (Claude/GPT) used only for this build-time task. Confirm before implementation.
- **Incremental, not full-rebuild:** adding one new approved document appends its chunks rather than reprocessing the whole corpus; an edited or removed source has its chunks identified and replaced via `source_id`, not a full wipe-and-redo.
- **`source_manifest` tracks curation status** (`draft` → `claims_extracted` → `flagged_review` → `approved`) — the additional intermediate states reflect the claim-verification step, not just draft/approved as before.
- **No redistribution:** paraphrased content, and the claim sets extracted from it, stay in `data/`, gitignored, never shipped. Anyone else populates their own knowledge base by running the same pipeline against their own discovered and vetted sources.



## 4. Data Model

**Structured tables** (SQLite, one database file shared with the vector table):
- `ships(id, name, class, hull, shields, cargo_capacity, speed, ...)`
- `wares(id, name, category, min_price, avg_price, max_price)`
- `production_recipes(ware_id, input_ware_id, input_amount, output_amount, production_time, method)` — enables production-chain traversal via recursive queries; `method` distinguishes multiple valid production recipes for the same output ware (X4 commonly has more than one production method per ware — omitting this column would silently collapse distinct recipes together)
- `factions(id, name, relations_summary)`
- `sectors(id, name, faction_id, resource_yields)`

Exact column lists and types are implementation detail for the relevant spec, not this document — this section exists to confirm the *shape* of the structured layer, not its final schema.

**Entity name resolution:** the router extracts natural-language strings from user questions ("Cerberus Vanguard," or just "Cerberus"), but the schema's primary keys are internal macro/ware IDs (per ADR-0006), not display names. A resolution layer is needed between the two — case-insensitive exact match against `display_name` first, falling back to partial/substring match if no exact match is found. An ambiguous partial match (multiple candidates) is a distinct outcome from "no match at all" and needs its own handling, not a silent pick of the first result (see `SPEC-001` §8, which needs this failure mode added).

**Dataset provenance:** every generated database carries a `dataset_metadata` record (game version, build, extraction timestamp, base-game-only flag, schema version) — without this, "grounded" doesn't distinguish between grounded-in-the-correct-game-version and grounded-in-whatever-was-extracted-once. Directly addresses the game-update staleness risk.

**Vector layer metadata:** `knowledge_chunks` carries filterable metadata alongside the embedding (source, section, game-version scope, topic, related entity IDs) — sqlite-vec supports metadata columns and filtering in KNN queries directly, so this is a matter of schema design now rather than a retrofit later. This is also the mechanism that enforces the base-game-only scope at the vector layer, not just at extraction time.

**Known gaps, flagged as scope decisions rather than settled here:** the current sketch omits ship equipment/loadouts (turret/shield/engine connection slots and compatibility), station/production modules, universe topology (sectors/clusters/gates as distinct from a flat sector list), and a clean distinction between static price bounds vs. save-specific actual prices (relevant once Phase 2 exists). These are real categories of player questions, and whether each is in Phase 1 scope or a deliberate cut is exactly the kind of decision `scope-boundary.md` exists to make explicitly — this document isn't the place to silently expand or silently omit them. Also missing: a game-version field (see dataset provenance above).

**Unstructured layer:** a single sqlite-vec virtual table (`knowledge_chunks`) storing chunk text, its embedding vector, and a source-attribution field (which wiki page/section it came from, for traceability — never displayed to the user as a citation of copyrighted text, only used internally for the grounding check).

**Why one database file, not two systems:** sqlite-vec supports both relational tables and vector search in the same SQLite file, so structured and unstructured data live side by side without needing a separate vector database process — consistent with the ADR decision to use sqlite-vec specifically for its embedded, single-file simplicity.

## 5. Routing Logic

The router is an LLM call using native tool-calling (Ollama, supported by both Gemma 4 and Qwen3-family models), not free-text classification. Four outcomes are possible, not three:

- `query_structured_data(query_type, parameters)` — for fact-lookup and comparison questions (e.g., "cargo capacity of the Cerberus Vanguard," "which L-class miners have the most cargo")
- `search_knowledge_base(query_text)` — for strategy/heuristic/explanatory questions
- **Both** — for hybrid questions (e.g., "what does Hull Parts production require, and why is it strategically important?" — needs both the production-chain data and strategic context). Note: an earlier draft used "why is Hull Parts production profitable" as this example — that's a poor example, since profitability requires production cost, workforce, and market-price data that Phase 1's structured scope explicitly excludes (`scope-boundary.md` §1.1). Don't let an illustrative example silently imply scope the actual data model doesn't support. Evaluation should test route *completeness*, not just accuracy: a hybrid question routed to only one path is a distinct failure mode from picking the wrong single path.
- **Retrieved content is data, never instructions.** Community-sourced content (curated, but still external text) is ultimately fed to an LLM alongside the user's question. A source containing something like "ignore previous instructions and..." must never be able to alter routing behavior, tool permissions, or output policy — this needs to be an explicit architectural boundary (retrieved documents are evidence, structurally separated from system/tool instructions in the prompt), not an assumption that curation alone prevents it.
- **Tool boundary, made explicit:** `query_structured_data` and `search_knowledge_base` are both read-only at the connection level (no write capability exists to misuse, not just "isn't used"); unknown tool calls are rejected; parameters are validated before use; result counts are capped. Cheap to build in from the start, expensive to retrofit.
- **Abstention has distinct reasons, not one collapsed "I don't know":** no evidence found, question is out-of-scope (e.g., DLC content), and conflicting evidence between sources are three different situations a trustworthy advisor should surface differently — collapsing them loses information the user could act on (e.g., knowing "this is a DLC question" is actionable in a way "I don't know" isn't).

`query_type` is drawn from a small, fixed set of query templates (not arbitrary SQL generation by the LLM) — this is a deliberate safety and predictability choice, not a limitation to work around later. The router's job is classification and parameter extraction, not query authorship.

**Source authority when structured and unstructured evidence conflict:** current installed base-game structured data outranks curated community explanatory material for factual/statistical claims — the synthesizer's prompt should encode this hierarchy explicitly rather than leaving it to the model to arbitrate implicitly.

This mechanism is deliberately the simplest reliable option available, consistent with the earlier decision not to over-build the router before the core loop works end-to-end.

## 6. External Dependencies

- **Ollama** — local inference runtime for both the LLM and the embedding model (see runtime ADR for why Ollama over llama.cpp/vLLM)
- **sqlite-vec** — still pre-1.0 and explicitly warns of possible breaking changes; the version in use should be pinned explicitly (in the relevant ADR or `pyproject.toml`) with a documented upgrade procedure, rather than floating on "latest"
- **`.cat`/`.dat` extraction tooling (`x4cat`)** — a real prerequisite, not incidental. Since the structured pipeline depends on `x4cat`'s continued correct behavior, a small set of golden extraction fixtures (a handful of known ship/ware/recipe records with expected output) should exist so an `x4cat` version bump can be tested against known-good results rather than trusted blindly — cheap insurance against a supply-chain surprise silently changing extracted data.
- **Base-game X4 installation** — the source of structured data; the project reads from it, never modifies it
- **Wiki/community sources** — read at ingestion time, paraphrased into the knowledge base; not a runtime dependency once ingestion has run

## 7. Non-Functional Considerations

- **VRAM budget:** verified against actual GGUF file listings across multiple independent builds (not estimated from first principles) — Gemma 4 12B Q6_K weight files run 9.8–10.2GB depending on the specific quantization build, Q5_K_M ~8.4–8.6GB. **Weight file size is not the same as runtime VRAM** — KV cache, CUDA/runtime buffers, and context all sit on top of it, so "the weights fit" doesn't by itself confirm "this configuration runs on the target hardware." Granite 4.1 8B (Q6_K ~7.2GB) and Qwen3 14B (Q4_K_M ~9.3GB) both have more comfortable headroom. Final model, quantization, and context-length configuration remain an empirical Phase 1 decision, validated on the actual RTX 3080 Ti with the actual Ollama version and router/synthesis prompts — not assumed from file sizes alone.
- **Grounding — three layers, not one runtime step:**
  - **Layer 1 (deterministic, every query, no LLM cost):** structured answers can only contain values SQL actually returned; every retrieval result carries an immutable source/chunk ID; malformed tool calls are rejected; empty or low-confidence retrieval triggers abstention upstream of the synthesizer, not as a prompt instruction to it
  - **Layer 2 (offline CI gate, the charter's required Phase 1 deliverable):** a curated evaluation corpus scored for routing accuracy, route *completeness* (not just correctness — a hybrid question routed to only one path is a distinct failure mode), claim-level entailment, and unsupported/contradicted claim rate, blocking merge below a threshold. Granite Guardian 4.1 8B (IBM's purpose-built RAG-hallucination/groundedness judge model, a comfortable fit on 12GB) is a strong candidate to use here rather than inventing a grounding metric from scratch.
  - **Layer 3 (optional, selective runtime verification):** invoked only for low-confidence retrieval, hybrid questions, or when structured and unstructured evidence disagree — not run on every query, avoiding the latency cost of a universal third LLM call while still catching the cases most likely to be wrong
  - **Claim taxonomy:** answers should be evaluable as FACT (directly supported), SUPPORTED_INFERENCE (conclusion follows from evidence but isn't verbatim), ADVICE (a recommendation, not a factual claim), UNSUPPORTED (introduced, not in evidence), or CONTRADICTED (differs from retrieved material) — this is what gives "grounded" an actual, checkable definition rather than a vague aspiration
- **Latency:** no hard target set yet, though the layered grounding model above removes the largest latency risk (no universal third LLM call). A provisional target (e.g., "under 20 seconds for a typical question") belongs in `scope-boundary.md` once the router+synthesizer path can actually be measured, rather than estimated in advance.
- **Multi-turn conversation:** not yet decided. The chat interface naturally invites multi-turn use, but whether the router and synthesizer receive conversation history — and how much — is an open design question with real context-budget implications given the VRAM constraints above. Recommend explicitly scoping Phase 1 as single-turn (no conversation memory) in `scope-boundary.md` unless there's a strong reason to solve this now.
- **Error handling:** the major failure modes need at least a one-line intended behavior each, to be finalized in the relevant spec — a structured query template receiving parameters that don't match any record; vector search returning only low-similarity results; Ollama unreachable or the model not loaded; the database not yet built (ingestion hasn't run); a question about out-of-scope DLC content. "Say so explicitly rather than fabricating an answer" is the general principle from the charter; each specific mode still needs its own stated behavior.
- **Configuration model:** `config.py` and `.env` (`X4_INSTALL_PATH`, `X4_SAVE_FOLDER`, `OLLAMA_ENDPOINT`) are scaffolded but undesigned — where the user sets these, what defaults exist, and what validation happens on startup affects the charter's "third party can clone and run" success criterion directly, and deserves a few lines in whichever spec implements the CLI/app entry point.
- **Source attribution, reconsidered:** the earlier decision was "never displayed to the user, internal use only" to avoid any copyright exposure — but *naming* a source ("Source: X4 Community Wiki — Production") is a different thing from *displaying* copyrighted text, and costs nothing in copyright risk while adding real user-facing value (transparency, easier debugging, and a natural way to signal "this is strategic guidance, not a direct game-data fact"). Worth allowing source labels in the UI even though the underlying copyrighted text itself stays out.
- **Voice subsystem resource contention (Phase 3, architectural constraint recorded now):** the STT/TTS stack should not reserve VRAM while the LLM is actively serving a request unless explicitly configured to do so — there's no requirement for the voice stack to be resident concurrently with LLM inference, and designing around that now avoids a three-way VRAM contention problem (LLM + Whisper + TTS) that Phase 3 would otherwise inherit by default.
- **DLC entities in Phase 2 save files:** already resolved in the charter — skip and note as unrecognized, rather than fail

---

*Next: `scope-boundary.md` will define Phase 1's precise in/out boundary at implementation granularity — exact question categories, exact data categories, and explicit non-goals — so `SPEC-001` has a hard edge to build inside rather than an open invitation to scope-creep.*