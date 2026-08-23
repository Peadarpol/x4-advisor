# Scope Boundary — X4 Advisor (All Phases)

**Status:** Approved
**Companion documents:** `project-charter.md` (why/what), `solution-design.md` (how, Phase 1 detail)

This document bounds scope across the whole project, not just Phase 1. Phase 1 is specified in full detail, since it's next to be built. Phases 2 and 3 are specified at the level of detail already decided during planning — real constraints and decisions, not vague aspirations — with open items explicitly marked as deferred to that phase's own spec, rather than omitted as if undecided.

---

## Cross-phase non-goals (apply regardless of phase, never revisited without a deliberate decision)

- No dependency on third-party X4 mods, of any kind, in any phase — this was the reason Phase 2 uses save-file parsing instead of live telemetry in the first place
- No autopilot or decision-making on the player's behalf — the tool advises, it never acts or controls ship functions
- No DLC content, structured or unstructured, in any phase
- No cloud dependency at runtime — fully local/offline, always

---

## Phase 1 — Chat Advisor

### 1.1 Structured data in scope

| Category | Phase 1 scope | Deferred |
|---|---|---|
| Ships | Hull, cargo capacity, speed, shield rating (aggregate value), turret/weapon **slot counts** | Full equipment compatibility matrix (which specific weapons/shields/engines fit which slots) — **Phase 1.x** |
| Wares | Name, category, static price bounds (min/avg/max from game files) | Save-specific actual prices — this is a Phase 2 concept, see §2.1 |
| Production chains | Ware-level recipe traversal (what inputs produce ware X, what X produces) | Station/production-module modeling (which modules exist, workforce requirements, construction resources) — **Phase 1.x** |
| Factions | Name, sector ownership | Detailed relations/diplomacy state — **Phase 1.x** |
| Sectors | Name, owning faction, resource yield | Adjacency/gate-connection graph (sector-to-sector routing) — **Phase 1.x** |

**Rationale for the cuts:** equipment compatibility and station-module data are each large enough to be their own significant modeling effort, and Phase 1's purpose is proving the retrieval/synthesis loop works end-to-end, not maximizing data coverage on the first pass.

### 1.2 Query templates — the structured retrieval surface, enumerated

- **T1 — Single-entity fact lookup:** "What's the cargo capacity of the Cerberus Vanguard?"
- **T2 — Comparison/ranking:** "Which L-class miners have the most cargo?"
- **T3 — Production chain traversal:** "What do I need to produce Claytronics?"
- **T4 — Category listing:** "List all Argon ships" / "What wares are in the Ore category?"

A structured question that doesn't fit one of these four templates routes to the knowledge-base path rather than the router silently failing or the templates silently growing without a deliberate decision to add a fifth.

### 1.3 Unstructured knowledge — scope

Mechanism already decided (solution design): manually curated documents, not automated scraping.

- **In scope:** strategy/heuristic content relevant to the four query templates above, plus explanatory "why" content that doesn't reduce to a database lookup
- **Out of scope:** lore/narrative content, patch-note history, anything DLC-specific
- **Initial corpus size:** no fixed number — start with enough to cover the four templates' strategic dimension, grow only when a real gap surfaces during testing

### 1.4 Conversation model

**Single-turn.** No conversation history passed to the router or synthesizer. Multi-turn support has real costs (co-reference resolution, context-window budget competing with retrieved content) not worth paying before the single-turn loop is proven. Revisit explicitly once Phase 1 is stable.

**One narrow, deliberate exception:** when entity name resolution (`solution-design.md` §4, `SPEC-001` §4/§8) finds an ambiguous match, the system asks a clarifying question ("did you mean X or Y?") and accepts the user's answer as a direct follow-up — settling the ambiguity rather than guessing, per an explicit preference over silently picking one candidate to keep the interaction brief. This is **not** general conversation memory — it's a single, scoped clarification tied to one specific unresolved parameter, not a standing capability to reference earlier turns generally. The distinction matters: a future "let's add real multi-turn" decision shouldn't quietly ride in on this narrow exception's coattails.

### 1.5 Latency — provisional target

**Under 20 seconds** for a single-path question, **under 30 seconds** for a hybrid question — measured warm (model already loaded, not counting first-request cold start) on the actual target hardware. Provisional, to validate empirically — not derived from the unverified tok/s estimates flagged earlier in planning. If measurement misses these targets, revisit model/quantization choice first, not the target.

### 1.6 Grounding evaluation — Phase 1 floor

**Minimum of 30 curated question/expected-answer pairs**, spanning all four query templates plus knowledge-base and hybrid questions, scored via the FACT / SUPPORTED_INFERENCE / ADVICE / UNSUPPORTED / CONTRADICTED taxonomy from the solution design. A floor to clear Phase 1, not a ceiling.

### 1.7 Base game vs. DLC, at implementation granularity

- Ingestion extracts only from root-level `.cat`/`.dat` archives; `extensions/ego_dlc_*` is never read
- Accepted trade-off: DLC-patched base files mean the advisor's data can diverge slightly from what a DLC-owning player's live game shows — known, deliberate, not a Phase 1 bug
- No DLC content in the curated knowledge base, curator's responsibility to exclude at source-selection time

### 1.8 Explicit non-goals (Phase 1-specific, beyond the cross-phase list)

- No multi-turn conversation (§1.4)
- No automated wiki scraping (§1.3)
- No equipment/loadout compatibility modeling, station/production-module modeling, or sector adjacency graph (§1.1)
- No save-specific pricing — doesn't exist as a concept until Phase 2

### 1.9 Phase 1 exit criteria

- All four query templates return correct, grounded answers against the evaluation corpus
- At least one genuine knowledge-base-only question and one hybrid question are answered correctly and grounded per the claim taxonomy
- The 30-question evaluation floor passes the offline grounding gate
- The full loop runs via the Streamlit chat interface, single-turn, within the provisional latency targets
- **Secondary, not gating:** the codebase and pipelines are clean enough that a third party could clone the repository, follow the README, and run both pipelines against their own X4 installation and their own sourced content — but per the project's actual priority order, this isn't something Phase 1 completion depends on, and no pre-collected data (structured or unstructured) is ever distributed as part of meeting it

---

## Phase 2 — Save-File-Aware Advisor

Phase 2 builds on Phase 1's working retrieval/synthesis loop, adding a third data source: the player's own save file, read as a periodic snapshot rather than live telemetry (the mod-dependent live-telemetry path was explicitly ruled out earlier in planning, specifically so the project stays usable by others without requiring them to alter their own game install).

### 2.1 Data categories from the save file (provisional — exact field mapping is Phase 2 spec work)

- Ships currently owned (name, class, location, assigned role)
- Credits / treasury
- Faction standing
- Station inventories and production status
- **Actual, current ware prices** — this is where "save-specific pricing," deferred from Phase 1 (§1.1), actually lands; it doesn't exist as a meaningful concept until there's a save file to read prices from

### 2.2 Mechanism

- Read-only parsing of the gzip-compressed XML save file, using a streaming parser (iterparse) rather than loading the full tree — naive full-load parsing of a large save has been flagged as memory-expensive (a 1GB save can cost roughly 16GB of RAM to process naively) and this project's hardware budget doesn't have that kind of headroom to spare casually
- **Snapshot-based, not real-time:** the advisor's picture of the player's empire is "as of the last save," not continuously updated — this is the direct consequence of ruling out live telemetry, not an oversight
- No modification of the save file under any circumstance — read-only, always

### 2.3 Trigger model (open question for the Phase 2 spec)

Two candidate approaches, not yet chosen between:
- **On-demand:** the player asks ("check my empire") and the advisor re-parses the most recent save at that moment
- **Background watcher:** a process notices when a new save file appears and refreshes automatically

Either is compatible with the architecture; which one (or both) ships in Phase 2 is a decision for that phase's own spec, not this document.

### 2.4 DLC entities in save files

Already decided: if a player has DLC installed, their save may reference DLC ships/wares. These are skipped from detailed answers, but **unrecognized entities must never become silently absent data** — if a player owns 10 ships and 1 is an unrecognized DLC entity, the correct answer is "9 of 10 ships are supported by this advisor; 1 is an unrecognized DLC entity," not a silent "you own 9 ships." The difference matters: the latter is simply wrong, not just incomplete.

### 2.5 Explicit non-goals (Phase 2)

- No live/continuous telemetry of any kind
- No named-pipes or mod-based data bridge (SirNukes-style APIs, x4-simpit, or similar) — ruled out at the project level, restated here for completeness
- No writing to the save file
- No autopilot or in-game action-taking based on save state

### 2.6 Provisional exit criteria

- The advisor correctly answers a situational question (e.g., "what ships do I currently own?") directly from a parsed save file
- Unrecognized DLC entities are handled gracefully (skipped and noted, not a crash or silent wrong answer)
- Memory usage during save parsing has been validated against a real, large save file on the target hardware — not assumed safe from the streaming-parser choice alone

### 2.7 Open questions, explicitly deferred to the Phase 2 spec

- Exact save-file field mapping (which XML paths correspond to which data categories in §2.1)
- Final trigger mechanism choice (§2.3)
- Whether conversational self-reporting (the player telling the advisor their situation in chat, as a zero-integration fallback) is folded into Phase 2 or remains a standing Phase 1 capability regardless

---

## Phase 3 — Voice Interface

Phase 3 adds voice input/output on top of the working text pipeline from Phases 1–2. It is explicitly a thin layer over an already-working system, not a parallel build.

### 3.1 Components (decided)

- **STT:** faster-whisper, chosen specifically for GPU-accelerated performance on the project's NVIDIA hardware
- **TTS:** not yet finalized — Piper is the lightweight default but now ships under a GPL-3.0 fork rather than its original MIT license (worth a licensing check before committing, given this project's own MIT license); Kokoro-82M is a similarly lightweight alternative with reportedly more natural output, worth a direct listening comparison before choosing

### 3.2 Interaction model (decided)

- Global hotkey or wake-word activation, not overlay-based — the second-monitor display setup removes the focus-contention problem that would otherwise have required an in-game overlay or click-through window
- A persistent window on the second monitor shows transcript and microphone status; no in-game overlay, no borderless-windowed-mode dependency

### 3.3 VRAM constraint (decided, architectural)

The voice subsystem must not reserve VRAM while the LLM is actively serving a request unless explicitly configured to do so. There's no requirement for STT/TTS to be resident concurrently with LLM inference — designing around that now avoids a three-way VRAM contention problem (LLM + Whisper + TTS) that would otherwise be inherited by default.

### 3.4 Explicit non-goals (Phase 3)

- No new live-game-state access beyond what Phase 2 already established — Phase 3 is a new input/output modality, not a new data source
- No overlay or click-through window — superseded by the second-monitor decision
- No requirement to hold voice models resident in VRAM when idle

### 3.5 Provisional exit criteria

- A full voice loop (STT → retrieval/synthesis → TTS) completes an exchange without exceeding available VRAM or requiring fallback to system RAM

### 3.6 Open questions, explicitly deferred to the Phase 3 spec

- Final TTS model choice (Piper vs. Kokoro), pending a direct quality comparison and the licensing check above
- Wake-word vs. hotkey as the actual activation mechanism (either is architecturally compatible; not yet chosen)
- The real VRAM budget for the full three-model stack (LLM + STT + TTS), which can't be validated until Phase 1's model/quantization choice is empirically settled

---

Anything not named as in-scope for its phase above is out of scope for that phase, by default, not by omission.