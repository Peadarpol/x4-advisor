# Project Charter — X4 Advisor

**Status:** Approved
**Repository:** github.com/Peadarpol/x4-advisor
**License:** MIT
**Owner / Sponsor / Approver:** Peter (Peadarpol)

---

## 1. Project Name and Overview

**X4 Advisor** is a local, offline AI advisor for the game *X4: Foundations*, grounded via retrieval-augmented generation (RAG) — meaning its answers are built from facts and text retrieved from a curated knowledge base at query time, rather than relying solely on what the underlying language model already knows. It answers questions about ships, wares, production chains, factions, and strategy using knowledge extracted from the base game's own data files and curated community sources, synthesized by a locally-run LLM — no cloud dependency, no internet connection required at run time.

This is an unofficial fan project, not affiliated with or endorsed by Egosoft.

## 2. Purpose and Business Case

This project serves three motivations, in priority order — the first two are the actual drivers; the third is a welcome side effect, not a design goal to optimize around:

1. **A genuinely fun, useful tool the owner wants for their own X4 play.** 
2. **A learning vehicle.** This is the owner's first hands-on build of an LLM + datastore (RAG) agent, following six months of agentic development experience concentrated in CRUD/RBAC-style work. Chunking, embedding, retrieval routing, and grounding are new territory, and the build process is deliberately structured to develop that understanding directly, not just to produce a working tool.
3. **Sharing with others, if they want it — not the reason it's being built.** The code will be public and usable by others, but this is explicitly secondary: the project isn't designed around making onboarding frictionless for third parties, and decisions (like requiring everyone to run their own ingestion pipeline rather than shipping pre-collected data) are made in favor of the first two motivations even where they add friction for a hypothetical third-party user. Research conducted during planning confirmed no mature, dedicated, local-first knowledge advisor currently exists for X4: Foundations — existing efforts are either early-alpha live-telemetry mods (Andromeda) or generic, poorly-tuned overlay tools (Gaming Copilot). There is real room for a well-scoped, base-game-focused advisor.

## 3. Objectives

- **Phase 1:** A working local chat advisor, answering ship/ware/economy/strategy questions, grounded in structured game-data extraction and curated wiki/community content, synthesized via a local LLM (see Technology Stack for the current model shortlist — Qwen3.6 has since been evaluated and eliminated). Intended build order: structured data extraction first, unstructured/wiki ingestion second, routing third, synthesis fourth, chat interface last — each layer working end-to-end before the next is added, rather than building all layers in parallel.
- **Phase 2:** Save-file-aware advice — parsing the player's most recent save/quicksave to give situated recommendations, without any live in-game telemetry dependency.
- **Phase 3:** Voice interface (local STT/TTS) layered on top of the working text pipeline, using a second monitor for the advisor's display surface.

## 4. Scope

**In scope:**
- Base game only — no DLC-specific content, ships, or mechanics
- Fully local/offline operation — no cloud LLM calls, no external API dependency at runtime
- Structured data extraction from the player's own installed game files (read-only, no mod installation)
- Curated, base-game-scoped wiki/community content, chunked and embedded locally
- Save-file parsing for Phase 2 (read-only, no mod dependency)
- Public, MIT-licensed repository, shareable and usable by others without requiring them to install any third-party game modification

**Explicitly out of scope:**
- Any dependency on third-party X4 mods (SirNukes Mod Support APIs, named-pipes/HTTP telemetry bridges, x4-simpit, or similar) — ruled out specifically because the project must be usable by others without requiring them to alter their own game installation
- Live, continuous in-game telemetry — superseded by save-file snapshot parsing
- Autopilot or decision-making behavior — the tool advises, it does not act on the player's behalf or control ship functions (a deliberate distinction from "copilot"-style tools in the genre, and one that sidesteps the X4 community's documented skepticism toward LLMs making strategic decisions)
- DLC content of any kind

## 5. Stakeholders and Roles

| Role | Party | Responsibility |
|---|---|---|
| Owner / Sponsor / Approver | Peter (Peadarpol) | Sets direction, reviews and approves all specs and PRs, sole authority to merge to `main` |
| Implementation agent | Gemini / Antigravity, via the `Peadarpol-AiDelivery` account | Scaffolds, implements, and delivers code against approved specs; cannot self-approve or merge to `main` |
| Architecture & review consultant | Claude | Technical consulting, spec drafting support, adversarial review, research validation |

## 6. Governance Approach

A deliberately lightweight, *spec-anchored* discipline. The owner maintains a separate, more elaborate governance harness (`ai-delivery-control`) for larger, multi-contributor projects; this project borrows that harness's core practices — spec-before-code, adversarial review, real enforcement gates — without adopting the harness's tooling itself. The parts of that harness aimed at coordinating many contributors or multiple interlocking codebases (decision ledgers for cross-module dependencies, configurable enforcement strictness for gradual team-wide rollout, and so on) don't carry over here, since this is a solo project with a single, self-contained codebase and none of those coordination problems exist to solve.

- **Specs before code**, stored under `docs/planning/specs/`, following DRAFT → APPROVED → DELIVERED lifecycle
- **Self-approval prohibited** — the implementing agent cannot mark its own work approved or delivered
- **Branch protection is real, not aspirational**: a GitHub ruleset on `main` requires a pull request and at least one approval before merge; the delivery agent has Write-only access and cannot push directly to `main`
- **A working pre-commit test gate**, not just a config file — verified installed and passing before being relied upon
- **ADRs** (`docs/adr/`) capture standalone technology decisions (embedding model, vector store, RAG framework, etc.) separately from feature specs
- **Adversarial review is required before any merge to `main`.** In practice, since the review consultant (Claude) has no GitHub account and does not comment on PRs directly, the owner performs the actual PR review and merge decision, informed by consulting the review consultant beforehand. This is a deliberate adaptation for a solo project, not a gap — the discipline is preserved (nothing merges without independent scrutiny of the implementation agent's own work) even though the reviewing and approving party is the same person
- **A project-local incident log** (`docs/incidents/`) tracks new, RAG/agent-specific failure modes as they're discovered — starting with the required Phase 1 deliverable below
- **Outstanding, explicitly tracked gap:** an automated retrieval-grounding check (verifying that any answer the synthesizer claims is "grounded" actually traces to real retrieved content) does not yet exist and is a required Phase 1 deliverable — nothing should be considered fully gated until it's built

## 7. Technology Stack

- **Language/tooling:** Python, managed with Poetry, src-layout package structure
- **Local LLM:** Qwen3.6 has been evaluated and eliminated — no variant below 27B exists, and 27B's real quantized footprint (~19GB at Q4_K_M) does not fit a 12GB card without compromises severe enough to defeat the purpose. The current shortlist is **Gemma 4 12B (Q6_K)** as the provisional default, **Granite 4.1 8B** as a strong alternative purpose-built for tool-calling/RAG/classification without reasoning overhead, and **Qwen3 14B** as a mature control with field evidence of successful local RAG use. Final selection between Gemma 4 12B and Granite 4.1 8B is deliberately deferred to empirical testing using Phase 1's own retrieval-grounding evaluation harness, rather than decided from desk research alone — see the LLM selection ADR.
- **Local inference runtime:** Ollama, chosen after evaluating llama.cpp (direct) and vLLM as alternatives — vLLM's throughput advantage is irrelevant to a single-user workload with no concurrent requests to serve, and llama.cpp direct is a "graduate to this if you outgrow Ollama" step, not a starting point, given no current need for cutting-edge model architectures or fine-grained KV-cache control that Ollama doesn't already support. See the inference runtime ADR.
- **Embeddings:** Qwen3-Embedding-0.6B
- **Vector store:** sqlite-vec (embedded, single-file, no server process)
- **RAG approach:** a hybrid retrieval architecture, not pure RAG. Structured game data (ships, wares, production chains, factions) is parsed into SQLite relational tables and queried directly; unstructured knowledge (strategy guides, community wisdom) is chunked, embedded, and retrieved via sqlite-vec vector search. A routing step classifies each question and directs it to the appropriate path (or both), then an LLM synthesizes the final answer from whatever was retrieved. The router uses native tool-calling (structured function selection, supported directly by Ollama for both Gemma 4 and Qwen3-family models) rather than free-text classification — this is both the simpler and more reliable mechanism, consistent with the earlier decision to keep the router deliberately lightweight rather than building a sophisticated classifier before the core loop even works end-to-end. No LangChain/LlamaIndex, deliberately, given the single-user scale and the value of building this by hand for the learning objective
- **Structured data:** extracted directly from the player's installed X4 game files into SQLite
- **Later phases:** faster-whisper (STT) and Piper or Kokoro (TTS) for the voice interface; save-file XML parsing (streaming/iterparse, given memory constraints on large saves) for Phase 2

## 8. Success Criteria

- Phase 1: a functioning end-to-end chat loop, answering representative ship/economy/strategy questions with retrieval-grounded, non-fabricated responses
- The retrieval-grounding check exists and passes as part of the pre-commit/CI gate before Phase 1 is considered complete
- Phase 2: the advisor can correctly answer a situational question (e.g., "what ships do I currently own?") directly from a parsed save file. Save files may contain DLC ships/wares even though this project is base-game-scoped; the advisor must handle unrecognized DLC entities gracefully (skip and note as unrecognized, rather than fail) — the exact handling is a deliberate Phase 2 design decision, not an accident to discover later
- Phase 3: the full voice loop (STT → retrieval/synthesis → TTS) completes an exchange without exceeding available VRAM or requiring a fallback to system RAM
- At least one new, genuinely distinct incident-taxonomy entry is identified and documented over the course of the build, demonstrating the harness's applicability beyond CRUD/RBAC domains
- **Secondary, not gating:** the repository is structured cleanly enough that a third party *could* clone it and run the code — but per the project's actual priority order (§2), this is a welcome side effect, not something Phase 1 completion depends on. Specifically: the code and pipelines are shareable; the collected/curated knowledge base is not, and never will be. Redistributing collected wiki/community content — even paraphrased — would mean passing off other people's collected knowledge as the project's own asset, which the project deliberately avoids regardless of copyright technicalities. Anyone who wants the tool populated with data runs the extraction and curation pipelines against their own game install and their own sourced content, the same way the owner does. This applies equally to structured game data, not just unstructured content — nobody's extracted database gets redistributed either, since shipping a pre-built one would silently go stale against a recipient's actual game version (see the versioning risk already noted in Assumptions).

## 9. Assumptions and Constraints

- Local hardware: RTX 3080 Ti, 12GB VRAM — constrains which model size/quantization tiers are viable for both the LLM and any concurrently-running STT model. Model selection has progressed from fully open to a confirmed shortlist (Gemma 4 12B, Granite 4.1 8B, Qwen3 14B); final choice between the top two candidates is deferred to empirical testing via Phase 1's grounding harness. Phase 3's additional STT/TTS VRAM budget remains unprofiled and should be resolved via ADR before Phase 3 planning proceeds.
- Game data extraction is not a simple file read: X4 stores its data in `.cat`/`.dat` archives that require dedicated extraction tooling (community tools such as X Catalog Tool exist for this) before the structured data pipeline can ingest anything. This is a real prerequisite step in the ingestion pipeline, not an incidental detail.
- No live telemetry access; Phase 2 situational awareness is therefore snapshot-based (as of last save), not real-time
- Wiki/community-sourced content is paraphrased and restructured, never reproduced verbatim, consistent with copyright constraints
- Egosoft's own fan content guidelines apply; existing precedent (numerous unopposed "X4"-branded community tools) suggests a low practical risk, but the project maintains a clear unofficial-fan-content disclaimer regardless. **Note:** a separate charter review claimed Egosoft requires a specific verbatim disclaimer format including a "Copyright © EGOSOFT 1990-2026" attribution line. That claim could not be verified — Egosoft's fan content policy page blocks automated access, and the copyright string in question is Egosoft's standard site-wide footer notice, not confirmed fan-content-specific language. Treat any specific wording requirement as unconfirmed until someone reads the actual policy page directly in a browser.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Retrieval quality is poor / answers are ungrounded | Retrieval-grounding check as a required, gated deliverable, not an afterthought |
| Scope creep toward live telemetry / mod dependency | Explicitly ruled out in Scope (Section 4); revisit only as a deliberate, separately-scoped future decision |
| VRAM constraints limit model quality | Resolved: Qwen3.6 was confirmed unsuitable for 12GB hardware (no sub-27B variant exists) and eliminated. Current shortlist (Gemma 4 12B, Granite 4.1 8B, Qwen3 14B as control) is confirmed to fit 12GB at usable quantization levels; final selection between the top two is deferred to empirical testing via Phase 1's grounding harness rather than assumed |
| Copyright exposure from scraped content | Structured facts and paraphrased summaries only; no verbatim reproduction; `data/` directory is gitignored in its entirety |
| Solo-developer bus factor / continuity | Specs, ADRs, and incident log serve as the continuity mechanism, same rationale as `ai-delivery-control` |

## 11. Timeline

No fixed external deadlines — this is a personal project. Phases are sequenced (1 → 2 → 3) rather than dated, with each phase's completion gated on its own success criteria rather than a calendar milestone.

## 12. Budget

Not applicable — no monetary budget; the constrained resource is the owner's own time and existing local hardware.