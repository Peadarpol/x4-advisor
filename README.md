# x4-advisor

x4-advisor is an AI-powered assistant and decision support tool designed to analyze saved game states, game data, and player objectives for X4: Foundations.

> **Disclaimer**: This project is unofficial fan content. It is not affiliated with, endorsed, sponsored, or specifically approved by Egosoft GmbH.

## What this currently supports
*(TODO — fill in once Phase 1 is functional; don't claim capabilities that don't exist yet)*

## Hardware requirements
*(TODO — document only the actually-tested configuration once one exists; see the "supported vs. expected-to-work vs. unsupported" distinction before writing this)*

## Dependencies and prerequisites

### Software dependencies
*(TODO — summarizes and links out to required external tools and runtimes:*
*Python packages: see `pyproject.toml`/`poetry.lock`.*
*Tooling & Runtimes: `uv`, `x4cat` (see ADR-0006), and Ollama (see ADR-0004).*
*X4: Foundations itself, owned separately by the user — this project reads from an existing installation, it doesn't provide the game.)*

### Model dependencies

The system requires two models running via local [Ollama](https://ollama.com):
- **Embedding model**: `qwen3-embedding:0.6b` (~640 MB disk, ~0.6 GB VRAM) — see [ADR-0001](docs/adr/adr-0001-embedding-model.md)
  ```powershell
  ollama pull qwen3-embedding:0.6b
  ```
- **Language model**: `gemma4:12b` (~7.6 GB disk, ~8.0 GB VRAM at Q4_K_M) — ADR-0005 provisional baseline (final model selection subject to M6 evaluation bake-off)
  ```powershell
  ollama pull gemma4:12b
  ```

## Installation
*(TODO — after M1–M7 exist to actually describe)*

## Configuring your X4 installation
*(TODO)*

## Building the local knowledge base

The local unstructured knowledge base is built through interactive 3-pass claim extraction, dual-loop verification, operator approval, and vector embedding ingestion:

### Step 1: Pre-Vetting & Discovery
Register a candidate community guide or wiki source in `source_registry`:
```powershell
poetry run python -m x4_advisor.curation.cli register --source-id "src_001" --url "https://example.com/guide" --title "X4 Mining Guide" --proposed-by "peter_manual" --category "forum_guide"
```

### Step 2: Interactive 3-Pass Claim Extraction
Follow the instructions in [`docs/curation/claim-extraction-prompt.md`](docs/curation/claim-extraction-prompt.md):
1. Run the 3-pass prompt against your raw guide text in an LLM interface (Claude 3.5 Sonnet, ChatGPT 4o, Gemini 1.5 Pro).
2. Save output Pass 1 claims to `data/sources/<source_id>_c1.json`.
3. Save output Pass 2 paraphrased article to `data/sources/<source_id>_p.md`.
4. Save output Pass 3 re-extracted claims to `data/sources/<source_id>_c2.json`.

### Step 3: Dual-Loop Verification
Run automated fidelity verification (C1 vs C2 for epistemic/numeric drift) and database fact verification (C1 vs M1 SQLite DB):
```powershell
poetry run python -m x4_advisor.curation.cli verify --manifest-id "src_man_001" --source-id "src_001" --title "X4 Mining Guide" --c1-path "data/sources/src_001_c1.json" --c2-path "data/sources/src_001_c2.json"
```

### Step 4: Explicit Operator Approval (Mandatory Gate)
Inspect flagged discrepancies (if status is `flagged_review`) and grant explicit human approval:
```powershell
poetry run python -m x4_advisor.curation.cli approve --manifest-id "src_man_001"
```

### Step 5: Chunking & Ollama Vector Ingestion
Chunk paraphrased Markdown text, compute 1024-dimensional dense embeddings via `qwen3-embedding:0.6b` (Ollama), and populate `knowledge_chunks` and `sqlite-vec` virtual table:
```powershell
poetry run python -m x4_advisor.curation.cli ingest --manifest-id "src_man_001" --paraphrase-path "data/sources/src_001_p.md" --c1-path "data/sources/src_001_c1.json"
```

## Running the advisor

X4 Advisor provides a unified command-line tool `x4-advisor` (or `poetry run x4-advisor`):

### Diagnostic Pre-Flight Check (`doctor`)
Inspect the local Ollama daemon, model availability/VRAM residency, database schema integrity, and dataset freshness:
```powershell
poetry run x4-advisor doctor
```

### Single-Turn Query (`ask`)
Ask a natural-language question directly:
```powershell
poetry run x4-advisor ask "What is the cargo capacity of the Cerberus Vanguard?"
```
Use `--explain` or `-v` to inspect pipeline telemetry, route decisions, chunk retrieval metadata, and latency breakdown:
```powershell
poetry run x4-advisor ask --explain "What is the cargo capacity of the Cerberus Vanguard?"
```

### Interactive Terminal Session (`interactive`)
Start an interactive single-turn REPL:
```powershell
poetry run x4-advisor
# or
poetry run x4-advisor interactive
```
Available in-session REPL commands:
* `/help` — Display interactive help and available commands
* `/doctor` — Run environment and dataset diagnostics inline
* `/explain` — Toggle verbose routing and execution telemetry
* `/exit` (or `quit` / `q`) — Exit session cleanly

### Entity Disambiguation Resumption
When queries partially match multiple entities (e.g., `"What is the speed of the Magnetar?"`), the CLI displays a formatted selection menu and directly resumes execution via the canonical ID without re-prompting the router.

### Empirical Boundaries & Evidence Provenance
* **Evidence Provenance Transparency:** Every vector-derived answer displays explicit evidence chunk citations, source titles, and similarity scores.
* **Layer 1 Retrieval Recall Ceiling:** Unstructured chunk retrieval on strategic gameplay questions is bounded at **56.5%–65.2%**. When retrieval misses relevant evidence, the advisor produces explicit visible abstention badges (`[ABSTAIN: OUT_OF_SCOPE_DLC]`, `[ABSTAIN: NO_EVIDENCE]`, `[ABSTAIN: OUT_OF_SCOPE_OTHER]`).

## Licenses
MIT License. Note that no third-party game data or community content is distributed with this repository.

## Learning objectives
This project exists to learn RAG and local SLM architecture firsthand; design choices, like avoiding monolithic RAG frameworks and enforcing grammar-constrained routing, are deliberate — see the ADRs under `docs/adr/`.
