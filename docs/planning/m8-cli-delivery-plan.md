# Milestone M8 Implementation Plan: CLI Delivery

## Goal Description
Deliver the unified, production-ready Command-Line Interface (`x4-advisor`) for X4 Advisor. This milestone enhances the existing CLI ([`src/x4_advisor/cli.py`](file:///c:/projects/x4-advisor/src/x4_advisor/cli.py)), registers the package entrypoint, and provides an epistemically transparent, production-hardened terminal interface supporting interactive REPL querying, single-shot question answering, single-turn entity disambiguation resumption, diagnostic health checks (`doctor`), dataset staleness validation, and responsive request cancellation.

---

## Scope Invariants & Architectural Boundaries

> [!IMPORTANT]
> **Carried-Forward Empirical Boundaries & Risk Register (from M6/M7):**
> 1. **Provisional Model Baseline:** The default operational model is `gemma4:12b` (Q4_K_M) per ADR-0005. It is documented as a provisional selection operating under known empirical gaps (10.9% UCR, 5 benchmark contradictions) rather than a gate-cleared model. Startup-time `--model` flag allows selecting `qwen3:14b` with configuration validation.
> 2. **Layer 1 Retrieval Recall Ceiling:** Unstructured chunk retrieval on strategic queries remains bounded at **56.5%–65.2%**. When vector retrieval misses relevant chunks, the synthesizer may abstain or produce bounded guidance.
> 3. **Evidence Provenance Transparency (Runtime Invariant):**
>    - Offline 5-class claim verification is *not* run inline at runtime to protect latency.
>    - Runtime honesty is delivered through **evidence provenance transparency**:
>      * **Vector / Hybrid queries:** Joins `synthesis_result.evidence_chunk_ids` with `vector_result.chunks` to display explicit `heading_hierarchy`, `source_attribution`, `chunk_id`, and `similarity_score`.
>      * **Structured queries:** Surfaces exact database records, matching entities, and redirected categories.
>      * **Abstentions:** Displays explicit visible badges for all engine reasons (`[ABSTAIN: OUT_OF_SCOPE_DLC]`, `[ABSTAIN: NO_EVIDENCE]`, `[ABSTAIN: OUT_OF_SCOPE_OTHER]`).

> [!CAUTION]
> **Architectural Boundary Guardrails (SPEC-001 §8 & §14, `scope-boundary.md` §1.4):**
> - **Zero Conversation Memory:** The REPL is strictly single-turn. Every question is evaluated in complete isolation with zero conversation history passed to router or synthesizer prompts.
> - **Single-Turn Disambiguation Resume:** Entity disambiguation is the one deliberate, narrow single-parameter exception. It reuses M5's `pending_route` continuation token directly with `resolved_entity_id`, avoiding a second router call and preventing name re-ambiguation loops. State is strictly rolled back if the user cancels selection.
> - **Connection Management:** Query execution delegates entirely to `AdvisorEngine(config=config)` enforcing `PRAGMA query_only = ON;`. Diagnostic inspection opens an isolated read-only SQLite connection.
> - **Diagnostics Isolation:** `x4-advisor doctor` bypasses startup validation probing (`validate_m5_config`) so that it executes and reports complete diagnostics even when Ollama is offline or configuration is invalid.

---

## Proposed Changes

### 1. Robust Request Cancellation via Socket Shutdown & Client Refactor

#### [MODIFY] [`src/x4_advisor/llm/client.py`](file:///c:/projects/x4-advisor/src/x4_advisor/llm/client.py)
- Define new exception class:
  ```python
  class OllamaCancelledError(RuntimeError):
      """Raised when an in-flight Ollama generation request is cancelled by caller."""
  ```
- Refactor `_post_json` to use explicit `http.client.HTTPConnection` (or `HTTPSConnection`):
  - Parses host/port from `urlparse(self.endpoint)` and extracts path/query from the target `url` argument.
  - Initializes `HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_sec)`.
  - Spawns worker thread for the blocking `conn.request("POST", path, ...)` and `conn.getresponse()`.
  - Worker thread records the response/exception in context and sets a per-request `done_event: threading.Event`.
  - Main thread executes a bounded polling loop wrapped in `try ... except BaseException:`:
    ```python
    try:
        while not done_event.wait(0.05):
            if cancel_event and cancel_event.is_set():
                _cancel_socket(conn)
                raise OllamaCancelledError("Request cancelled by caller")
            if time.monotonic() - start_time > timeout_sec:
                _cancel_socket(conn)
                raise OllamaTimeoutError(f"Ollama call timed out after {timeout_sec:.2f}s")
    except BaseException:
        _cancel_socket(conn)
        raise
    ```
  - On cancellation (`KeyboardInterrupt`, `cancel_event`, or timeout):
    * `_cancel_socket(conn)`: checks `sock = getattr(conn, "sock", None)` and calls `sock.shutdown(socket.SHUT_RDWR)` & `conn.close()` inside a `try/except (OSError, AttributeError)` block.
    * This breaks the TCP connection at the OS level, causing Ollama server-side to receive client disconnect and abort generation immediately, releasing GPU resources.
  - **Exception & Status Translation Rewrite:**
    * Inspects `response.status` explicitly:
      - `response.status == 404` $\rightarrow$ raises `OllamaModelNotFoundError(f"Model '{model}' not found. Run 'ollama pull {model}' to download it.")`.
      - Other `response.status >= 400` $\rightarrow$ raises `OllamaConnectionError(f"Ollama API returned HTTP {response.status}: {error_body}")`.
    * Translates `http.client.RemoteDisconnected`, `ConnectionRefusedError`, `socket.timeout`, and `OSError` into `OllamaTimeoutError` and `OllamaConnectionError`.
- **Cancellation Guard in `warmup()`:** Insert `except OllamaCancelledError: raise` ahead of the broad `except Exception:` handlers (lines 148, 155), preventing startup warmup cancellation from being silently swallowed.

#### [MODIFY] [`src/x4_advisor/retrieval/router.py`](file:///c:/projects/x4-advisor/src/x4_advisor/retrieval/router.py)
- **Cancellation Propagation Guard:** `OllamaCancelledError` subclasses `RuntimeError`, so the three broad `except Exception` handlers (lines 102, 146, 174) would otherwise catch it and silently degrade the query to `ABSTAIN`. Insert an explicit re-raise ahead of each broad handler:
  ```python
  except OllamaCancelledError:
      raise
  except Exception as e:
      ...  # existing ABSTAIN fallback
  ```

#### [MODIFY] [`src/x4_advisor/llm/synthesizer.py`](file:///c:/projects/x4-advisor/src/x4_advisor/llm/synthesizer.py)
- **Cancellation Propagation Guard:** Insert `except OllamaCancelledError: raise` ahead of the broad `except Exception` handler at line 163, preventing cancelled synthesis from falling back to a degraded `SynthesisResult`.

#### [MODIFY] [`tests/unit/test_ollama_client.py`](file:///c:/projects/x4-advisor/tests/unit/test_ollama_client.py)
- Update mock harness to patch `http.client.HTTPConnection`, asserting `OllamaModelNotFoundError` on 404, `OllamaTimeoutError` on timeout, `OllamaCancelledError` on cancellation, and `OllamaConnectionError` on connection drops.

---

### 2. Schema Constant, Dataset Staleness & Diagnostics (`x4-advisor doctor`)

#### [MODIFY] [`src/x4_advisor/storage/schema.py`](file:///c:/projects/x4-advisor/src/x4_advisor/storage/schema.py)
- Define canonical schema version constant:
  ```python
  EXPECTED_SCHEMA_VERSION: str = "1.1.0"
  ```

#### [MODIFY] [`src/x4_advisor/storage/models.py`](file:///c:/projects/x4-advisor/src/x4_advisor/storage/models.py)
- Update `DatasetMetadata` dataclass default to reference the constant:
  ```python
  from x4_advisor.storage.schema import EXPECTED_SCHEMA_VERSION

  schema_version: str = EXPECTED_SCHEMA_VERSION
  ```

#### [NEW] [`src/x4_advisor/diagnostics.py`](file:///c:/projects/x4-advisor/src/x4_advisor/diagnostics.py)
- Implements `run_diagnostics(config: Config) -> DiagnosticReport`:
  1. **Isolated Execution:** Invoked with a `Config` constructed by the CLI using `Config(validate=False)` with `--db` and `--model` overrides already applied. `run_diagnostics` performs no probing validation and never raises `ConfigError` — all failure conditions are reported as structured report entries.
  2. **Ollama Daemon Status:** Probes `config.ollama_endpoint` for connectivity and server version.
  3. **Configured Model Residency & VRAM Ratio:** Queries `/api/tags` and `/api/ps` to check if `config.model_name` and `config.embedding_model` exist. Formats `size_vram` vs `size` from `/api/ps` to report VRAM ratio and detect CPU offloading. All model names sourced dynamically from `config.model_name`.
  4. **Database & Ingestion Integrity:** Verifies `config.database_path` existence, schema integrity, and non-zero row counts across all 6 core tables (`ships`, `wares`, `sectors`, `sector_resources`, `factions`, `production_recipes`) and `knowledge_chunks`.
  5. **Dataset Staleness Validation (SPEC-001 §14):** Inspects the SQLite `dataset_metadata` table for:
     * `game_version`, `build`, `extraction_timestamp`, `is_base_game_only`, `schema_version`.
     * Validates `schema_version == EXPECTED_SCHEMA_VERSION` and `is_base_game_only == 1`.
     * Warns if `dataset_metadata` table is missing or unpopulated.
  6. **Calibrated Thresholds & Known Boundaries:** Reports active retrieval threshold ($\tau = 0.50$), model status (conditionally labeling `gemma4:12b` as ADR-0005 provisional default), and Layer 1 recall boundary (56.5%–65.2%).
  7. **Exit Codes:** Returns `0` on all critical checks green, `1` if any critical component (Ollama, models, database) fails.

---

### 3. Production CLI Overhaul & 4-State REPL Interaction

#### [MODIFY] [`src/x4_advisor/cli.py`](file:///c:/projects/x4-advisor/src/x4_advisor/cli.py)
- **Subcommand & Flag Structure (via `parents=[common_parser]`):**
  - `x4-advisor ask [OPTIONS] "QUESTION"`: Single-shot question execution. Exits `0` on successful answers (including abstentions).
  - `x4-advisor interactive [OPTIONS]` (or bare `x4-advisor [OPTIONS]`): Interactive REPL session.
  - `x4-advisor doctor [OPTIONS]`: Runs pre-flight diagnostics (bypasses probe validation; ignores `--explain`).
  - **Shared Options:**
    - `--model <name>`: Startup-time model selection override (validated via `validate_m5_config`).
    - `--explain` / `-v`: Displays verbose routing parameters, SQL query details, chunk retrieval details, and latency breakdown.
    - `--no-warmup`: Bypasses startup model warmup (`engine.client.warmup()`).
    - `--skip-probe`: Bypasses startup Ollama endpoint reachability check during configuration validation.
    - `--db <path>`: Overrides database path by setting `config.database_path_str = str(path)`.
    - `--no-color`: Disables ANSI color codes (auto-disabled if `not sys.stdout.isatty()` or `NO_COLOR` env var is present).
- **REPL Command Surface:**
  - Supported REPL commands:
    * `/help` (or `help`): Displays session help and commands.
    * `/doctor` (or `doctor`): Runs diagnostic audit inline.
    * `/explain` (or `explain`): Toggles verbose telemetry for subsequent answers.
    * `/exit` (or `exit`, `quit`, `q`): Exits session cleanly (exit code `0`).
    * `EOFError` (Ctrl+D on Unix, Ctrl+Z+Enter on Windows): Exits cleanly (exit code `0`).
- **Explicit 4-State Ctrl+C Handling:**
  1. `WARMUP_IN_PROGRESS`: Catches `OllamaCancelledError` / `KeyboardInterrupt` during startup warmup, prints `"[Warmup skipped by user — first query may incur cold-start latency]"` to stderr, and immediately proceeds to REPL prompt without raising an unhandled traceback.
  2. `IN_FLIGHT_GENERATION`: Catches `OllamaCancelledError` / `KeyboardInterrupt`, cancels active query, triggers socket shutdown, rolls back `pending_route = None`, and prints `"[Query cancelled by user]"` to stderr without exiting REPL.
  3. `AWAITING_DISAMBIGUATION_INPUT`: Catches Ctrl+C, rolls back `pending_route = None`, prints `"[Selection cancelled]"`, and returns to top-level REPL prompt.
  4. `IDLE_PROMPT`: Catches Ctrl+C at empty prompt and exits session cleanly with exit code `130`.
- **Exit Codes:**
  - `0`: Successful execution (including abstentions and clean REPL exit).
  - `1`: Diagnostic failure (`doctor` reports unhealthy critical components).
  - `2`: Configuration or CLI argument syntax error (`ConfigError` / invalid flags).
  - `130`: User interrupt / SIGINT at idle prompt.
- **Single-Turn Disambiguation Continuation:**
  - When `resp.ambiguous_candidates` is returned, renders formatted numbered list:
    ```text
    Multiple entities matched 'Magnetar':
      [1] Magnetar (Mineral) Vanguard (ID: ship_arg_m_miner_solid_01_a_macro, Type: ship)
      [2] Magnetar (Mineral) Sentinel (ID: ship_arg_m_miner_solid_01_b_macro, Type: ship)
      [3] Magnetar (Gas) Vanguard (ID: ship_arg_m_miner_liquid_01_a_macro, Type: ship)
      [4] Magnetar (Gas) Sentinel (ID: ship_arg_m_miner_liquid_01_b_macro, Type: ship)
    Select entity [1-4] or 'c' to cancel: 
    ```
  - Upon valid numeric input, calls `engine.answer(question=original_question, pending_route=resp.pending_route, resolved_entity_id=selected_candidate.id)`.
  - If cancelled (`'c'`, empty, or Ctrl+C), resets `pending_route = None` and returns to prompt.
- **Output Streams & Evidence Provenance:**
  - Answers and evidence provenance (`heading_hierarchy`, `source_attribution`, `chunk_id`, `similarity_score`) written to `stdout`.
  - Spinners, warmups, status banners, and diagnostics written to `stderr`.
  - Windows Virtual Terminal Processing enabled via `ctypes` on Windows conhost.

---

### 4. Package Metadata & Documentation Updates

#### [MODIFY] [`pyproject.toml`](file:///c:/projects/x4-advisor/pyproject.toml)
- Add console script entrypoint:
  ```toml
  [tool.poetry.scripts]
  x4-advisor = "x4_advisor.cli:main"
  ```

#### [MODIFY] [`README.md`](file:///c:/projects/x4-advisor/README.md)
- Updates CLI usage documentation, subcommand syntax (`ask`, `interactive`, `doctor`), observability flags, and documented Layer 1 recall boundaries.

#### [MODIFY] [`AGENTS.md`](file:///c:/projects/x4-advisor/AGENTS.md)
- Updates Milestone M8 description to reflect single-turn REPL and CLI interface, single-turn entity disambiguation resumption, evidence provenance transparency, dataset staleness validation, and pipeline observability.

#### [MODIFY] [`docs/planning/specs/spec-001.md`](file:///c:/projects/x4-advisor/docs/planning/specs/spec-001.md)
- Updates §14 (Milestone M5 Implementation Addendum, line 174) to mark dataset staleness validation DELIVERED in M8.

---

## Verification Plan

### Automated Tests
1. **Unit Tests (`tests/unit/test_cli.py`, `tests/unit/test_diagnostics.py`, `tests/unit/test_ollama_client.py`, `tests/unit/test_router.py`, `tests/unit/test_synthesizer.py`):**
   - Subcommand parsing: bare invocation, `ask`, `interactive`, `doctor`, and `--model` startup override via `parents=[common_parser]`.
   - `http.client` socket shutdown cancellation seam in `OllamaClient`: verify socket shutdown is invoked and `OllamaTimeoutError`, `OllamaModelNotFoundError`, `OllamaCancelledError`, and `OllamaConnectionError` contracts hold.
   - 4-State Ctrl+C state machine: warmup cancel vs in-flight query cancel vs disambiguation cancel vs idle prompt exit (code 130).
   - Disambiguation continuation flow: assert numeric selection invokes `answer(..., pending_route=..., resolved_entity_id=...)` without second router call.
   - REPL command parsing: `/help`, `/doctor`, `/explain`, `/exit`, and `EOFError` handling.
   - `doctor` checks: mock Ollama tags, `/api/ps` VRAM residency ratio (`size_vram` vs `size`), missing database, offline daemon handling (exits 1 without throwing ConfigError), `--db` override diagnostics, and `dataset_metadata` verification with `EXPECTED_SCHEMA_VERSION == "1.1.0"`.
   - Stream separation: verify answer is on stdout, diagnostics on stderr.
   - Exit code checks: `0` on successful `ask` queries (including abstentions), `1` on failed `doctor`, `2` on bad config/args, `130` on idle SIGINT.
   - `--db` override: verifies setting `database_path_str` correctly overrides database path.
   - **Cancellation propagation:** setting `cancel_event` mid-query propagates `OllamaCancelledError` out of `AdvisorEngine.answer()` rather than degrading to an `ABSTAIN` route or a fallback `SynthesisResult` — asserted separately for the router, synthesizer, and warmup paths.
2. **Integration Tests (`tests/integration/test_cli_integration.py`):**
   - End-to-end `doctor` run against live Ollama daemon and real SQLite database.
   - Split latency check: report warmup duration, assert post-warmup first answer latency (<20s single-path).
   - Read-only database assertion (`PRAGMA query_only = ON`).
   - Server-side cancellation test: cancel a heavy vector query / large generation and verify that a follow-up query completes within standard single-path SLA (<20s).
3. **Governance Suite (`tests/unit/test_model_strings_governance.py`):**
   - Assert all tests in suite pass with 0 hardcoded model string violations (all model strings in `cli.py` and `diagnostics.py` sourced dynamically from `config.model_name`).

### Manual Verification
1. Run `poetry run x4-advisor doctor` to inspect complete diagnostic output, VRAM residency, and dataset metadata.
2. Run single-shot: `poetry run x4-advisor ask "What is the cargo capacity of Cerberus Vanguard?"`
3. Run interactive REPL: `poetry run x4-advisor`, trigger disambiguation with `"What is the speed of the Magnetar?"`, select `[1]`, and verify provenance output.
4. Test Ctrl+C during long synthesis: verify query aborts without crashing REPL loop.
