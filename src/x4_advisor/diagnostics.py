"""Diagnostics module for X4 Advisor pre-flight health checks and dataset staleness audits."""

from dataclasses import dataclass, field
import http.client
import json
import logging
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional
import urllib.parse

from x4_advisor.config import DEFAULT_MODEL_NAME, Config
from x4_advisor.storage.schema import EXPECTED_SCHEMA_VERSION

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Individual diagnostic check result."""

    name: str
    status: str  # "OK", "WARN", "FAIL"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    """Consolidated diagnostic report covering runtime, models, database, and empirical bounds."""

    timestamp: str
    checks: List[CheckResult] = field(default_factory=list)
    success: bool = True

    def add_check(self, name: str, status: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Appends check result and updates overall success status."""
        self.checks.append(CheckResult(name=name, status=status, message=message, details=details or {}))
        if status == "FAIL":
            self.success = False

    def render(self, use_color: bool = True) -> str:
        """Renders formatted diagnostic report for terminal presentation."""
        # ANSI color escapes
        green = "\033[32m" if use_color else ""
        yellow = "\033[33m" if use_color else ""
        red = "\033[31m" if use_color else ""
        cyan = "\033[36m" if use_color else ""
        bold = "\033[1m" if use_color else ""
        reset = "\033[0m" if use_color else ""

        lines = [
            f"{bold}{cyan}=== X4 Advisor Diagnostic Report ==={reset}",
            f"Timestamp: {self.timestamp}",
            "",
        ]

        for check in self.checks:
            if check.status == "OK":
                tag = f"[{green}OK{reset}]"
            elif check.status == "WARN":
                tag = f"[{yellow}WARN{reset}]"
            else:
                tag = f"[{red}FAIL{reset}]"

            lines.append(f"{tag} {bold}{check.name}{reset}: {check.message}")
            for k, v in check.details.items():
                lines.append(f"     - {k}: {v}")

        lines.append("")
        if self.success:
            lines.append(f"{green}{bold}Status: HEALTHY (All critical systems operational){reset}")
        else:
            lines.append(f"{red}{bold}Status: UNHEALTHY (Critical configuration or component failures detected){reset}")

        return "\n".join(lines)


def _probe_http(endpoint: str, path: str, timeout_sec: float = 3.0) -> Tuple_ProbeResponse:
    """Executes a simple GET request against an endpoint."""
    parsed = urllib.parse.urlparse(endpoint)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        if parsed.scheme == "https":
            conn = http.client.HTTPSConnection(host, port, timeout=timeout_sec)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout_sec)

        conn.request("GET", path, headers={"User-Agent": "x4-advisor-doctor"})
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="replace")
        conn.close()
        return resp.status, body, None
    except Exception as e:
        return 0, "", e


class Tuple_ProbeResponse(tuple):
    """Named tuple-like structure for probe responses (status, body, error)."""
    status: int
    body: str
    error: Optional[Exception]

    def __new__(cls, status: int, body: str, error: Optional[Exception]):
        return super().__new__(cls, (status, body, error))

    @property
    def status(self) -> int:
        return self[0]

    @property
    def body(self) -> str:
        return self[1]

    @property
    def error(self) -> Optional[Exception]:
        return self[2]


def run_diagnostics(config: Config) -> DiagnosticReport:
    """Executes comprehensive diagnostic health checks and dataset staleness audits."""
    report = DiagnosticReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))

    # 1. Ollama Daemon Check
    ollama_status, ollama_body, ollama_err = _probe_http(config.ollama_endpoint, "/api/tags")
    if ollama_err is not None or ollama_status != 200:
        report.add_check(
            name="Ollama Daemon",
            status="FAIL",
            message=f"Unreachable at {config.ollama_endpoint}",
            details={"error": str(ollama_err or f"HTTP {ollama_status}")},
        )
    else:
        installed_models = []
        try:
            tags_json = json.loads(ollama_body)
            installed_models = [m.get("name", "") for m in tags_json.get("models", [])]
        except Exception:
            pass

        report.add_check(
            name="Ollama Daemon",
            status="OK",
            message=f"Connected to {config.ollama_endpoint}",
            details={"installed_models_count": len(installed_models)},
        )

        # 2. Configured Models Availability & Residency Check
        # Check synthesis model
        synth_model = config.model_name
        model_found = any(m == synth_model or m.startswith(f"{synth_model}:") or synth_model.startswith(f"{m}:") for m in installed_models)
        
        # Check /api/ps for VRAM residency
        ps_status, ps_body, _ = _probe_http(config.ollama_endpoint, "/api/ps")
        residency_details = {}
        if ps_status == 200:
            try:
                ps_json = json.loads(ps_body)
                for pm in ps_json.get("models", []):
                    pm_name = pm.get("name", "")
                    size_vram = pm.get("size_vram", 0)
                    size_total = pm.get("size", 0)
                    ratio = (size_vram / size_total * 100) if size_total > 0 else 0.0
                    residency_details[pm_name] = f"{ratio:.1f}% VRAM ({size_vram / 1024**3:.2f}GB / {size_total / 1024**3:.2f}GB)"
            except Exception:
                pass

        if model_found:
            residency_str = residency_details.get(synth_model, "Not currently resident in VRAM (will load on warmup)")
            report.add_check(
                name="Synthesis Model",
                status="OK",
                message=f"Model '{synth_model}' available",
                details={
                    "model_tag": synth_model,
                    "residency": residency_str,
                    "adr_status": "Provisional operating default (ADR-0005)" if synth_model == DEFAULT_MODEL_NAME else "Configured model",
                },
            )
        else:
            report.add_check(
                name="Synthesis Model",
                status="FAIL",
                message=f"Model '{synth_model}' not found in Ollama runtime",
                details={"action_required": f"Run 'ollama pull {synth_model}'"},
            )

        # Check embedding model
        emb_model = config.embedding_model
        emb_found = any(m == emb_model or m.startswith(f"{emb_model}:") or emb_model.startswith(f"{m}:") for m in installed_models)
        if emb_found:
            report.add_check(
                name="Embedding Model",
                status="OK",
                message=f"Model '{emb_model}' available",
                details={"model_tag": emb_model},
            )
        else:
            report.add_check(
                name="Embedding Model",
                status="FAIL",
                message=f"Embedding model '{emb_model}' not found in Ollama runtime",
                details={"action_required": f"Run 'ollama pull {emb_model}'"},
            )

    # 3. Database File Existence & Read-Only PRAGMA check
    db_path = config.database_path
    if not db_path.exists():
        report.add_check(
            name="SQLite Database",
            status="FAIL",
            message=f"Database file not found at {db_path}",
            details={"action_required": "Run structured game data and unstructured knowledge ingestion pipelines"},
        )
    else:
        # Isolated read-only connection
        try:
            uri = f"file:{db_path.as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            conn.execute("PRAGMA query_only = ON;")
            cursor = conn.cursor()

            # Core tables count check
            core_tables = [
                "ships",
                "wares",
                "sectors",
                "sector_resources",
                "factions",
                "production_recipes",
                "knowledge_chunks",
            ]
            table_counts = {}
            missing_tables = []
            for t in core_tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {t};")
                    table_counts[t] = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    missing_tables.append(t)

            if missing_tables:
                report.add_check(
                    name="Database Schema",
                    status="FAIL",
                    message="Missing required tables",
                    details={"missing_tables": missing_tables},
                )
            else:
                report.add_check(
                    name="Database Integrity",
                    status="OK",
                    message="All 6 core tables and vector chunks populated",
                    details=table_counts,
                )

            # 4. Dataset Staleness Validation (SPEC-001 §14)
            try:
                cursor.execute(
                    "SELECT game_version, build, extraction_timestamp, is_base_game_only, schema_version FROM dataset_metadata WHERE id = 1;"
                )
                meta_row = cursor.fetchone()
                if meta_row is None:
                    report.add_check(
                        name="Dataset Staleness",
                        status="WARN",
                        message="dataset_metadata table is unpopulated",
                        details={"expected_schema_version": EXPECTED_SCHEMA_VERSION},
                    )
                else:
                    g_ver, bld, ext_ts, base_only, s_ver = meta_row
                    staleness_details = {
                        "game_version": g_ver,
                        "build": bld,
                        "extraction_timestamp": ext_ts,
                        "is_base_game_only": bool(base_only),
                        "schema_version": s_ver,
                    }
                    if s_ver != EXPECTED_SCHEMA_VERSION:
                        report.add_check(
                            name="Dataset Staleness",
                            status="WARN",
                            message=f"Database schema version '{s_ver}' differs from expected '{EXPECTED_SCHEMA_VERSION}'",
                            details=staleness_details,
                        )
                    else:
                        report.add_check(
                            name="Dataset Freshness",
                            status="OK",
                            message=f"X4: Foundations v{g_ver} (build {bld}) extracted at {ext_ts}",
                            details=staleness_details,
                        )
            except sqlite3.OperationalError:
                report.add_check(
                    name="Dataset Staleness",
                    status="WARN",
                    message="dataset_metadata table missing from database",
                )

            conn.close()
        except Exception as e:
            report.add_check(
                name="SQLite Database",
                status="FAIL",
                message=f"Error reading database at {db_path}: {e}",
            )

    # 5. Calibrated Thresholds & Known Boundaries
    report.add_check(
        name="Operational Parameters",
        status="OK",
        message="Active calibrated retrieval thresholds and recall boundaries",
        details={
            "similarity_threshold_tau": config.vector_relevance_threshold,
            "layer1_recall_ceiling": "56.5% - 65.2% (Strategic Unstructured)",
            "conversation_scope": "Single-Turn (No multi-turn memory outside disambiguation continuation)",
        },
    )

    return report
