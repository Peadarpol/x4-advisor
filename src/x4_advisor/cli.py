"""Command-line interface for X4 Advisor queries, diagnostics, and interactive sessions."""

import argparse
import ctypes
import os
import sys
import time
from typing import List, Optional

from x4_advisor.config import Config, ConfigError, get_config
from x4_advisor.diagnostics import run_diagnostics
from x4_advisor.llm.client import OllamaCancelledError
from x4_advisor.retrieval.advisor_engine import AdvisorEngine
from x4_advisor.retrieval.models import AbstainReason, AdvisorResponse, RouteType


def _init_windows_vt() -> None:
    """Enables virtual terminal processing on Windows conhost for ANSI color support."""
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass


def _should_use_color(args: argparse.Namespace) -> bool:
    """Determines whether ANSI color codes should be emitted."""
    if getattr(args, "no_color", False):
        return False
    if os.getenv("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def format_advisor_output(
    response: AdvisorResponse,
    explain: bool = False,
    use_color: bool = True,
) -> str:
    """Formats the AdvisorResponse with evidence provenance, badges, and optional telemetry."""
    # Color formatting
    cyan = "\033[36m" if use_color else ""
    green = "\033[32m" if use_color else ""
    yellow = "\033[33m" if use_color else ""
    magenta = "\033[35m" if use_color else ""
    bold = "\033[1m" if use_color else ""
    reset = "\033[0m" if use_color else ""

    output: List[str] = []
    output.append(f"\n{bold}{cyan}--- X4 Advisor Answer ---{reset}")

    # Check for Abstention Badge
    abstain_reason = None
    if response.route_result and response.route_result.abstain_reason:
        abstain_reason = response.route_result.abstain_reason
    elif response.synthesis_result and response.synthesis_result.abstain_reason:
        abstain_reason = response.synthesis_result.abstain_reason

    if abstain_reason:
        reason_str = abstain_reason.value if hasattr(abstain_reason, "value") else str(abstain_reason)
        output.append(f"{bold}{yellow}[ABSTAIN: {reason_str.upper()}]{reset}")

    if response.synthesis_result:
        output.append(response.synthesis_result.answer_text)

        if response.synthesis_result.notes:
            output.append(f"\n{bold}[Operational Notes]:{reset}")
            for note in response.synthesis_result.notes:
                output.append(f"  * {note}")

        # Evidence Provenance Attribution
        if response.synthesis_result.evidence_chunk_ids:
            output.append(f"\n{bold}[Evidence Sources]:{reset}")
            # Map chunk IDs to chunk metadata from vector result
            chunk_map = {}
            if response.vector_result and response.vector_result.chunks:
                for c in response.vector_result.chunks:
                    chunk_map[c.chunk_id] = c

            for cid in response.synthesis_result.evidence_chunk_ids:
                chunk = chunk_map.get(cid)
                if chunk:
                    if isinstance(chunk.heading_hierarchy, list):
                        title = " > ".join(chunk.heading_hierarchy)
                    elif chunk.heading_hierarchy:
                        title = str(chunk.heading_hierarchy)
                    else:
                        title = chunk.chunk_id

                    source = chunk.source_attribution or "Community Guide"
                    score_str = f"(similarity: {chunk.similarity_score:.3f})" if chunk.similarity_score else ""
                    if source and source.strip().lower() not in title.lower():
                        output.append(f"  * {bold}{title}{reset} — {source} {score_str}")
                    else:
                        output.append(f"  * {bold}{title}{reset} {score_str}")
                else:
                    output.append(f"  * Chunk ID: {cid}")

    elif response.ambiguous_candidates:
        pass  # Handled interactively by caller
    else:
        output.append("No response synthesized.")

    # Verbose Observability Telemetry (--explain / -v)
    if explain:
        output.append(f"\n{bold}{magenta}--- Pipeline Telemetry & Execution Trace ---{reset}")
        if response.route_result:
            output.append(f"  * Route Type: {response.route_result.route_type.value}")
            if response.route_result.tool_calls:
                for tc in response.route_result.tool_calls:
                    output.append(f"  * Tool Call: {tc.name}({tc.arguments})")
        if response.vector_result:
            output.append(f"  * Vector Candidates: {response.vector_result.total_candidates} (filtered with threshold {response.vector_result.threshold_used})")
        if response.structured_result:
            output.append(f"  * Structured Result: {type(response.structured_result).__name__}")
        if getattr(response, "execution_durations", None):
            output.append("  * Latency Breakdown:")
            for stage, dur in response.execution_durations.items():
                output.append(f"      - {stage}: {dur*1000:.1f}ms")
        if getattr(response, "total_duration_seconds", None):
            output.append(f"  * Total Latency: {response.total_duration_seconds:.2f}s")

    output.append(f"{cyan}-------------------------{reset}\n")
    return "\n".join(output)


def handle_disambiguation(
    engine: AdvisorEngine,
    original_question: str,
    response: AdvisorResponse,
    explain: bool = False,
    use_color: bool = True,
) -> Optional[AdvisorResponse]:
    """Renders candidate entities and prompts user for numeric disambiguation selection."""
    candidates = response.ambiguous_candidates
    if not candidates or not response.pending_route:
        return None

    bold = "\033[1m" if use_color else ""
    reset = "\033[0m" if use_color else ""

    print(f"\n{bold}Multiple entities matched your query:{reset}", file=sys.stderr)
    for idx, c in enumerate(candidates, start=1):
        print(f"  [{idx}] {c.name} (ID: {c.id}, Type: {c.entity_type})", file=sys.stderr)

    while True:
        try:
            choice = input(f"Select entity [1-{len(candidates)}] or 'c' to cancel: ").strip()
            if not choice or choice.lower() == "c":
                print("[Selection cancelled]", file=sys.stderr)
                return None

            try:
                sel_num = int(choice)
                if 1 <= sel_num <= len(candidates):
                    selected = candidates[sel_num - 1]
                    # Direct M5 resumption: bypass router, pass pending_route and canonical ID
                    res = engine.answer(
                        question=original_question,
                        pending_route=response.pending_route,
                        resolved_entity_id=selected.id,
                    )
                    return res
                else:
                    print(f"Please enter a number between 1 and {len(candidates)}, or 'c' to cancel.", file=sys.stderr)
            except ValueError:
                print("Invalid input. Please enter a number or 'c' to cancel.", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n[Selection cancelled]", file=sys.stderr)
            return None


def ask_command(args: argparse.Namespace) -> int:
    """Executes a single natural-language question in non-interactive mode."""
    _init_windows_vt()
    use_color = _should_use_color(args)

    config = get_config(validate=False)
    if args.model:
        config.model_name = args.model
    if args.db:
        config.database_path_str = str(args.db)

    try:
        config.validate_m5_config(probe_ollama=not args.skip_probe)
    except ConfigError as e:
        print(f"\n[Configuration Error]: {e}\n", file=sys.stderr)
        return 2

    question = " ".join(args.question) if isinstance(args.question, list) else str(args.question)
    if not question.strip():
        print("Error: No question provided.", file=sys.stderr)
        return 2

    try:
        with AdvisorEngine(config=config) as engine:
            if not args.no_warmup:
                try:
                    print("Warming up models in Ollama (warmup pass)...", file=sys.stderr)
                    engine.client.warmup(embedder=engine.embedder)
                except (OllamaCancelledError, KeyboardInterrupt):
                    print("[Warmup skipped by user — first query may incur cold-start latency]", file=sys.stderr)

            try:
                response = engine.answer(question)
                if response.ambiguous_candidates and response.pending_route:
                    resolved_resp = handle_disambiguation(
                        engine=engine,
                        original_question=question,
                        response=response,
                        explain=args.explain,
                        use_color=use_color,
                    )
                    if resolved_resp:
                        print(format_advisor_output(resolved_resp, explain=args.explain, use_color=use_color))
                    return 0

                print(format_advisor_output(response, explain=args.explain, use_color=use_color))
                return 0
            except (OllamaCancelledError, KeyboardInterrupt):
                print("\n[Query cancelled by user]", file=sys.stderr)
                return 130
    except Exception as e:
        print(f"\n[Error]: {e}\n", file=sys.stderr)
        return 1


def interactive_command(args: argparse.Namespace) -> int:
    """Runs an interactive single-turn REPL session."""
    _init_windows_vt()
    use_color = _should_use_color(args)
    explain_mode = args.explain

    config = get_config(validate=False)
    if args.model:
        config.model_name = args.model
    if args.db:
        config.database_path_str = str(args.db)

    try:
        config.validate_m5_config(probe_ollama=not args.skip_probe)
    except ConfigError as e:
        print(f"\n[Configuration Error]: {e}\n", file=sys.stderr)
        return 2

    print("=================================================", file=sys.stderr)
    print("       X4 Advisor — Interactive Session          ", file=sys.stderr)
    print("  (Single-turn query advisor for base-game X4)   ", file=sys.stderr)
    print("  Commands: /help, /doctor, /explain, /exit       ", file=sys.stderr)
    print("=================================================\n", file=sys.stderr)

    try:
        with AdvisorEngine(config=config) as engine:
            if not args.no_warmup:
                try:
                    print("Warming up models in Ollama...", file=sys.stderr)
                    engine.client.warmup(embedder=engine.embedder)
                except (OllamaCancelledError, KeyboardInterrupt):
                    print("[Warmup skipped by user — first query may incur cold-start latency]", file=sys.stderr)

            while True:
                try:
                    user_input = input("x4-advisor> ").strip()
                except EOFError:
                    print("\nExiting session...", file=sys.stderr)
                    return 0
                except KeyboardInterrupt:
                    # Idle prompt Ctrl+C exits session
                    print("\nExiting session...", file=sys.stderr)
                    return 130

                if not user_input:
                    continue

                # REPL Commands
                lower_cmd = user_input.lower()
                if lower_cmd in ("/exit", "exit", "quit", "q"):
                    print("Goodbye!", file=sys.stderr)
                    return 0

                if lower_cmd in ("/help", "help"):
                    print("\nCommands:", file=sys.stderr)
                    print("  /help    - Show this help message", file=sys.stderr)
                    print("  /doctor  - Run environment and dataset diagnostics", file=sys.stderr)
                    print("  /explain - Toggle verbose query telemetry", file=sys.stderr)
                    print("  /exit    - Exit the interactive session\n", file=sys.stderr)
                    continue

                if lower_cmd in ("/explain", "explain"):
                    explain_mode = not explain_mode
                    state_str = "ENABLED" if explain_mode else "DISABLED"
                    print(f"[Verbose Telemetry {state_str}]\n", file=sys.stderr)
                    continue

                if lower_cmd in ("/doctor", "doctor"):
                    diag_report = run_diagnostics(config)
                    print(diag_report.render(use_color=use_color), file=sys.stderr)
                    continue

                # Execute Query
                try:
                    response = engine.answer(user_input)
                    if response.ambiguous_candidates and response.pending_route:
                        resolved_resp = handle_disambiguation(
                            engine=engine,
                            original_question=user_input,
                            response=response,
                            explain=explain_mode,
                            use_color=use_color,
                        )
                        if resolved_resp:
                            print(format_advisor_output(resolved_resp, explain=explain_mode, use_color=use_color))
                        continue

                    print(format_advisor_output(response, explain=explain_mode, use_color=use_color))

                except (OllamaCancelledError, KeyboardInterrupt):
                    print("\n[Query cancelled by user]\n", file=sys.stderr)

    except Exception as e:
        print(f"\n[Engine Error]: {e}\n", file=sys.stderr)
        return 1


def doctor_command(args: argparse.Namespace) -> int:
    """Runs pre-flight diagnostics and dataset staleness audits."""
    _init_windows_vt()
    use_color = _should_use_color(args)

    config = get_config(validate=False)
    if args.model:
        config.model_name = args.model
    if args.db:
        config.database_path_str = str(args.db)

    report = run_diagnostics(config)
    print(report.render(use_color=use_color))
    return 0 if report.success else 1


def build_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser with shared parent options."""
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--model", type=str, help="Override active synthesis model tag")
    common_parser.add_argument("--explain", "-v", action="store_true", help="Display verbose routing and execution telemetry")
    common_parser.add_argument("--no-warmup", action="store_true", help="Skip pre-query model warmup pass")
    common_parser.add_argument("--skip-probe", action="store_true", help="Skip Ollama reachability check during startup validation")
    common_parser.add_argument("--db", type=str, help="Override database path")
    common_parser.add_argument("--no-color", action="store_true", help="Disable ANSI color codes in terminal output")

    parser = argparse.ArgumentParser(
        prog="x4-advisor",
        description="X4 Advisor: AI-powered advisory system for X4: Foundations.",
        parents=[common_parser],
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # ask subcommand
    ask_parser = subparsers.add_parser("ask", help="Ask a single question", parents=[common_parser])
    ask_parser.add_argument("question", nargs="+", help="Natural language question")

    # interactive subcommand
    subparsers.add_parser("interactive", help="Start interactive single-turn REPL session", parents=[common_parser])

    # doctor subcommand
    doctor_parser = subparsers.add_parser("doctor", help="Run environment and dataset diagnostics", parents=[common_parser])

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ask":
        sys.exit(ask_command(args))
    elif args.command == "doctor":
        sys.exit(doctor_command(args))
    elif args.command == "interactive" or args.command is None:
        sys.exit(interactive_command(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
