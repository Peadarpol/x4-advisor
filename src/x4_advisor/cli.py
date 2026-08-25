"""Command-line interface for X4 Advisor queries and interactive sessions."""

import argparse
import logging
import sys
from typing import Optional

from x4_advisor.config import ConfigError, get_config
from x4_advisor.retrieval.advisor_engine import AdvisorEngine
from x4_advisor.retrieval.models import AdvisorResponse

logger = logging.getLogger(__name__)


def format_advisor_output(response: AdvisorResponse) -> str:
    """Formats the AdvisorResponse for clean terminal display."""
    output = []
    output.append(f"\n--- X4 Advisor Answer ---")
    if response.synthesis_result:
        output.append(response.synthesis_result.answer_text)
        if response.synthesis_result.notes:
            output.append("\n[Notes]:")
            for note in response.synthesis_result.notes:
                output.append(f"  * {note}")
        if response.synthesis_result.evidence_chunk_ids:
            output.append(
                f"\n[Evidence Sources]: {len(response.synthesis_result.evidence_chunk_ids)} chunk(s) consulted"
            )
    else:
        output.append("No response synthesized.")

    output.append("-------------------------\n")
    return "\n".join(output)


def ask_command(args: argparse.Namespace) -> int:
    """Executes a single natural-language question."""
    config = get_config(validate=False)

    try:
        config.validate_m5_config(probe_ollama=not args.skip_probe)
    except ConfigError as e:
        print(f"\n[Configuration Error]: {e}\n", file=sys.stderr)
        return 1

    try:
        with AdvisorEngine(config=config) as engine:
            if not args.no_warmup:
                print("Warming up models in Ollama (warmup pass)...", file=sys.stderr)
                engine.client.warmup(embedder=engine.embedder)

            response = engine.answer(args.question)
            print(format_advisor_output(response))
            return 0
    except Exception as e:
        print(f"\n[Error]: {e}\n", file=sys.stderr)
        return 1


def interactive_command(args: argparse.Namespace) -> int:
    """Runs a single-turn interactive query session."""
    config = get_config(validate=False)

    try:
        config.validate_m5_config(probe_ollama=not args.skip_probe)
    except ConfigError as e:
        print(f"\n[Configuration Error]: {e}\n", file=sys.stderr)
        return 1

    print("=================================================")
    print("       X4 Advisor — Interactive Session          ")
    print("  (Single-turn query advisor for base-game X4)   ")
    print("  Type 'exit', 'quit', or 'q' to stop.           ")
    print("=================================================\n")

    try:
        with AdvisorEngine(config=config) as engine:
            if not args.no_warmup:
                print("Warming up models in Ollama...", file=sys.stderr)
                engine.client.warmup(embedder=engine.embedder)

            pending_route = None

            while True:
                try:
                    prompt_label = "Select entity candidate (ID or name)> " if pending_route else "x4-advisor> "
                    user_input = input(prompt_label).strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ("exit", "quit", "q"):
                        break

                    if pending_route:
                        # Resuming from disambiguation
                        response = engine.answer(
                            question="clarification",
                            pending_route=pending_route,
                            resolved_entity_id=user_input,
                        )
                        pending_route = None
                    else:
                        response = engine.answer(user_input)

                    print(format_advisor_output(response))

                    if response.ambiguous_candidates and response.pending_route:
                        pending_route = response.pending_route

                except KeyboardInterrupt:
                    print("\nExiting session...")
                    break
                except Exception as e:
                    print(f"[Error processing query]: {e}\n")
                    pending_route = None

            return 0
    except Exception as e:
        print(f"\n[Engine Error]: {e}\n", file=sys.stderr)
        return 1


def main() -> None:
    """Main CLI argument parser and entry point."""
    parser = argparse.ArgumentParser(
        prog="x4-advisor",
        description="X4 Advisor: AI-powered advisory system for X4: Foundations.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", type=str, help="Natural language question")
    ask_parser.add_argument("--skip-probe", action="store_true", help="Skip Ollama reachability probe")
    ask_parser.add_argument("--no-warmup", action="store_true", help="Skip model pre-warmup call")

    # Interactive command
    inter_parser = subparsers.add_parser("interactive", help="Start interactive single-turn session")
    inter_parser.add_argument("--skip-probe", action="store_true", help="Skip Ollama reachability probe")
    inter_parser.add_argument("--no-warmup", action="store_true", help="Skip model pre-warmup call")

    args = parser.parse_args()

    if args.command == "ask":
        sys.exit(ask_command(args))
    elif args.command == "interactive":
        sys.exit(interactive_command(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
