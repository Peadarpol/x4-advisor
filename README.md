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
*(TODO — documents required models, tags, approximate disk footprint, runtime VRAM expectations, and how to obtain them via Ollama:*
*Embeddings: `qwen3-embedding:0.6b` (~640 MB disk, ~0.6 GB VRAM) — see ADR-0001.*
*LLM: candidate model winning the Phase 1 empirical bake-off (e.g. `gemma4:12b`, `granite4.1:8b`, `qwen3:14b`) — see ADR-0005.)*

## Installation
*(TODO — after M1–M7 exist to actually describe)*

## Configuring your X4 installation
*(TODO)*

## Building the local knowledge base
*(TODO — describe the ingestion pipeline once it's real)*

## Running the advisor
*(TODO)*

## Running the evaluation suite
*(TODO)*

## Licenses
*(TODO — code is MIT; note that no third-party game data or community content is distributed with this repository)*

## Learning objectives
*(TODO — this project exists partly to learn RAG/agent architecture firsthand; several design choices, like avoiding a RAG framework, are deliberate for that reason — see the ADRs)*
