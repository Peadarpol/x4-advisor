# Contributing

This is a personal learning project first, and a shareable tool second — see `docs/planning/project-charter.md` for the full reasoning behind that priority order.

If you're considering a contribution:

- Read `docs/planning/project-charter.md` and `docs/planning/scope-boundary.md` first — they define what this project is and, just as importantly, what it deliberately isn't.
- Architectural changes require an ADR (`docs/adr/`) before implementation, not after.
- Behavioral changes require a spec/test update, not just a code change.
- Don't commit generated or third-party data — extracted game data, curated knowledge base content, save files, or anything else under `data/` is never part of this repository (see the redistribution principle in the charter).
- Given the project's learning objective, some choices that look like "reinventing the wheel" (no RAG framework, custom retrieval) are deliberate, not oversights — see the relevant ADR before proposing a framework swap.

There's no formal review SLA here — this is a solo project, and contributions are reviewed as time allows.
