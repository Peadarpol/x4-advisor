# ADR 0006 — Game Data Extraction Tool: x4cat

**Status:** Accepted

## Context

Structured game data (ships, wares, production recipes, factions, sectors) needs extracting from X4's `.cat`/`.dat` archives. Three tools were compared: Egosoft's own **X Catalog Tool**, the community tool **X4FProjector**, and the community tool **x4cat**.

## Decision

**`x4cat`** is the primary extraction/indexing dependency. **X Catalog Tool** is kept as a reference extractor for troubleshooting format questions, not a pipeline dependency. **X4FProjector** is used only as an independent cross-check oracle during schema development — comparing its parsed output against this project's own parser is a cheap way to catch parsing bugs — but is not itself a runtime dependency.

## Rationale

Verified directly against the repository: `x4cat` is MIT-licensed, actively released (confirmed current version, not abandoned), and its own documentation states it's tested against current X4 game versions — the strongest current-maintenance signal of the three. X4FProjector has genuinely excellent semantic coverage of exactly the categories this project needs, but its maintenance signals (low commit count, no confirmed current-version compatibility) don't clear the bar for a foundational dependency. X Catalog Tool is official but its most recent listed release predates active current development, making it a safe reference point rather than a convenient pipeline foundation (it does no semantic parsing — output is raw archive contents).

This project's own domain schema and normalization layer stay independent of `x4cat`'s internal data model regardless of which tool is used — `x4cat` is a way to get raw game data out, not a source of this project's own schema decisions.

## Consequences

- A dependency on an actively-maintained but small community project. Mitigation: golden extraction fixtures (a handful of known ship/ware/recipe records with expected output) so an `x4cat` version bump is tested against known-good results rather than trusted blindly.
- DLC/base-game separation is enforced by this project's own ingestion logic (root-archive-only extraction), not delegated to any of the three tools as a policy engine