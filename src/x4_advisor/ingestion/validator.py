"""Domain invariant validation engine for extracted records."""

import logging
from typing import Dict, List, Set, Tuple

from x4_advisor.storage.models import (
    FactionRecord,
    ProductionRecipeRecord,
    SectorRecord,
    SectorResourceRecord,
    ShipRecord,
    WareRecord,
)

logger = logging.getLogger(__name__)

PROVISIONAL_SHIP_CLASSES: Set[str] = {
    "ship_xs",
    "ship_s",
    "ship_m",
    "ship_l",
    "ship_xl",
    "station",
}


class ValidationReport:
    """Ingestion validation report capturing overall and record-level metrics."""

    def __init__(self) -> None:
        self.total_processed: int = 0
        self.total_valid: int = 0
        self.total_skipped: int = 0
        self.raw_counts: Dict[str, int] = {}
        self.valid_counts: Dict[str, int] = {}
        self.warnings: List[str] = []
        self.skipped_records: List[str] = []

    def log_warning(self, msg: str) -> None:
        """Logs a warning entry."""
        self.warnings.append(msg)
        logger.warning(msg)

    def log_skip(self, record_desc: str, reason: str) -> None:
        """Logs a skipped record with reason."""
        self.total_skipped += 1
        msg = f"Skipped {record_desc}: {reason}"
        self.skipped_records.append(msg)
        logger.warning(msg)


class DomainValidator:
    """Validates domain invariants on extracted records according to SPEC-001 §2."""

    def validate_dataset(
        self,
        factions: List[FactionRecord],
        wares: List[WareRecord],
        sectors: List[SectorRecord],
        sector_resources: List[SectorResourceRecord],
        ships: List[ShipRecord],
        recipes: List[ProductionRecipeRecord],
    ) -> Tuple[
        List[FactionRecord],
        List[WareRecord],
        List[SectorRecord],
        List[SectorResourceRecord],
        List[ShipRecord],
        List[ProductionRecipeRecord],
        ValidationReport,
    ]:
        """Validates all batch records, returning valid records and a ValidationReport."""
        report = ValidationReport()

        valid_factions: List[FactionRecord] = []
        faction_ids: Set[str] = set()

        # 1. Validate Factions
        for f in factions:
            report.total_processed += 1
            if not f.id or not f.name:
                report.log_skip(f"Faction({f.id})", "Missing required id or name")
                continue
            if f.id in faction_ids:
                report.log_skip(f"Faction({f.id})", "Duplicate primary key ID")
                continue
            faction_ids.add(f.id)
            valid_factions.append(f)
            report.total_valid += 1

        # 2. Validate Wares
        valid_wares: List[WareRecord] = []
        ware_ids: Set[str] = set()

        for w in wares:
            report.total_processed += 1
            if not w.id or not w.name:
                report.log_skip(f"Ware({w.id})", "Missing required id or name")
                continue
            if w.id in ware_ids:
                report.log_skip(f"Ware({w.id})", "Duplicate primary key ID")
                continue
            if w.min_price > w.avg_price or w.avg_price > w.max_price or w.min_price < 0:
                report.log_skip(
                    f"Ware({w.id})",
                    f"Invalid price bounds (min={w.min_price}, avg={w.avg_price}, max={w.max_price})",
                )
                continue
            if w.volume <= 0:
                report.log_skip(f"Ware({w.id})", f"Volume ({w.volume}) must be > 0")
                continue
            ware_ids.add(w.id)
            valid_wares.append(w)
            report.total_valid += 1

        # 3. Validate Sectors
        valid_sectors: List[SectorRecord] = []
        sector_ids: Set[str] = set()

        for s in sectors:
            report.total_processed += 1
            if not s.id or not s.name:
                report.log_skip(f"Sector({s.id})", "Missing required id or name")
                continue
            if s.id in sector_ids:
                report.log_skip(f"Sector({s.id})", "Duplicate primary key ID")
                continue
            if s.sunlight < 0:
                report.log_skip(f"Sector({s.id})", f"Sunlight ({s.sunlight}) must be >= 0")
                continue
            if s.faction_id and s.faction_id not in faction_ids:
                report.log_skip(
                    f"Sector({s.id})",
                    f"Referenced faction_id '{s.faction_id}' does not exist in factions",
                )
                continue
            sector_ids.add(s.id)
            valid_sectors.append(s)
            report.total_valid += 1

        # 4. Validate Sector Resources
        valid_sector_resources: List[SectorResourceRecord] = []
        resource_keys: Set[Tuple[str, str]] = set()

        for sr in sector_resources:
            report.total_processed += 1
            if not sr.sector_id or not sr.resource_id:
                report.log_skip(
                    f"SectorResource({sr.sector_id}, {sr.resource_id})",
                    "Missing sector_id or resource_id",
                )
                continue
            key = (sr.sector_id, sr.resource_id)
            if key in resource_keys:
                report.log_skip(f"SectorResource{key}", "Duplicate primary key")
                continue
            if sr.resource_yield < 0:
                report.log_skip(f"SectorResource{key}", f"Yield ({sr.resource_yield}) must be >= 0")
                continue
            if sr.sector_id not in sector_ids:
                report.log_skip(
                    f"SectorResource{key}",
                    f"Referenced sector_id '{sr.sector_id}' does not exist in sectors",
                )
                continue
            resource_keys.add(key)
            valid_sector_resources.append(sr)
            report.total_valid += 1

        # 5. Validate Ships
        valid_ships: List[ShipRecord] = []
        ship_ids: Set[str] = set()

        for sh in ships:
            report.total_processed += 1
            if not sh.id or not sh.name:
                report.log_skip(f"Ship({sh.id})", "Missing required id or name")
                continue
            if sh.id in ship_ids:
                report.log_skip(f"Ship({sh.id})", "Duplicate primary key ID")
                continue
            if sh.hull <= 0:
                report.log_skip(f"Ship({sh.id})", f"Hull ({sh.hull}) must be > 0")
                continue
            if sh.shields < 0 or sh.cargo_capacity < 0 or sh.speed < 0:
                report.log_skip(
                    f"Ship({sh.id})",
                    f"Negative shields, cargo, or speed (shields={sh.shields}, cargo={sh.cargo_capacity}, speed={sh.speed})",
                )
                continue
            if sh.ship_class != "station" and sh.speed == 0:
                report.log_skip(
                    f"Ship({sh.id})",
                    f"Speed must be > 0 for non-station ships (class={sh.ship_class}, speed={sh.speed})",
                )
                continue
            if sh.ship_class not in PROVISIONAL_SHIP_CLASSES:
                report.log_warning(
                    f"Ship({sh.id}) has unverified class '{sh.ship_class}'. Record will be inserted for enum refinement."
                )
            if sh.faction_id and sh.faction_id not in faction_ids:
                report.log_skip(
                    f"Ship({sh.id})",
                    f"Referenced faction_id '{sh.faction_id}' does not exist in factions",
                )
                continue
            if sh.ware_id and sh.ware_id not in ware_ids:
                report.log_skip(
                    f"Ship({sh.id})",
                    f"Referenced ware_id '{sh.ware_id}' does not exist in wares",
                )
                continue

            ship_ids.add(sh.id)
            valid_ships.append(sh)
            report.total_valid += 1

        # 6. Validate Production Recipes
        valid_recipes: List[ProductionRecipeRecord] = []
        recipe_method_map: Dict[Tuple[str, str], Tuple[int, float]] = {}

        for r in recipes:
            report.total_processed += 1
            if not r.ware_id or not r.input_ware_id:
                report.log_skip(f"Recipe({r.ware_id}, {r.method})", "Missing ware_id or input_ware_id")
                continue
            if r.ware_id == r.input_ware_id:
                report.log_skip(f"Recipe({r.ware_id}, {r.method})", "Self-referencing recipe")
                continue
            if r.input_amount <= 0 or r.output_amount <= 0 or r.production_time <= 0:
                report.log_skip(
                    f"Recipe({r.ware_id}, {r.method})",
                    f"Quantities or time must be > 0 (input={r.input_amount}, output={r.output_amount}, time={r.production_time})",
                )
                continue
            if r.ware_id not in ware_ids:
                report.log_skip(
                    f"Recipe({r.ware_id}, {r.method})",
                    f"Referenced output ware_id '{r.ware_id}' does not exist in wares",
                )
                continue
            if r.input_ware_id not in ware_ids:
                report.log_skip(
                    f"Recipe({r.ware_id}, {r.method})",
                    f"Referenced input_ware_id '{r.input_ware_id}' does not exist in wares",
                )
                continue

            # Recipe consistency check across input rows sharing (ware_id, method)
            recipe_key = (r.ware_id, r.method)
            if recipe_key in recipe_method_map:
                prev_output, prev_time = recipe_method_map[recipe_key]
                if prev_output != r.output_amount or prev_time != r.production_time:
                    report.log_skip(
                        f"Recipe({r.ware_id}, {r.method})",
                        f"Inconsistent output_amount/time across input rows (expected {prev_output}/{prev_time}, got {r.output_amount}/{r.production_time})",
                    )
                    continue
            else:
                recipe_method_map[recipe_key] = (r.output_amount, r.production_time)

            valid_recipes.append(r)
            report.total_valid += 1

        report.raw_counts = {
            "factions": len(factions),
            "wares": len(wares),
            "sectors": len(sectors),
            "sector_resources": len(sector_resources),
            "ships": len(ships),
            "recipes": len(recipes),
        }
        report.valid_counts = {
            "factions": len(valid_factions),
            "wares": len(valid_wares),
            "sectors": len(valid_sectors),
            "sector_resources": len(valid_sector_resources),
            "ships": len(valid_ships),
            "recipes": len(valid_recipes),
        }

        return (
            valid_factions,
            valid_wares,
            valid_sectors,
            valid_sector_resources,
            valid_ships,
            valid_recipes,
            report,
        )
