"""Normalization adapter mapping extracted XML structures into domain records."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

from x4_advisor.ingestion.text_resolver import TextResolver
from x4_advisor.storage.models import (
    FactionRecord,
    ProductionRecipeRecord,
    SectorRecord,
    SectorResourceRecord,
    ShipRecord,
    WareRecord,
)

logger = logging.getLogger(__name__)


class NormalizationAdapter:
    """Decoupled normalization adapter translating raw X4 XML files into domain records."""

    def __init__(self, text_resolver: TextResolver):
        self.text_resolver = text_resolver

    def parse_wares_and_recipes(
        self, wares_xml_path: Path
    ) -> Tuple[List[WareRecord], List[ProductionRecipeRecord], Dict[str, str]]:
        """Parses wares.xml into WareRecords, ProductionRecipeRecords, and a macro_to_ware_id mapping."""
        wares: List[WareRecord] = []
        recipes: List[ProductionRecipeRecord] = []
        macro_to_ware_id: Dict[str, str] = {}

        if not wares_xml_path.exists():
            logger.warning(f"Wares XML path '{wares_xml_path}' does not exist.")
            return wares, recipes, macro_to_ware_id

        try:
            tree = ET.parse(str(wares_xml_path))
            root = tree.getroot()

            for ware_elem in root.findall("ware"):
                ware_id = ware_elem.attrib.get("id")
                if not ware_id:
                    continue

                raw_name = ware_elem.attrib.get("name", ware_id)
                name = self.text_resolver.resolve(raw_name, default=ware_id)
                category = ware_elem.attrib.get("group", "general")

                try:
                    volume = int(ware_elem.attrib.get("volume", "1"))
                except ValueError:
                    volume = 1

                # Prices
                price_elem = ware_elem.find("price")
                if price_elem is not None:
                    try:
                        min_price = int(price_elem.attrib.get("min", "0"))
                        avg_price = int(price_elem.attrib.get("average", "0"))
                        max_price = int(price_elem.attrib.get("max", "0"))
                    except ValueError:
                        min_price = avg_price = max_price = 0
                else:
                    min_price = avg_price = max_price = 0

                ware_rec = WareRecord(
                    id=ware_id,
                    name=name,
                    category=category,
                    min_price=min_price,
                    avg_price=avg_price,
                    max_price=max_price,
                    volume=volume,
                )
                wares.append(ware_rec)

                # Link ship macro component if present
                component_elem = ware_elem.find("component")
                if component_elem is not None:
                    macro_ref = component_elem.attrib.get("ref")
                    if macro_ref:
                        macro_to_ware_id[macro_ref] = ware_id

                # Production recipes
                for prod_elem in ware_elem.findall("production"):
                    method = prod_elem.attrib.get("method", "default")
                    try:
                        production_time = float(prod_elem.attrib.get("time", "0"))
                        output_amount = int(prod_elem.attrib.get("amount", "1"))
                    except ValueError:
                        production_time = 0.0
                        output_amount = 1

                    primary_elem = prod_elem.find("primary")
                    if primary_elem is not None:
                        for input_ware_elem in primary_elem.findall("ware"):
                            input_ware_id = input_ware_elem.attrib.get("ware")
                            if not input_ware_id:
                                continue

                            try:
                                input_amount = int(input_ware_elem.attrib.get("amount", "1"))
                            except ValueError:
                                input_amount = 1

                            recipe_rec = ProductionRecipeRecord(
                                ware_id=ware_id,
                                method=method,
                                input_ware_id=input_ware_id,
                                input_amount=input_amount,
                                output_amount=output_amount,
                                production_time=production_time,
                            )
                            recipes.append(recipe_rec)

        except Exception as e:
            logger.error(f"Failed parsing wares XML '{wares_xml_path}': {e}")

        return wares, recipes, macro_to_ware_id

    def parse_factions(self, factions_xml_path: Path) -> List[FactionRecord]:
        """Parses factions.xml into FactionRecords."""
        factions: List[FactionRecord] = []
        if not factions_xml_path.exists():
            logger.warning(f"Factions XML path '{factions_xml_path}' does not exist.")
            return factions

        try:
            tree = ET.parse(str(factions_xml_path))
            root = tree.getroot()

            for fact_elem in root.findall("faction"):
                faction_id = fact_elem.attrib.get("id")
                if not faction_id:
                    continue

                raw_name = fact_elem.attrib.get("name", faction_id)
                name = self.text_resolver.resolve(raw_name, default=faction_id)

                raw_short = fact_elem.attrib.get("shortname")
                short_name = (
                    self.text_resolver.resolve(raw_short, default=name)
                    if raw_short
                    else None
                )

                description = fact_elem.attrib.get("description")
                relations_summary = (
                    self.text_resolver.resolve(description) if description else None
                )

                factions.append(
                    FactionRecord(
                        id=faction_id,
                        name=name,
                        short_name=short_name,
                        relations_summary=relations_summary,
                    )
                )

        except Exception as e:
            logger.error(f"Failed parsing factions XML '{factions_xml_path}': {e}")

        return factions

    def parse_sectors(
        self, mapdefaults_xml_path: Path
    ) -> Tuple[List[SectorRecord], List[SectorResourceRecord]]:
        """Parses mapdefaults.xml / sectors into SectorRecords and SectorResourceRecords."""
        sectors: List[SectorRecord] = []
        sector_resources: List[SectorResourceRecord] = []

        if not mapdefaults_xml_path.exists():
            logger.warning(f"Map defaults XML path '{mapdefaults_xml_path}' does not exist.")
            return sectors, sector_resources

        try:
            tree = ET.parse(str(mapdefaults_xml_path))
            root = tree.getroot()

            known_resources = ["ore", "silicon", "hydrogen", "ice", "helium", "methane", "nividium"]

            # Iterate over sector dataset entries in mapdefaults
            for dataset_elem in root.findall(".//dataset"):
                macro_name = dataset_elem.attrib.get("macro")
                if not macro_name or "sector" not in macro_name.lower():
                    continue

                sector_id = macro_name

                # Name lookup: properties/identification/@name or dataset/@name
                name_elem = dataset_elem.find("properties/identification")
                raw_name = (
                    name_elem.attrib.get("name")
                    if name_elem is not None and "name" in name_elem.attrib
                    else dataset_elem.attrib.get("name", sector_id)
                )
                name = self.text_resolver.resolve(raw_name, default=sector_id)

                # Faction lookup: dataset/@owner or properties/ownership/@faction
                faction_id = dataset_elem.attrib.get("owner")
                if not faction_id:
                    owner_elem = dataset_elem.find("properties/ownership")
                    if owner_elem is not None:
                        faction_id = owner_elem.attrib.get("faction")

                # Sunlight lookup: properties/area/@sunlight or dataset/@sunlight
                sunlight = 1.0
                area_elem = dataset_elem.find("properties/area")
                if area_elem is not None and "sunlight" in area_elem.attrib:
                    try:
                        sunlight = float(area_elem.attrib.get("sunlight", "1.0"))
                    except ValueError:
                        sunlight = 1.0
                elif "sunlight" in dataset_elem.attrib:
                    try:
                        sunlight = float(dataset_elem.attrib.get("sunlight", "1.0"))
                    except ValueError:
                        sunlight = 1.0

                sectors.append(
                    SectorRecord(
                        id=sector_id,
                        name=name,
                        faction_id=faction_id,
                        sunlight=sunlight,
                    )
                )

                # Parse resources (supports resourceareas format and legacy resources format)
                accumulated_yields: Dict[str, float] = {}

                # 1. New resourceareas format (<resourceareas><resourcearea amount="3" ref="sphere_large_hydrogen_high_slow" />)
                for res_area in dataset_elem.findall(".//resourceareas/resourcearea"):
                    ref = res_area.attrib.get("ref", "").lower()
                    try:
                        amount = float(res_area.attrib.get("amount", "1.0"))
                    except ValueError:
                        amount = 1.0

                    for r_name in known_resources:
                        if r_name in ref:
                            accumulated_yields[r_name] = accumulated_yields.get(r_name, 0.0) + amount
                            break

                # 2. Legacy resources format (<resources><resource ware="ore" yield="1.5" />)
                resources_elem = dataset_elem.find("resources")
                if resources_elem is not None:
                    for res_elem in resources_elem.findall("resource"):
                        res_id = res_elem.attrib.get("ware")
                        if not res_id:
                            continue
                        try:
                            yield_val = float(res_elem.attrib.get("yield", "0.0"))
                        except ValueError:
                            yield_val = 0.0
                        accumulated_yields[res_id] = accumulated_yields.get(res_id, 0.0) + yield_val

                for res_id, total_yield in accumulated_yields.items():
                    sector_resources.append(
                        SectorResourceRecord(
                            sector_id=sector_id,
                            resource_id=res_id,
                            resource_yield=total_yield,
                        )
                    )

        except Exception as e:
            logger.error(f"Failed parsing sector mapdefaults XML '{mapdefaults_xml_path}': {e}")

        return sectors, sector_resources

    def parse_ship_macro(
        self,
        macro_path: Path,
        storage_macros: Dict[str, float],
        macro_to_ware_id: Dict[str, str],
    ) -> Optional[ShipRecord]:
        """Parses a ship macro XML file into a ShipRecord."""
        if not macro_path.exists():
            return None

        try:
            tree = ET.parse(str(macro_path))
            root = tree.getroot()
            macro_elem = root.find("macro")
            if macro_elem is None:
                return None

            macro_id = macro_elem.attrib.get("name")
            if not macro_id:
                return None

            ship_class = macro_elem.attrib.get("class", "ship_m")

            props_elem = macro_elem.find("properties")
            if props_elem is None:
                return None

            id_elem = props_elem.find("identification")
            if id_elem is not None:
                raw_name = id_elem.attrib.get("name", macro_id)
                name = self.text_resolver.resolve(raw_name, default=macro_id)
                faction_id = id_elem.attrib.get("makerrace")
            else:
                name = macro_id
                faction_id = None

            # Hull
            hull_elem = props_elem.find("hull")
            try:
                hull = float(hull_elem.attrib.get("max", "0.0")) if hull_elem is not None else 0.0
            except ValueError:
                hull = 0.0

            # Speed (estimate from jerk/physics if present)
            speed = 0.0
            jerk_elem = props_elem.find("jerk/forward")
            if jerk_elem is not None:
                try:
                    speed = float(jerk_elem.attrib.get("ratio", "1.0")) * 100.0
                except ValueError:
                    speed = 100.0

            # Slot counts
            all_connections = macro_elem.findall(".//connection")
            weapon_slots = 0
            turret_slots = 0
            shield_slots = 0
            for conn_elem in all_connections:
                ref = conn_elem.attrib.get("ref", "")
                if "con_primary" in ref or "con_weapon" in ref:
                    weapon_slots += 1
                elif "con_turret" in ref:
                    turret_slots += 1
                elif "con_shield" in ref:
                    shield_slots += 1

            shields = shield_slots * 500.0  # Aggregate estimate rating

            # Cargo capacity from storage connection macro
            cargo_capacity = 0.0
            cargo_type = "container"
            for conn_elem in all_connections:
                ref = conn_elem.attrib.get("ref", "")
                if "con_storage" in ref:
                    storage_macro_elem = conn_elem.find("macro")
                    if storage_macro_elem is not None:
                        storage_ref = storage_macro_elem.attrib.get("ref")
                        if storage_ref and storage_ref in storage_macros:
                            cargo_capacity = storage_macros[storage_ref]

            purpose_elem = props_elem.find("purpose")
            purpose = purpose_elem.attrib.get("primary") if purpose_elem is not None else None

            ware_id = macro_to_ware_id.get(macro_id)

            return ShipRecord(
                id=macro_id,
                name=name,
                ship_class=ship_class,
                hull=hull,
                shields=shields,
                cargo_capacity=cargo_capacity,
                cargo_type=cargo_type,
                speed=speed,
                weapon_slots=weapon_slots,
                turret_slots=turret_slots,
                shield_slots=shield_slots,
                purpose=purpose,
                faction_id=faction_id,
                ware_id=ware_id,
                raw_macro=macro_id,
            )

        except Exception as e:
            logger.error(f"Failed parsing ship macro '{macro_path}': {e}")
            return None

    def parse_storage_macro(self, storage_macro_path: Path) -> Tuple[Optional[str], float]:
        """Parses a storage macro XML file to obtain storage cargo capacity."""
        if not storage_macro_path.exists():
            return None, 0.0

        try:
            tree = ET.parse(str(storage_macro_path))
            root = tree.getroot()
            macro_elem = root.find("macro")
            if macro_elem is None:
                return None, 0.0

            macro_id = macro_elem.attrib.get("name")
            cargo_elem = macro_elem.find("properties/cargo")
            if cargo_elem is not None:
                try:
                    cargo_max = float(cargo_elem.attrib.get("max", "0.0"))
                    return macro_id, cargo_max
                except ValueError:
                    pass
            return macro_id, 0.0
        except Exception:
            return None, 0.0
