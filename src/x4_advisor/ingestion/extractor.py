"""Subprocess extractor wrapper for x4cat and extraction pipeline orchestrator."""

from datetime import datetime, timezone
import logging
from pathlib import Path
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from x4_advisor.ingestion.adapter import NormalizationAdapter
from x4_advisor.ingestion.text_resolver import TextResolver
from x4_advisor.ingestion.validator import DomainValidator, ValidationReport
from x4_advisor.storage.models import (
    DatasetMetadata,
    FactionRecord,
    ProductionRecipeRecord,
    SectorRecord,
    SectorResourceRecord,
    ShipRecord,
    WareRecord,
)

logger = logging.getLogger(__name__)


class DLCBoundaryError(ValueError):
    """Raised when an extraction path violates the base-game DLC boundary rule."""

    pass


class ExtractorError(RuntimeError):
    """Raised when x4cat extraction fails or times out."""

    pass


def validate_install_path(install_path: Path) -> None:
    """Enforces DLC boundary rules by validating the install path."""
    if not install_path.exists() or not install_path.is_dir():
        raise DLCBoundaryError(f"Install path '{install_path}' does not exist or is not a directory.")

    path_parts = [part.lower() for part in install_path.parts]
    if "extensions" in path_parts or any("ego_dlc" in part for part in path_parts):
        raise DLCBoundaryError(
            f"Install path '{install_path}' points inside extensions or DLC directories, violating base-game boundary."
        )

    root_cat = install_path / "01.cat"
    if not root_cat.exists():
        raise DLCBoundaryError(
            f"Install path '{install_path}' does not contain root catalog '01.cat'."
        )


def extract_catalog_files(install_path: Path, output_dir: Path, timeout: int = 300) -> None:
    """Executes x4cat extract to pull base-game XML archives to disk."""
    validate_install_path(install_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "x4cat",
        "extract",
        "-o",
        str(output_dir),
        "-p",
        "",
        "-g",
        "*.xml",
        str(install_path.resolve()),
    ]

    logger.info(f"Executing extraction command: {' '.join(cmd)}")

    try:
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        logger.info("x4cat extract completed successfully.")
        logger.debug(res.stdout)
    except FileNotFoundError:
        raise ExtractorError(
            "x4cat executable was not found on PATH. Please ensure x4cat is installed via 'uv tool install git+https://github.com/meethune/x4cat.git'."
        )
    except subprocess.CalledProcessError as e:
        raise ExtractorError(
            f"x4cat extract failed with exit code {e.returncode}.\nStderr: {e.stderr}"
        )
    except subprocess.TimeoutExpired as e:
        raise ExtractorError(f"x4cat extract timed out after {e.timeout} seconds.")


def detect_game_version(
    install_path: Optional[Path] = None,
    extracted_dir: Optional[Path] = None,
) -> Tuple[str, str]:
    """Detects X4 game version and build string from version.dat in install path or extracted dir."""
    candidate_paths = []
    if install_path:
        candidate_paths.append(install_path / "version.dat")
    if extracted_dir:
        candidate_paths.append(extracted_dir / "version.dat")

    for v_path in candidate_paths:
        if v_path.exists():
            try:
                content = v_path.read_text(encoding="utf-8").strip()
                if content.isdigit():
                    ver_int = int(content)
                    major = ver_int // 100
                    minor = ver_int % 100
                    return f"{major}.{minor:02d}", f"build-{ver_int}"
                elif content:
                    return content, "build-custom"
            except Exception as e:
                logger.warning(f"Could not read version file '{v_path}': {e}")

    return "7.x-base", "extracted-fixture"


def process_extracted_directory(
    extracted_dir: Path,
    install_path: Optional[Path] = None,
) -> Tuple[
    DatasetMetadata,
    List[FactionRecord],
    List[WareRecord],
    List[SectorRecord],
    List[SectorResourceRecord],
    List[ShipRecord],
    List[ProductionRecipeRecord],
    ValidationReport,
]:
    """Parses extracted XML directory, runs validation, and returns valid domain records."""
    # 1. Text localization
    text_resolver = TextResolver()
    lang_file = extracted_dir / "t" / "0001-l044.xml"
    if not lang_file.exists():
        lang_file = extracted_dir / "t" / "0001.xml"
    if lang_file.exists():
        text_resolver.load_from_file(lang_file)

    adapter = NormalizationAdapter(text_resolver)

    # 2. Parse Wares & Recipes
    wares_path = extracted_dir / "libraries" / "wares.xml"
    wares, recipes, macro_to_ware_id, macro_to_ware_name = adapter.parse_wares_and_recipes(wares_path)

    # 3. Parse Factions
    factions_path = extracted_dir / "libraries" / "factions.xml"
    factions = adapter.parse_factions(factions_path)

    # 4. Parse Sectors & Resources
    mapdefaults_path = extracted_dir / "libraries" / "mapdefaults.xml"
    sectors, sector_resources = adapter.parse_sectors(mapdefaults_path)

    # 5. Parse Storage Macros & Ship Macros
    storage_macros: Dict[str, float] = {}

    # Gather storage macros
    for storage_file in extracted_dir.glob("assets/units/**/macros/storage_*.xml"):
        st_id, cap = adapter.parse_storage_macro(storage_file)
        if st_id:
            storage_macros[st_id] = cap

    # Gather ship macros
    ships: List[ShipRecord] = []
    for ship_file in extracted_dir.glob("assets/units/**/macros/ship_*.xml"):
        ship_rec = adapter.parse_ship_macro(ship_file, storage_macros, macro_to_ware_id, macro_to_ware_name)
        if ship_rec:
            ships.append(ship_rec)

    # 6. Validate
    validator = DomainValidator()
    (
        v_factions,
        v_wares,
        v_sectors,
        v_sector_resources,
        v_ships,
        v_recipes,
        report,
    ) = validator.validate_dataset(
        factions=factions,
        wares=wares,
        sectors=sectors,
        sector_resources=sector_resources,
        ships=ships,
        recipes=recipes,
    )

    game_ver, build_str = detect_game_version(install_path=install_path, extracted_dir=extracted_dir)
    timestamp_str = datetime.now(timezone.utc).isoformat()

    metadata = DatasetMetadata(
        game_version=game_ver,
        build=build_str,
        extraction_timestamp=timestamp_str,
        is_base_game_only=True,
    )

    return (
        metadata,
        v_factions,
        v_wares,
        v_sectors,
        v_sector_resources,
        v_ships,
        v_recipes,
        report,
    )


def run_extraction_pipeline(
    install_path: Optional[Path] = None,
    from_extracted_dir: Optional[Path] = None,
) -> Tuple[
    DatasetMetadata,
    List[FactionRecord],
    List[WareRecord],
    List[SectorRecord],
    List[SectorResourceRecord],
    List[ShipRecord],
    List[ProductionRecipeRecord],
    ValidationReport,
]:
    """Runs end-to-end extraction pipeline from game install or pre-extracted folder."""
    if from_extracted_dir is not None:
        if not from_extracted_dir.exists() or not from_extracted_dir.is_dir():
            raise ValueError(f"Extracted directory '{from_extracted_dir}' does not exist or is not a directory.")
        return process_extracted_directory(from_extracted_dir, install_path=install_path)

    if install_path is None:
        raise ValueError("Either install_path or from_extracted_dir must be provided.")

    validate_install_path(install_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        staging_dir = Path(temp_dir)
        extract_catalog_files(install_path, staging_dir)
        return process_extracted_directory(staging_dir, install_path=install_path)
