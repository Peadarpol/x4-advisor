"""Command-line entrypoint for X4 Advisor game data extraction and ingestion."""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Tuple

from x4_advisor.config import ConfigError, get_config
from x4_advisor.ingestion.extractor import run_extraction_pipeline
from x4_advisor.storage.db import atomic_ingest_to_db, insert_domain_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("x4_advisor.ingestion.cli")


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract and ingest structured game data from X4 Foundations base game into SQLite."
    )
    parser.add_argument(
        "--install-path",
        type=Path,
        help="Path to base X4 Foundations installation directory (containing 01.cat).",
    )
    parser.add_argument(
        "--from-extracted",
        type=Path,
        help="Path to directory containing pre-extracted XML files (for testing/offline use).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        help="Target SQLite database path (defaults to DATABASE_PATH env var or data/db/x4_advisor.db).",
    )
    return parser.parse_args()


def main() -> None:
    """Executes ingestion CLI pipeline."""
    args = parse_args()
    start_time = time.time()

    # Determine configuration & paths
    try:
        config = get_config(validate=False)
    except Exception as e:
        logger.error(f"Configuration load error: {e}")
        sys.exit(1)

    db_path = args.db_path or config.database_path

    # Evaluate CLI argument priority
    install_path: Path
    from_extracted_dir: Path = args.from_extracted

    if from_extracted_dir is not None:
        logger.info(f"Ingesting pre-extracted XML data from '{from_extracted_dir}'...")
        install_path = None
    elif args.install_path is not None:
        install_path = args.install_path
        logger.info(f"Extracting from X4 installation path '{install_path}'...")
    else:
        # Fall back to env var
        try:
            config.validate_m1_config()
            install_path = config.x4_install_path
            logger.info(f"Extracting from configured X4 installation path '{install_path}'...")
        except ConfigError as e:
            logger.error(
                f"Missing extraction source: {e}\n"
                "Please provide --install-path <PATH>, --from-extracted <DIR>, or set X4_INSTALL_PATH in .env."
            )
            sys.exit(1)

    try:
        # Run extraction & domain validation
        (
            metadata,
            factions,
            wares,
            sectors,
            sector_resources,
            ships,
            recipes,
            report,
        ) = run_extraction_pipeline(
            install_path=install_path,
            from_extracted_dir=from_extracted_dir,
        )

        def populate_fn(conn) -> Tuple[int, int]:
            return insert_domain_data(
                conn=conn,
                metadata=metadata,
                factions=factions,
                wares=wares,
                sectors=sectors,
                sector_resources=sector_resources,
                ships=ships,
                recipes=recipes,
            )

        # Atomic SQLite write
        inserted_count, skipped_db_count = atomic_ingest_to_db(db_path, populate_fn)
        duration = time.time() - start_time

        print("\n==================================================")
        print("           X4 ADVISOR INGESTION REPORT            ")
        print("==================================================")
        print(f"Target Database      : {db_path.resolve()}")
        print(f"Game Version         : {metadata.game_version} ({metadata.build})")
        print(f"Base Game Only       : {metadata.is_base_game_only}")
        raw = report.raw_counts
        print(f"Factions (Raw/Valid) : {raw.get('factions', len(factions))} / {len(factions)}")
        print(f"Wares (Raw/Valid)    : {raw.get('wares', len(wares))} / {len(wares)}")
        print(f"Sectors (Raw/Valid)  : {raw.get('sectors', len(sectors))} / {len(sectors)}")
        print(f"Sector Yields        : {raw.get('sector_resources', len(sector_resources))} / {len(sector_resources)}")
        print(f"Ships (Raw/Valid)    : {raw.get('ships', len(ships))} / {len(ships)}")
        print(f"Recipes (Raw/Valid)  : {raw.get('recipes', len(recipes))} / {len(recipes)}")
        print("--------------------------------------------------")
        print(f"Validation Processed : {report.total_processed}")
        print(f"Validation Valid     : {report.total_valid}")
        print(f"Validation Skipped   : {report.total_skipped}")
        print(f"Database Inserted    : {inserted_count}")
        print(f"Execution Duration   : {duration:.2f} seconds")
        print("==================================================\n")

    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
