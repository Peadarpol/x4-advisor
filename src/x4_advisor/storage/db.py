"""Database connection, batch insertion, and atomic swap management."""

import logging
import os
from pathlib import Path
import sqlite3
from typing import Callable, List, Optional, Tuple

from x4_advisor.storage.models import (
    DatasetMetadata,
    FactionRecord,
    ProductionRecipeRecord,
    SectorRecord,
    SectorResourceRecord,
    ShipRecord,
    WareRecord,
)
from x4_advisor.storage.schema import init_db_schema

logger = logging.getLogger(__name__)


def get_connection(db_path: Path, load_vec: bool = True) -> sqlite3.Connection:
    """Connects to SQLite database, ensuring parent directories exist and foreign keys are enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    if load_vec:
        try:
            conn.enable_load_extension(True)
            import sqlite_vec

            sqlite_vec.load(conn)
        except Exception as e:
            logger.debug(f"sqlite-vec extension not loaded for '{db_path}': {e}")
    return conn


def insert_domain_data(
    conn: sqlite3.Connection,
    metadata: DatasetMetadata,
    factions: List[FactionRecord],
    wares: List[WareRecord],
    sectors: List[SectorRecord],
    sector_resources: List[SectorResourceRecord],
    ships: List[ShipRecord],
    recipes: List[ProductionRecipeRecord],
) -> Tuple[int, int]:
    """Inserts domain records in strict foreign-key dependency order.

    Returns (inserted_count, skipped_count).
    """
    inserted = 0
    skipped = 0

    cursor = conn.cursor()

    # 1. Dataset Metadata
    cursor.execute(
        """
        INSERT OR REPLACE INTO dataset_metadata (id, game_version, build, extraction_timestamp, is_base_game_only, schema_version)
        VALUES (1, ?, ?, ?, ?, ?)
        """,
        (
            metadata.game_version,
            metadata.build,
            metadata.extraction_timestamp,
            1 if metadata.is_base_game_only else 0,
            metadata.schema_version,
        ),
    )

    # 2. Factions (no FK)
    for f in factions:
        try:
            cursor.execute(
                """
                INSERT INTO factions (id, name, short_name, relations_summary)
                VALUES (?, ?, ?, ?)
                """,
                (f.id, f.name, f.short_name, f.relations_summary),
            )
            inserted += 1
        except sqlite3.IntegrityError as e:
            logger.warning(f"Skipping faction {f.id} due to integrity error: {e}")
            skipped += 1

    # 3. Wares (no FK)
    for w in wares:
        try:
            cursor.execute(
                """
                INSERT INTO wares (id, name, category, min_price, avg_price, max_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (w.id, w.name, w.category, w.min_price, w.avg_price, w.max_price, w.volume),
            )
            inserted += 1
        except sqlite3.IntegrityError as e:
            logger.warning(f"Skipping ware {w.id} due to integrity error: {e}")
            skipped += 1

    # 4. Sectors (FK -> factions)
    for s in sectors:
        try:
            cursor.execute(
                """
                INSERT INTO sectors (id, name, faction_id, sunlight)
                VALUES (?, ?, ?, ?)
                """,
                (s.id, s.name, s.faction_id, s.sunlight),
            )
            inserted += 1
        except sqlite3.IntegrityError as e:
            logger.warning(f"Skipping sector {s.id} due to integrity error: {e}")
            skipped += 1

    # 5. Sector Resources (FK -> sectors)
    for sr in sector_resources:
        try:
            cursor.execute(
                """
                INSERT INTO sector_resources (sector_id, resource_id, yield)
                VALUES (?, ?, ?)
                """,
                (sr.sector_id, sr.resource_id, sr.resource_yield),
            )
            inserted += 1
        except sqlite3.IntegrityError as e:
            logger.warning(
                f"Skipping sector resource ({sr.sector_id}, {sr.resource_id}) due to integrity error: {e}"
            )
            skipped += 1

    # 6. Ships (FK -> factions, wares)
    for sh in ships:
        try:
            cursor.execute(
                """
                INSERT INTO ships (
                    id, name, class, hull, shields, cargo_capacity, cargo_type,
                    speed, weapon_slots, turret_slots, shield_slots, purpose, faction_id, ware_id, raw_macro
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sh.id,
                    sh.name,
                    sh.ship_class,
                    sh.hull,
                    sh.shields,
                    sh.cargo_capacity,
                    sh.cargo_type,
                    sh.speed,
                    sh.weapon_slots,
                    sh.turret_slots,
                    sh.shield_slots,
                    sh.purpose,
                    sh.faction_id,
                    sh.ware_id,
                    sh.raw_macro,
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError as e:
            logger.warning(f"Skipping ship {sh.id} due to integrity error: {e}")
            skipped += 1

    # 7. Production Recipes (FK -> wares)
    for r in recipes:
        try:
            cursor.execute(
                """
                INSERT INTO production_recipes (
                    ware_id, method, input_ware_id, input_amount, output_amount, production_time
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    r.ware_id,
                    r.method,
                    r.input_ware_id,
                    r.input_amount,
                    r.output_amount,
                    r.production_time,
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError as e:
            logger.warning(
                f"Skipping recipe ({r.ware_id}, {r.method}, {r.input_ware_id}) due to integrity error: {e}"
            )
            skipped += 1

    conn.commit()
    return inserted, skipped


def atomic_ingest_to_db(
    target_db_path: Path,
    populate_fn: Callable[[sqlite3.Connection], Tuple[int, int]],
) -> Tuple[int, int]:
    """Executes database population in a process-isolated temporary file and atomically swaps it over target_db_path on success using os.replace.

    Preserves existing curation tables (source_registry, source_manifest, knowledge_chunks, knowledge_chunks_vec)
    across the swap if target_db_path already contains them.
    """
    target_db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_db_path = target_db_path.parent / f"{target_db_path.name}.tmp.{os.getpid()}"

    if temp_db_path.exists():
        try:
            temp_db_path.unlink()
        except OSError:
            pass

    # Snapshot existing curation counts if target DB exists
    curation_tables = ["source_registry", "source_manifest", "knowledge_chunks", "knowledge_chunks_vec"]
    snapshot_counts = {}
    if target_db_path.exists():
        src_conn = get_connection(target_db_path)
        try:
            for t in curation_tables:
                try:
                    count = src_conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                    snapshot_counts[t] = count
                except sqlite3.OperationalError:
                    pass
        finally:
            src_conn.close()

    conn = get_connection(temp_db_path)
    try:
        init_db_schema(conn)
        inserted, skipped = populate_fn(conn)
        conn.commit()

        # Preserve curation tables from target database in strict FK order
        if target_db_path.exists() and snapshot_counts:
            src_conn = get_connection(target_db_path)
            try:
                for t in ["source_registry", "source_manifest", "knowledge_chunks"]:
                    if snapshot_counts.get(t, 0) > 0:
                        columns = [col[1] for col in src_conn.execute(f"PRAGMA table_info({t})").fetchall()]
                        col_names = ", ".join(columns)
                        placeholders = ", ".join(["?"] * len(columns))
                        rows = src_conn.execute(f"SELECT {col_names} FROM {t}").fetchall()
                        conn.executemany(f"INSERT OR REPLACE INTO {t} ({col_names}) VALUES ({placeholders})", rows)

                # 4. knowledge_chunks_vec (Virtual table)
                if snapshot_counts.get("knowledge_chunks_vec", 0) > 0:
                    rows = src_conn.execute("SELECT chunk_id, embedding FROM knowledge_chunks_vec").fetchall()
                    conn.executemany("INSERT OR REPLACE INTO knowledge_chunks_vec (chunk_id, embedding) VALUES (?, ?)", rows)

                conn.commit()
            finally:
                src_conn.close()

        # Post-commit verification
        fk_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
        if fk_violations:
            raise RuntimeError(f"Foreign key violations detected in temp DB: {fk_violations}")

        conn.close()
        del conn

        # Ensure all connection handles are fully garbage collected before os.replace on Windows
        import gc
        gc.collect()

        # Cross-platform atomic overwrite
        os.replace(temp_db_path, target_db_path)
        logger.info(f"Atomically updated database at '{target_db_path}'")

        # Post-swap verification against pre-swap snapshot counts
        if snapshot_counts:
            verify_conn = get_connection(target_db_path)
            try:
                for t, expected_count in snapshot_counts.items():
                    actual_count = verify_conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                    if actual_count != expected_count:
                        raise RuntimeError(
                            f"Curation table '{t}' count mismatch after atomic swap: expected {expected_count}, got {actual_count}"
                        )
            finally:
                verify_conn.close()

        return inserted, skipped
    except Exception as e:
        try:
            if "conn" in locals() and conn:
                conn.close()
        except Exception:
            pass
        if temp_db_path.exists():
            try:
                temp_db_path.unlink()
            except OSError:
                pass
        raise e
