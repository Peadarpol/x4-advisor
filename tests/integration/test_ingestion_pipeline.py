"""End-to-end integration tests for extraction and ingestion pipeline."""

from pathlib import Path

from x4_advisor.ingestion.extractor import process_extracted_directory
from x4_advisor.storage.db import atomic_ingest_to_db, get_connection, insert_domain_data


def test_end_to_end_ingestion_pipeline(tmp_path: Path):
    """Executes full ingestion pipeline from golden extracted fixtures to SQLite database."""
    fixture_dir = Path("tests/fixtures/golden_extracted")
    target_db = tmp_path / "x4_advisor.db"

    (
        metadata,
        factions,
        wares,
        sectors,
        sector_resources,
        ships,
        recipes,
        report,
    ) = process_extracted_directory(fixture_dir)

    def populate(conn):
        return insert_domain_data(
            conn, metadata, factions, wares, sectors, sector_resources, ships, recipes
        )

    inserted, skipped = atomic_ingest_to_db(target_db, populate)
    assert inserted > 0
    assert target_db.exists()

    conn = get_connection(target_db)

    # Verify tables populated
    f_count = conn.execute("SELECT COUNT(*) FROM factions").fetchone()[0]
    w_count = conn.execute("SELECT COUNT(*) FROM wares").fetchone()[0]
    s_count = conn.execute("SELECT COUNT(*) FROM ships").fetchone()[0]
    m_row = conn.execute("SELECT is_base_game_only FROM dataset_metadata WHERE id = 1").fetchone()

    assert f_count == 2
    assert w_count == 4
    assert s_count == 1
    assert m_row[0] == 1

    conn.close()

    # Re-run for idempotency test
    inserted2, skipped2 = atomic_ingest_to_db(target_db, populate)
    assert inserted2 > 0
