"""SQLite DDL schema definitions for X4 Advisor database."""

import sqlite3

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS dataset_metadata (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        game_version TEXT NOT NULL,
        build TEXT NOT NULL,
        extraction_timestamp TEXT NOT NULL,
        is_base_game_only INTEGER NOT NULL CHECK (is_base_game_only IN (0, 1)),
        schema_version TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS factions (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        short_name TEXT,
        relations_summary TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_factions_name ON factions(name COLLATE NOCASE);",
    """
    CREATE TABLE IF NOT EXISTS wares (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        min_price INTEGER NOT NULL,
        avg_price INTEGER NOT NULL,
        max_price INTEGER NOT NULL,
        volume INTEGER NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_wares_name ON wares(name COLLATE NOCASE);",
    "CREATE INDEX IF NOT EXISTS idx_wares_category ON wares(category);",
    """
    CREATE TABLE IF NOT EXISTS sectors (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        faction_id TEXT,
        sunlight REAL NOT NULL DEFAULT 1.0,
        FOREIGN KEY (faction_id) REFERENCES factions(id) ON DELETE SET NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_sectors_name ON sectors(name COLLATE NOCASE);",
    """
    CREATE TABLE IF NOT EXISTS sector_resources (
        sector_id TEXT NOT NULL,
        resource_id TEXT NOT NULL,
        yield REAL NOT NULL CHECK (yield >= 0),
        PRIMARY KEY (sector_id, resource_id),
        FOREIGN KEY (sector_id) REFERENCES sectors(id) ON DELETE CASCADE
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_sector_resources_lookup ON sector_resources(resource_id, yield DESC);",
    """
    CREATE TABLE IF NOT EXISTS ships (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        class TEXT NOT NULL,
        hull REAL NOT NULL,
        shields REAL NOT NULL,
        cargo_capacity REAL NOT NULL,
        cargo_type TEXT,
        speed REAL NOT NULL,
        weapon_slots INTEGER NOT NULL DEFAULT 0,
        turret_slots INTEGER NOT NULL DEFAULT 0,
        shield_slots INTEGER NOT NULL DEFAULT 0,
        purpose TEXT,
        faction_id TEXT,
        ware_id TEXT,
        raw_macro TEXT,
        FOREIGN KEY (faction_id) REFERENCES factions(id) ON DELETE SET NULL,
        FOREIGN KEY (ware_id) REFERENCES wares(id) ON DELETE SET NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_ships_name ON ships(name COLLATE NOCASE);",
    "CREATE INDEX IF NOT EXISTS idx_ships_class ON ships(class);",
    "CREATE INDEX IF NOT EXISTS idx_ships_purpose ON ships(purpose);",
    "CREATE INDEX IF NOT EXISTS idx_ships_class_purpose ON ships(class, purpose);",
    "CREATE INDEX IF NOT EXISTS idx_ships_ware ON ships(ware_id);",
    """
    CREATE TABLE IF NOT EXISTS production_recipes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ware_id TEXT NOT NULL,
        method TEXT NOT NULL DEFAULT 'default',
        input_ware_id TEXT NOT NULL,
        input_amount INTEGER NOT NULL,
        output_amount INTEGER NOT NULL,
        production_time REAL NOT NULL,
        FOREIGN KEY (ware_id) REFERENCES wares(id) ON DELETE CASCADE,
        FOREIGN KEY (input_ware_id) REFERENCES wares(id) ON DELETE RESTRICT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_recipes_ware ON production_recipes(ware_id, method);",
    "CREATE INDEX IF NOT EXISTS idx_recipes_input ON production_recipes(input_ware_id);",
    """
    CREATE TABLE IF NOT EXISTS source_registry (
        source_id TEXT PRIMARY KEY,
        url TEXT NOT NULL,
        title TEXT NOT NULL,
        proposed_by TEXT NOT NULL,
        category TEXT NOT NULL,
        topic_tags TEXT,
        proposed_date TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'proposed',
        trust_rationale TEXT,
        reviewed_by TEXT,
        reviewed_date TEXT,
        notes TEXT,
        content_date TEXT,
        last_checked TEXT,
        superseded_by TEXT REFERENCES source_registry(source_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_source_registry_status ON source_registry(status);",
    """
    CREATE TABLE IF NOT EXISTS source_manifest (
        manifest_id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL REFERENCES source_registry(source_id),
        title TEXT NOT NULL,
        file_path TEXT NOT NULL,
        curation_status TEXT NOT NULL DEFAULT 'draft',
        raw_hash TEXT NOT NULL,
        claims_hash TEXT,
        fidelity_discrepancies INTEGER DEFAULT 0,
        db_discrepancies INTEGER DEFAULT 0,
        approved_at TEXT,
        approved_by TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_source_manifest_status ON source_manifest(curation_status);",
    """
    CREATE TABLE IF NOT EXISTS knowledge_chunks (
        id TEXT PRIMARY KEY,
        manifest_id TEXT NOT NULL REFERENCES source_manifest(manifest_id) ON DELETE CASCADE,
        heading_hierarchy TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        token_count INTEGER NOT NULL,
        source_attribution TEXT NOT NULL,
        topic TEXT,
        related_entity_ids TEXT,
        game_version_scope TEXT NOT NULL DEFAULT 'base_game',
        created_at TEXT NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_manifest ON knowledge_chunks(manifest_id);",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_topic ON knowledge_chunks(topic);",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_vec USING vec0(
        chunk_id TEXT PRIMARY KEY,
        embedding float[1024]
    );
    """,
]


def init_db_schema(conn: sqlite3.Connection) -> None:
    """Executes all table creation DDL statements on the given SQLite connection."""
    import sqlite_vec

    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except Exception:
        pass  # In case extension is already loaded or environment handles it

    cursor = conn.cursor()
    for statement in DDL_STATEMENTS:
        cursor.execute(statement)
    conn.commit()

