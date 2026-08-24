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
]


def init_db_schema(conn: sqlite3.Connection) -> None:
    """Executes all table creation DDL statements on the given SQLite connection."""
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()
    for statement in DDL_STATEMENTS:
        cursor.execute(statement)
    conn.commit()
