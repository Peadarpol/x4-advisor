"""Single source of truth for router query classification vocabularies and allowlists."""

import sqlite3
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Code-derived constants (contracts with structured_query.py and SPEC-001)
# ---------------------------------------------------------------------------

VALID_OPERATIONS: List[str] = [
    "lookup_entity",
    "compare_entities",
    "production_chain",
    "list_category",
    "sector_yield",
    "none",
]

ALLOWED_METRICS: List[str] = [
    "speed",
    "cargo_capacity",
    "hull",
    "shields",
    "weapon_slots",
    "turret_slots",
    "shield_slots",
    "min_price",
    "avg_price",
    "max_price",
    "volume",
    "none",
]

VALID_PURPOSES: List[str] = [
    "fight",
    "trade",
    "mine",
    "build",
    "auxiliary",
    "salvage",
    "none",
]

VALID_SHIP_CLASSES: List[str] = [
    "s",
    "m",
    "l",
    "xl",
    "ship_s",
    "ship_m",
    "ship_l",
    "ship_xl",
    "none",
]

VALID_ENTITY_TYPES: List[str] = [
    "ship",
    "ware",
    "sector",
    "faction",
    "none",
]

VALID_ABSTAIN_REASONS: List[str] = [
    "NO_EVIDENCE",
    "OUT_OF_SCOPE_DLC",
    "OUT_OF_SCOPE_OTHER",
    "MALFORMED_TOOL_CALL",
    "CONFLICTING_EVIDENCE",
    "NONE",
]

# ---------------------------------------------------------------------------
# Dynamic Database-derived Vocabularies
# ---------------------------------------------------------------------------


class DynamicVocabularies:
    """Manages database-derived distinct allowlists loaded at startup from SQLite."""

    def __init__(self, conn: Optional[sqlite3.Connection] = None) -> None:
        self.categories: List[str] = []
        self.resources: List[str] = []
        self.production_methods: List[str] = []

        if conn is not None:
            self.load_from_db(conn)
        else:
            self._load_fallback_defaults()

    def load_from_db(self, conn: sqlite3.Connection) -> None:
        """Loads distinct allowlists directly from core database tables."""
        # 1. Ware categories
        cat_rows = conn.execute(
            "SELECT DISTINCT category FROM wares WHERE category IS NOT NULL AND category != '' ORDER BY category"
        ).fetchall()
        self.categories = [r[0].lower() for r in cat_rows]
        if "none" not in self.categories:
            self.categories.append("none")

        # 2. Sector resources
        res_rows = conn.execute(
            "SELECT DISTINCT resource_id FROM sector_resources WHERE resource_id IS NOT NULL AND resource_id != '' ORDER BY resource_id"
        ).fetchall()
        self.resources = [r[0].lower() for r in res_rows]
        if "none" not in self.resources:
            self.resources.append("none")

        # 3. Production recipe methods (base game only)
        meth_rows = conn.execute(
            "SELECT DISTINCT method FROM production_recipes WHERE method IS NOT NULL AND method != '' ORDER BY method"
        ).fetchall()
        self.production_methods = [r[0].lower() for r in meth_rows]
        if "none" not in self.production_methods:
            self.production_methods.append("none")

    def _load_fallback_defaults(self) -> None:
        """Fallback defaults if database connection is not provided at construction."""
        self.categories = [
            "agricultural", "contraband", "countermeasures", "curiosity", "drones",
            "energy", "engines", "food", "gases", "general", "generalitem", "hardware",
            "hightech", "ice", "luxuryitem", "minerals", "missiles", "pharmaceutical",
            "refined", "shields", "shiptech", "software", "thrusters", "turrets",
            "water", "weapons", "none"
        ]
        self.resources = ["ore", "silicon", "ice", "hydrogen", "helium", "methane", "nividium", "none"]
        self.production_methods = ["default", "xenon", "teladi", "paranid", "recycling", "processing", "none"]


def build_router_json_schema(vocab: Optional[DynamicVocabularies] = None) -> Dict[str, Any]:
    """Constructs the strict grammar-constrained JSON Schema for Ollama format decoding."""
    v = vocab or DynamicVocabularies()

    return {
        "type": "object",
        "properties": {
            "route_type": {
                "type": "string",
                "enum": ["STRUCTURED", "VECTOR", "BOTH", "ABSTAIN"],
            },
            "structured": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": VALID_OPERATIONS,
                    },
                    "query_name": {"type": "string"},
                    "entity_type": {
                        "type": "string",
                        "enum": VALID_ENTITY_TYPES,
                    },
                    "category": {
                        "type": "string",
                        "enum": v.categories,
                    },
                    "metric": {
                        "type": "string",
                        "enum": ALLOWED_METRICS,
                    },
                    "sort_desc": {"type": "boolean"},
                    "ship_class": {
                        "type": "string",
                        "enum": VALID_SHIP_CLASSES,
                    },
                    "purpose": {
                        "type": "string",
                        "enum": VALID_PURPOSES,
                    },
                    "faction": {"type": "string"},
                    "production_method": {
                        "type": "string",
                        "enum": v.production_methods,
                    },
                    "resource_id": {
                        "type": "string",
                        "enum": v.resources,
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["operation"],
                "additionalProperties": False,
            },
            "vector": {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string"},
                },
                "required": ["query_text"],
                "additionalProperties": False,
            },
            "abstain_reason": {
                "type": "string",
                "enum": VALID_ABSTAIN_REASONS,
            },
        },
        "required": ["route_type", "structured", "vector", "abstain_reason"],
        "additionalProperties": False,
    }
