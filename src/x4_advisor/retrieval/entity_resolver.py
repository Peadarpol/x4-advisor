"""Entity name resolution engine resolving natural language queries to internal primary key IDs."""

import logging
from pathlib import Path
import sqlite3
from typing import List, Optional, Union

from x4_advisor.retrieval.models import (
    AmbiguousEntityResult,
    EntityNotFoundResult,
    EntityResolutionOutcome,
    ResolvedEntity,
)

logger = logging.getLogger(__name__)


def escape_like_pattern(term: str) -> str:
    """Escapes SQLite LIKE wildcard special characters (%, _, \\) in user input."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class EntityResolver:
    """2-stage entity name resolver operating against X4 Advisor SQLite database."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        if conn is not None:
            self.conn = conn
            self._close_conn_on_exit = False
        elif db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(db_path))
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self._close_conn_on_exit = True
        else:
            raise ValueError("Either db_path or conn must be provided to EntityResolver.")

        self.conn.execute("PRAGMA query_only = ON;")

    def close(self) -> None:
        """Closes connection if owned by this instance."""
        if self._close_conn_on_exit and self.conn:
            self.conn.close()

    def __enter__(self) -> "EntityResolver":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def resolve_entity(
        self,
        name_query: str,
        entity_types: Optional[List[str]] = None,
    ) -> EntityResolutionOutcome:
        """Resolves natural language entity name to a ResolvedEntity, AmbiguousEntityResult, or EntityNotFoundResult."""
        clean_query = name_query.strip()
        if not clean_query:
            return EntityNotFoundResult(query_name=name_query, message="Empty query string.")

        valid_types = {"ship", "ware", "sector", "faction"}
        if entity_types is not None:
            invalid_types = [t for t in entity_types if t.lower() not in valid_types]
            if invalid_types:
                raise ValueError(
                    f"Invalid entity_types filter: {invalid_types}. Allowed types: ['ship', 'ware', 'sector', 'faction']."
                )
            target_types = [t.lower() for t in entity_types]
        else:
            target_types = list(valid_types)

        # Stage 1: Exact Match (Case-insensitive)
        exact_matches = self._dedup_ship_wares(self._search_exact(clean_query, target_types))
        if len(exact_matches) == 1:
            return exact_matches[0]
        elif len(exact_matches) > 1:
            return AmbiguousEntityResult(query_name=clean_query, candidates=exact_matches)

        # Stage 2: Substring Match (Case-insensitive with ESCAPE '\')
        partial_matches = self._dedup_ship_wares(self._search_partial(clean_query, target_types))
        if len(partial_matches) == 1:
            return partial_matches[0]
        elif len(partial_matches) > 1:
            return AmbiguousEntityResult(query_name=clean_query, candidates=partial_matches)

        return EntityNotFoundResult(query_name=clean_query)

    def _dedup_ship_wares(self, matches: List[ResolvedEntity]) -> List[ResolvedEntity]:
        """If both a ship and its corresponding ware match the same name, keep the ship entity."""
        ship_names = {m.name.lower() for m in matches if m.entity_type == "ship"}
        if not ship_names:
            return matches
        return [m for m in matches if not (m.entity_type == "ware" and m.name.lower() in ship_names)]

    def _search_exact(self, query: str, target_types: List[str]) -> List[ResolvedEntity]:
        """Performs case-insensitive exact matches against selected domain tables."""
        results: List[ResolvedEntity] = []
        lower_q = query.lower()

        # Ships
        if "ship" in target_types:
            rows = self.conn.execute(
                "SELECT id, name FROM ships WHERE LOWER(name) = ? OR LOWER(id) = ? ORDER BY id ASC",
                (lower_q, lower_q),
            ).fetchall()
            for r_id, r_name in rows:
                results.append(ResolvedEntity(id=r_id, name=r_name, entity_type="ship"))

        # Wares
        if "ware" in target_types:
            rows = self.conn.execute(
                "SELECT id, name FROM wares WHERE LOWER(name) = ? OR LOWER(id) = ? ORDER BY id ASC",
                (lower_q, lower_q),
            ).fetchall()
            for r_id, r_name in rows:
                results.append(ResolvedEntity(id=r_id, name=r_name, entity_type="ware"))

        # Sectors
        if "sector" in target_types:
            rows = self.conn.execute(
                "SELECT id, name FROM sectors WHERE LOWER(name) = ? OR LOWER(id) = ? ORDER BY id ASC",
                (lower_q, lower_q),
            ).fetchall()
            for r_id, r_name in rows:
                results.append(ResolvedEntity(id=r_id, name=r_name, entity_type="sector"))

        # Factions
        if "faction" in target_types:
            rows = self.conn.execute(
                "SELECT id, name FROM factions WHERE LOWER(name) = ? OR LOWER(id) = ? OR LOWER(short_name) = ? ORDER BY id ASC",
                (lower_q, lower_q, lower_q),
            ).fetchall()
            for r_id, r_name in rows:
                results.append(ResolvedEntity(id=r_id, name=r_name, entity_type="faction"))

        return results

    def _search_partial(self, query: str, target_types: List[str]) -> List[ResolvedEntity]:
        """Performs case-insensitive substring matches with wildcard escaping."""
        results: List[ResolvedEntity] = []
        escaped_q = escape_like_pattern(query.lower())
        pattern = f"%{escaped_q}%"

        # Ships
        if "ship" in target_types:
            rows = self.conn.execute(
                "SELECT id, name FROM ships WHERE LOWER(name) LIKE ? ESCAPE '\\' OR LOWER(id) LIKE ? ESCAPE '\\' ORDER BY id ASC",
                (pattern, pattern),
            ).fetchall()
            for r_id, r_name in rows:
                results.append(ResolvedEntity(id=r_id, name=r_name, entity_type="ship"))

        # Wares
        if "ware" in target_types:
            rows = self.conn.execute(
                "SELECT id, name FROM wares WHERE LOWER(name) LIKE ? ESCAPE '\\' OR LOWER(id) LIKE ? ESCAPE '\\' ORDER BY id ASC",
                (pattern, pattern),
            ).fetchall()
            for r_id, r_name in rows:
                results.append(ResolvedEntity(id=r_id, name=r_name, entity_type="ware"))

        # Sectors
        if "sector" in target_types:
            rows = self.conn.execute(
                "SELECT id, name FROM sectors WHERE LOWER(name) LIKE ? ESCAPE '\\' OR LOWER(id) LIKE ? ESCAPE '\\' ORDER BY id ASC",
                (pattern, pattern),
            ).fetchall()
            for r_id, r_name in rows:
                results.append(ResolvedEntity(id=r_id, name=r_name, entity_type="sector"))

        # Factions
        if "faction" in target_types:
            rows = self.conn.execute(
                "SELECT id, name FROM factions WHERE LOWER(name) LIKE ? ESCAPE '\\' OR LOWER(id) LIKE ? ESCAPE '\\' OR LOWER(short_name) LIKE ? ESCAPE '\\' ORDER BY id ASC",
                (pattern, pattern, pattern),
            ).fetchall()
            for r_id, r_name in rows:
                results.append(ResolvedEntity(id=r_id, name=r_name, entity_type="faction"))

        return results
