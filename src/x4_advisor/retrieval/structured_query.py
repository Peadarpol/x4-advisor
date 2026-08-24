"""Structured Query Engine executing parameterized SQL templates (T1, T2, T3, T4) against SQLite."""

from collections import defaultdict
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

from x4_advisor.retrieval.models import (
    CategoryListResult,
    ProductionChainResult,
    ProductionNode,
    RankingItem,
    RankingResult,
    SingleEntityResult,
)

logger = logging.getLogger(__name__)

# Security Invariant: Dynamic ORDER BY metric whitelists to prevent identifier SQL injection
ALLOWED_SHIP_METRICS: Dict[str, Tuple[str, str]] = {
    "cargo_capacity": ("cargo_capacity", "m³"),
    "cargo": ("cargo_capacity", "m³"),
    "hull": ("hull", "points"),
    "shields": ("shields", "rating"),
    "speed": ("speed", "m/s"),
    "weapon_slots": ("weapon_slots", "slots"),
    "turret_slots": ("turret_slots", "slots"),
    "shield_slots": ("shield_slots", "slots"),
}

ALLOWED_WARE_METRICS: Dict[str, Tuple[str, str]] = {
    "min_price": ("min_price", "credits"),
    "avg_price": ("avg_price", "credits"),
    "max_price": ("max_price", "credits"),
    "volume": ("volume", "m³"),
}


class StructuredQueryEngine:
    """Deterministic, parameterized query engine executing against the X4 domain database."""

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
            raise ValueError("Either db_path or conn must be provided to StructuredQueryEngine.")

        self.conn.execute("PRAGMA query_only = ON;")

    def close(self) -> None:
        """Closes database connection if owned by this engine instance."""
        if self._close_conn_on_exit and self.conn:
            self.conn.close()

    def __enter__(self) -> "StructuredQueryEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Template T1: Single-Entity Fact Lookup
    # -------------------------------------------------------------------------
    def query_t1_fact_lookup(self, entity_id: str) -> Optional[SingleEntityResult]:
        """Fetches full attribute record for a resolved entity ID across ships, wares, sectors, or factions."""
        clean_id = entity_id.strip()

        # 1. Search Ships
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT s.id, s.name, s.class, s.hull, s.shields, s.cargo_capacity, s.cargo_type,
                   s.speed, s.weapon_slots, s.turret_slots, s.shield_slots, s.purpose,
                   s.faction_id, f.name AS faction_name, s.ware_id
            FROM ships s
            LEFT JOIN factions f ON s.faction_id = f.id
            WHERE s.id = ?
            """,
            (clean_id,),
        )
        row = cursor.fetchone()
        if row:
            data = {
                "id": row[0],
                "name": row[1],
                "ship_class": row[2],
                "hull": row[3],
                "shields": row[4],
                "cargo_capacity": row[5],
                "cargo_type": row[6],
                "speed": row[7],
                "weapon_slots": row[8],
                "turret_slots": row[9],
                "shield_slots": row[10],
                "purpose": row[11],
                "faction_id": row[12],
                "faction_name": row[13],
                "ware_id": row[14],
            }
            return SingleEntityResult(
                entity_id=clean_id,
                entity_name=row[1],
                entity_type="ship",
                data=data,
            )

        # 2. Search Wares
        cursor.execute(
            "SELECT id, name, category, min_price, avg_price, max_price, volume FROM wares WHERE id = ?",
            (clean_id,),
        )
        row = cursor.fetchone()
        if row:
            data = {
                "id": row[0],
                "name": row[1],
                "category": row[2],
                "min_price": row[3],
                "avg_price": row[4],
                "max_price": row[5],
                "volume": row[6],
            }
            return SingleEntityResult(
                entity_id=clean_id,
                entity_name=row[1],
                entity_type="ware",
                data=data,
            )

        # 3. Search Sectors
        cursor.execute(
            """
            SELECT s.id, s.name, s.faction_id, f.name AS faction_name, s.sunlight
            FROM sectors s
            LEFT JOIN factions f ON s.faction_id = f.id
            WHERE s.id = ?
            """,
            (clean_id,),
        )
        row = cursor.fetchone()
        if row:
            res_rows = cursor.execute(
                "SELECT resource_id, yield FROM sector_resources WHERE sector_id = ?",
                (clean_id,),
            ).fetchall()
            resources = {r[0]: r[1] for r in res_rows}
            data = {
                "id": row[0],
                "name": row[1],
                "faction_id": row[2],
                "faction_name": row[3],
                "sunlight": row[4],
                "resource_yields": resources,
            }
            return SingleEntityResult(
                entity_id=clean_id,
                entity_name=row[1],
                entity_type="sector",
                data=data,
            )

        # 4. Search Factions
        cursor.execute(
            "SELECT id, name, short_name, relations_summary FROM factions WHERE id = ?",
            (clean_id,),
        )
        row = cursor.fetchone()
        if row:
            data = {
                "id": row[0],
                "name": row[1],
                "short_name": row[2],
                "relations_summary": row[3],
            }
            return SingleEntityResult(
                entity_id=clean_id,
                entity_name=row[1],
                entity_type="faction",
                data=data,
            )

        return None

    # -------------------------------------------------------------------------
    # Template T2: Comparison / Ranking Queries
    # -------------------------------------------------------------------------
    def query_t2_ranking(
        self,
        category_or_class: Optional[str] = None,
        metric: str = "cargo_capacity",
        purpose: Optional[str] = None,
        sort_desc: bool = True,
        limit: int = 5,
    ) -> RankingResult:
        """Ranks ships or wares by metric, filterable by class and/or purpose."""
        lower_metric = metric.strip().lower()

        if lower_metric in ALLOWED_SHIP_METRICS:
            sql_col, unit = ALLOWED_SHIP_METRICS[lower_metric]
            return self._query_t2_ships_ranking(
                col_name=sql_col,
                unit=unit,
                metric_name=lower_metric,
                ship_class=category_or_class,
                purpose=purpose,
                sort_desc=sort_desc,
                limit=limit,
            )
        elif lower_metric in ALLOWED_WARE_METRICS:
            sql_col, unit = ALLOWED_WARE_METRICS[lower_metric]
            return self._query_t2_wares_ranking(
                col_name=sql_col,
                unit=unit,
                metric_name=lower_metric,
                category=category_or_class,
                sort_desc=sort_desc,
                limit=limit,
            )
        else:
            raise ValueError(
                f"Invalid or unwhitelisted metric '{metric}'. Allowed ship metrics: {list(ALLOWED_SHIP_METRICS.keys())}, allowed ware metrics: {list(ALLOWED_WARE_METRICS.keys())}"
            )

    def _query_t2_ships_ranking(
        self,
        col_name: str,
        unit: str,
        metric_name: str,
        ship_class: Optional[str],
        purpose: Optional[str],
        sort_desc: bool,
        limit: int,
    ) -> RankingResult:
        order_dir = "DESC" if sort_desc else "ASC"
        conditions = []
        params = []

        if ship_class:
            conditions.append("class = ?")
            params.append(ship_class.strip().lower())
        if purpose:
            conditions.append("purpose = ?")
            params.append(purpose.strip().lower())

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT id, name, {col_name}, purpose, class
            FROM ships
            {where_clause}
            ORDER BY {col_name} {order_dir}
            LIMIT ?
        """
        params.append(limit)

        cursor = self.conn.cursor()
        rows = cursor.execute(sql, params).fetchall()

        items = [
            RankingItem(
                id=r[0],
                name=r[1],
                value=float(r[2]),
                unit=unit,
                metric_name=metric_name,
                purpose=r[3],
                ship_class=r[4],
            )
            for r in rows
        ]

        cat_desc = f"ships (class={ship_class or 'any'}, purpose={purpose or 'any'})"
        return RankingResult(
            category=cat_desc,
            metric=metric_name,
            sort_order=order_dir,
            items=items,
        )

    def _query_t2_wares_ranking(
        self,
        col_name: str,
        unit: str,
        metric_name: str,
        category: Optional[str],
        sort_desc: bool,
        limit: int,
    ) -> RankingResult:
        order_dir = "DESC" if sort_desc else "ASC"
        conditions = []
        params = []

        if category:
            conditions.append("category = ?")
            params.append(category.strip().lower())

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT id, name, {col_name}
            FROM wares
            {where_clause}
            ORDER BY {col_name} {order_dir}
            LIMIT ?
        """
        params.append(limit)

        cursor = self.conn.cursor()
        rows = cursor.execute(sql, params).fetchall()

        items = [
            RankingItem(
                id=r[0],
                name=r[1],
                value=float(r[2]),
                unit=unit,
                metric_name=metric_name,
            )
            for r in rows
        ]

        cat_desc = f"wares (category={category or 'any'})"
        return RankingResult(
            category=cat_desc,
            metric=metric_name,
            sort_order=order_dir,
            items=items,
        )

    def query_t2_sector_yield_ranking(
        self,
        resource_id: str,
        limit: int = 5,
    ) -> RankingResult:
        """Ranks sectors by yield for a specific resource (e.g. 'ore', 'silicon', 'hydrogen')."""
        clean_res = resource_id.strip().lower()
        sql = """
            SELECT s.id, s.name, sr.yield
            FROM sector_resources sr
            JOIN sectors s ON sr.sector_id = s.id
            WHERE LOWER(sr.resource_id) = ?
            ORDER BY sr.yield DESC
            LIMIT ?
        """
        cursor = self.conn.cursor()
        rows = cursor.execute(sql, (clean_res, limit)).fetchall()

        items = [
            RankingItem(
                id=r[0],
                name=r[1],
                value=float(r[2]),
                unit="yield",
                metric_name=f"{clean_res}_yield",
            )
            for r in rows
        ]

        return RankingResult(
            category=f"sectors (resource={clean_res})",
            metric="yield",
            sort_order="DESC",
            items=items,
        )

    # -------------------------------------------------------------------------
    # Template T3: Production Chain Traversal
    # -------------------------------------------------------------------------
    def list_production_methods(self, ware_id: str) -> List[str]:
        """Returns list of available production method names for a target ware."""
        clean_id = ware_id.strip().lower()
        cursor = self.conn.cursor()
        rows = cursor.execute(
            "SELECT DISTINCT method FROM production_recipes WHERE ware_id = ?",
            (clean_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def query_t3_production_chain(
        self,
        ware_id: str,
        method: str = "default",
        max_depth: int = 10,
    ) -> Optional[ProductionChainResult]:
        """Computes multi-tier production recipe tree and total raw materials for a ware."""
        clean_ware_id = ware_id.strip().lower()
        cursor = self.conn.cursor()

        # Check ware exists
        ware_row = cursor.execute("SELECT id, name FROM wares WHERE id = ?", (clean_ware_id,)).fetchone()
        if not ware_row:
            return None

        ware_name = ware_row[1]

        # Multi-method fallback check
        available_methods = self.list_production_methods(clean_ware_id)
        if not available_methods:
            # Raw material (no recipe)
            node = ProductionNode(
                ware_id=clean_ware_id,
                ware_name=ware_name,
                method="none",
                amount_needed=1,
                direct_inputs=[],
            )
            return ProductionChainResult(
                target_ware_id=clean_ware_id,
                target_ware_name=ware_name,
                method="none",
                output_amount=1,
                production_time=0.0,
                tree=node,
                total_raw_materials={clean_ware_id: 1},
            )

        requested_method = method.strip().lower()
        was_fallback = False
        if requested_method not in available_methods:
            active_method = available_methods[0]
            was_fallback = True
            logger.info(
                f"Production method '{requested_method}' not available for ware '{clean_ware_id}'. "
                f"Falling back to available method '{active_method}' (available: {available_methods})."
            )
        else:
            active_method = requested_method

        # Get root recipe metadata
        recipe_meta = cursor.execute(
            "SELECT output_amount, production_time FROM production_recipes WHERE ware_id = ? AND method = ? LIMIT 1",
            (clean_ware_id, active_method),
        ).fetchone()

        output_amount = recipe_meta[0] if recipe_meta else 1
        production_time = recipe_meta[1] if recipe_meta else 0.0

        raw_totals: Dict[str, int] = defaultdict(int)
        visited: Set[str] = set()

        tree_node = self._build_production_node(
            ware_id=clean_ware_id,
            method=active_method,
            multiplier=1.0,
            visited=visited,
            raw_totals=raw_totals,
            current_depth=0,
            max_depth=max_depth,
        )

        return ProductionChainResult(
            target_ware_id=clean_ware_id,
            target_ware_name=ware_name,
            method=active_method,
            output_amount=output_amount,
            production_time=production_time,
            tree=tree_node,
            total_raw_materials=dict(raw_totals),
            was_method_fallback=was_fallback,
            requested_method=requested_method,
        )

    def _build_production_node(
        self,
        ware_id: str,
        method: str,
        multiplier: float,
        visited: Set[str],
        raw_totals: Dict[str, int],
        current_depth: int,
        max_depth: int,
    ) -> ProductionNode:
        cursor = self.conn.cursor()
        ware_row = cursor.execute("SELECT name FROM wares WHERE id = ?", (ware_id,)).fetchone()
        ware_name = ware_row[0] if ware_row else ware_id

        # Query direct input recipes for (ware_id, method)
        input_rows = cursor.execute(
            """
            SELECT input_ware_id, input_amount, output_amount
            FROM production_recipes
            WHERE ware_id = ? AND method = ?
            """,
            (ware_id, method),
        ).fetchall()

        if not input_rows or current_depth >= max_depth or ware_id in visited:
            # Leaf / raw material node
            needed_amount = max(1, int(round(multiplier)))
            raw_totals[ware_id] += needed_amount
            return ProductionNode(
                ware_id=ware_id,
                ware_name=ware_name,
                method=method,
                amount_needed=needed_amount,
                direct_inputs=[],
            )

        # Recursion with cycle prevention
        new_visited = set(visited)
        new_visited.add(ware_id)

        direct_nodes: List[ProductionNode] = []
        batch_output = input_rows[0][2]
        batches_needed = multiplier / float(batch_output)

        for in_id, in_amt, out_amt in input_rows:
            required_input_qty = batches_needed * float(in_amt)

            # Determine input ware recipe method
            in_methods = self.list_production_methods(in_id)
            sub_method = method if method in in_methods else (in_methods[0] if in_methods else "default")

            sub_node = self._build_production_node(
                ware_id=in_id,
                method=sub_method,
                multiplier=required_input_qty,
                visited=new_visited,
                raw_totals=raw_totals,
                current_depth=current_depth + 1,
                max_depth=max_depth,
            )
            direct_nodes.append(sub_node)

        return ProductionNode(
            ware_id=ware_id,
            ware_name=ware_name,
            method=method,
            amount_needed=max(1, int(round(multiplier))),
            direct_inputs=direct_nodes,
        )

    # -------------------------------------------------------------------------
    # Template T4: Category Listing & Filtering
    # -------------------------------------------------------------------------
    def query_t4_category_listing(
        self,
        filter_type: str,
        filter_value: str,
        limit: int = 50,
    ) -> CategoryListResult:
        """Lists entities matching filter criteria (faction, ship_class, purpose, ware_group)."""
        clean_type = filter_type.strip().lower()
        clean_val = filter_value.strip().lower()
        cursor = self.conn.cursor()

        items: List[Dict[str, Any]] = []

        if clean_type == "faction":
            sql = """
                SELECT s.id, s.name, s.class, s.purpose, f.name AS faction_name
                FROM ships s
                JOIN factions f ON s.faction_id = f.id
                WHERE LOWER(f.id) = ? OR LOWER(f.name) = ? OR LOWER(f.short_name) = ?
                ORDER BY s.class, s.name
                LIMIT ?
            """
            rows = cursor.execute(sql, (clean_val, clean_val, clean_val, limit)).fetchall()
            items = [
                {
                    "id": r[0],
                    "name": r[1],
                    "ship_class": r[2],
                    "purpose": r[3],
                    "faction_name": r[4],
                }
                for r in rows
            ]

        elif clean_type == "ship_class":
            sql = """
                SELECT id, name, purpose, hull, shields, cargo_capacity, speed
                FROM ships
                WHERE LOWER(class) = ?
                ORDER BY name
                LIMIT ?
            """
            rows = cursor.execute(sql, (clean_val, limit)).fetchall()
            items = [
                {
                    "id": r[0],
                    "name": r[1],
                    "purpose": r[2],
                    "hull": r[3],
                    "shields": r[4],
                    "cargo_capacity": r[5],
                    "speed": r[6],
                }
                for r in rows
            ]

        elif clean_type == "purpose":
            sql = """
                SELECT id, name, class, hull, cargo_capacity, speed
                FROM ships
                WHERE LOWER(purpose) = ?
                ORDER BY class, name
                LIMIT ?
            """
            rows = cursor.execute(sql, (clean_val, limit)).fetchall()
            items = [
                {
                    "id": r[0],
                    "name": r[1],
                    "ship_class": r[2],
                    "hull": r[3],
                    "cargo_capacity": r[4],
                    "speed": r[5],
                }
                for r in rows
            ]

        elif clean_type in ("ware_group", "ware_category", "category"):
            sql = """
                SELECT id, name, category, min_price, avg_price, max_price, volume
                FROM wares
                WHERE LOWER(category) = ?
                ORDER BY name
                LIMIT ?
            """
            rows = cursor.execute(sql, (clean_val, limit)).fetchall()
            items = [
                {
                    "id": r[0],
                    "name": r[1],
                    "category": r[2],
                    "min_price": r[3],
                    "avg_price": r[4],
                    "max_price": r[5],
                    "volume": r[6],
                }
                for r in rows
            ]

        else:
            raise ValueError(
                f"Unsupported T4 filter_type '{filter_type}'. Allowed types: 'faction', 'ship_class', 'purpose', 'ware_group'."
            )

        return CategoryListResult(
            category_type=clean_type,
            category_value=clean_val,
            items=items,
        )
