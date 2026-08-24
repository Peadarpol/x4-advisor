"""Domain dataclass models for extracted X4 game data."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetMetadata:
    """Provenance and schema metadata recorded per database creation."""

    game_version: str
    build: str
    extraction_timestamp: str
    is_base_game_only: bool
    schema_version: str = "1.1.0"


@dataclass
class ShipRecord:
    """Normalized ship entity record."""

    id: str  # Game internal macro ID (e.g., 'ship_arg_m_frigate_01_a_macro')
    name: str
    ship_class: str  # e.g., 'ship_s', 'ship_m', 'ship_l', 'ship_xl', 'station'
    hull: float
    shields: float
    cargo_capacity: float
    cargo_type: Optional[str] = None  # e.g., 'container', 'solid', 'liquid'
    speed: float = 0.0
    weapon_slots: int = 0
    turret_slots: int = 0
    shield_slots: int = 0
    purpose: Optional[str] = None  # e.g., 'mine', 'trade', 'fight', 'build'
    faction_id: Optional[str] = None  # Primary maker/owner faction ID if unambiguous
    ware_id: Optional[str] = None  # Corresponding ware ID in wares table
    raw_macro: Optional[str] = None


@dataclass
class WareRecord:
    """Normalized ware entity record."""

    id: str  # Ware internal ID (e.g., 'hullparts', 'energycells')
    name: str
    category: str
    min_price: int
    avg_price: int
    max_price: int
    volume: int


@dataclass
class ProductionRecipeRecord:
    """Production recipe input component record for ware production chain traversal."""

    ware_id: str
    method: str
    input_ware_id: str
    input_amount: int
    output_amount: int
    production_time: float


@dataclass
class FactionRecord:
    """Faction entity record."""

    id: str  # Faction internal ID (e.g., 'argon', 'teladi', 'paranid')
    name: str
    short_name: Optional[str] = None
    relations_summary: Optional[str] = None


@dataclass
class SectorRecord:
    """Universe sector entity record."""

    id: str  # Sector macro or system ID
    name: str
    faction_id: Optional[str] = None
    sunlight: float = 1.0


@dataclass
class SectorResourceRecord:
    """Resource yield within a specific sector for Template T2 ranking queries."""

    sector_id: str
    resource_id: str  # e.g., 'ore', 'silicon', 'ice', 'hydrogen', 'helium', 'methane'
    resource_yield: float  # Python field name (maps to SQLite column 'yield')


@dataclass
class SourceRegistryRecord:
    """Pre-pipeline candidate source discovery and vetting record."""

    source_id: str
    url: str
    title: str
    proposed_by: str  # 'peter_manual', 'claude_search', 'chatgpt_brief'
    category: str  # 'wiki', 'forum_guide', 'steam_guide', 'other'
    proposed_date: str
    status: str = "proposed"  # 'proposed', 'trusted', 'rejected', 'superseded'
    topic_tags: Optional[str] = None
    trust_rationale: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_date: Optional[str] = None
    notes: Optional[str] = None
    content_date: Optional[str] = None
    last_checked: Optional[str] = None
    superseded_by: Optional[str] = None


@dataclass
class SourceManifestRecord:
    """Vetted source manifest tracking curation lifecycle and discrepancy counts."""

    manifest_id: str
    source_id: str
    title: str
    file_path: str
    raw_hash: str
    curation_status: str = "draft"  # 'draft', 'claims_extracted', 'flagged_review', 'approved'
    claims_hash: Optional[str] = None
    fidelity_discrepancies: int = 0
    db_discrepancies: int = 0
    approved_at: Optional[str] = None
    approved_by: Optional[str] = None


@dataclass
class KnowledgeChunkRecord:
    """Heading-aware text chunk and metadata record."""

    id: str
    manifest_id: str
    heading_hierarchy: str
    chunk_index: int
    content: str
    token_count: int
    source_attribution: str
    created_at: str
    topic: Optional[str] = None
    related_entity_ids: Optional[str] = None
    game_version_scope: str = "base_game"

