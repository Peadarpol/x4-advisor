"""CLI entrypoint for unstructured curation: registration, dual-loop verification, human approval, and chunking/embedding ingestion."""

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import sys
from typing import List, Optional

from x4_advisor.config import get_config
from x4_advisor.curation.chunker import MarkdownChunker
from x4_advisor.curation.claim_verifier import ClaimVerifier
from x4_advisor.curation.models import TypedClaim
from x4_advisor.embeddings.ollama_embedder import OllamaEmbedder
from x4_advisor.storage.schema import init_db_schema

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def register_source(
    db_path: Path,
    source_id: str,
    url: str,
    title: str,
    proposed_by: str,
    category: str,
    topic_tags: Optional[str] = None,
    trust_rationale: Optional[str] = None,
    status: str = "trusted",
) -> None:
    """Registers a candidate source in source_registry table."""
    conn = sqlite3.connect(str(db_path))
    init_db_schema(conn)
    cursor = conn.cursor()

    now_str = datetime.now().isoformat()
    cursor.execute(
        """
        INSERT OR REPLACE INTO source_registry (
            source_id, url, title, proposed_by, category, topic_tags, proposed_date,
            status, trust_rationale, reviewed_by, reviewed_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_id,
            url,
            title,
            proposed_by,
            category,
            topic_tags,
            now_str,
            status,
            trust_rationale,
            "operator",
            now_str,
        ),
    )
    conn.commit()
    conn.close()
    logger.info(f"Registered source '{source_id}' ({title}) with status '{status}'.")


def verify_source_claims(
    db_path: Path,
    manifest_id: str,
    source_id: str,
    title: str,
    c1_path: Path,
    c2_path: Path,
) -> None:
    """Runs dual-loop verification (C1 vs C2 fidelity and C1 vs DB fact check) and records manifest status."""
    if not c1_path.exists() or not c2_path.exists():
        raise FileNotFoundError(f"Claims JSON files not found: {c1_path} or {c2_path}")

    c1_hash = compute_sha256(c1_path)

    with open(c1_path, "r", encoding="utf-8") as f:
        c1_data = json.load(f)
    with open(c2_path, "r", encoding="utf-8") as f:
        c2_data = json.load(f)

    c1_claims = [TypedClaim.from_dict(d) for d in c1_data]
    c2_claims = [TypedClaim.from_dict(d) for d in c2_data]

    verifier = ClaimVerifier()

    # 1. Fidelity check
    fidelity_diffs = verifier.verify_fidelity(c1_claims, c2_claims)
    fidelity_flags = [d for d in fidelity_diffs if d.status != "match"]

    # 2. Database fact check
    db_diffs = verifier.verify_against_db(c1_claims, db_path=db_path)
    db_flags = [d for d in db_diffs if d.status == "mismatch"]

    num_fidelity_flags = len(fidelity_flags)
    num_db_flags = len(db_flags)

    curation_status = "flagged_review" if (num_fidelity_flags + num_db_flags) > 0 else "claims_extracted"

    conn = sqlite3.connect(str(db_path))
    init_db_schema(conn)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO source_manifest (
            manifest_id, source_id, title, file_path, curation_status, raw_hash,
            claims_hash, fidelity_discrepancies, db_discrepancies
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            manifest_id,
            source_id,
            title,
            str(c1_path),
            curation_status,
            c1_hash,  # Using c1_hash for raw/claims tracking
            c1_hash,
            num_fidelity_flags,
            num_db_flags,
        ),
    )
    conn.commit()
    conn.close()

    logger.info(
        f"Verification complete for manifest '{manifest_id}': status='{curation_status}', "
        f"fidelity_flags={num_fidelity_flags}, db_flags={num_db_flags}."
    )


def approve_manifest(db_path: Path, manifest_id: str, operator_name: str = "operator") -> None:
    """Explicitly transitions source manifest status to 'approved' (Mandatory Human Gate)."""
    conn = sqlite3.connect(str(db_path))
    init_db_schema(conn)
    cursor = conn.cursor()

    row = cursor.execute(
        "SELECT curation_status FROM source_manifest WHERE manifest_id = ?",
        (manifest_id,),
    ).fetchone()

    if not row:
        conn.close()
        raise ValueError(f"Manifest '{manifest_id}' not found.")

    now_str = datetime.now().isoformat()
    cursor.execute(
        """
        UPDATE source_manifest
        SET curation_status = 'approved', approved_at = ?, approved_by = ?
        WHERE manifest_id = ?
        """,
        (now_str, operator_name, manifest_id),
    )
    conn.commit()
    conn.close()
    logger.info(f"Manifest '{manifest_id}' explicitly approved by '{operator_name}'.")


def reset_manifest(db_path: Path, manifest_id: str) -> None:
    """Reverts source manifest status back to 'flagged_review' and invalidates approval fields."""
    conn = sqlite3.connect(str(db_path))
    init_db_schema(conn)
    cursor = conn.cursor()

    row = cursor.execute(
        "SELECT curation_status FROM source_manifest WHERE manifest_id = ?",
        (manifest_id,),
    ).fetchone()

    if not row:
        conn.close()
        raise ValueError(f"Manifest '{manifest_id}' not found.")

    cursor.execute(
        """
        UPDATE source_manifest
        SET curation_status = 'flagged_review', approved_at = NULL, approved_by = NULL
        WHERE manifest_id = ?
        """,
        (manifest_id,),
    )
    conn.commit()
    conn.close()
    logger.info(f"Manifest '{manifest_id}' status reset to 'flagged_review' and approval invalidated.")


def ingest_manifest(
    db_path: Path,
    manifest_id: str,
    paraphrase_path: Path,
    c1_path: Path,
    force: bool = False,
    ollama_endpoint: str = "http://localhost:11434",
    embedding_model: str = "qwen3-embedding:0.6b",
) -> None:
    """Chunks paraphrased Markdown, computes embeddings, and stores vectors into database."""
    conn = sqlite3.connect(str(db_path))
    init_db_schema(conn)
    cursor = conn.cursor()

    manifest_row = cursor.execute(
        "SELECT title, curation_status, claims_hash FROM source_manifest WHERE manifest_id = ?",
        (manifest_id,),
    ).fetchone()

    if not manifest_row:
        conn.close()
        raise ValueError(f"Manifest '{manifest_id}' not found.")

    title, status, stored_claims_hash = manifest_row
    if status != "approved":
        conn.close()
        raise PermissionError(
            f"Cannot ingest manifest '{manifest_id}': curation status is '{status}', but 'approved' is required."
        )

    # Re-check claims_hash integrity (F2.2 Resolution)
    current_claims_hash = compute_sha256(c1_path)
    if current_claims_hash != stored_claims_hash:
        conn.close()
        raise ValueError(
            f"Claims hash mismatch for '{manifest_id}': Claims file was modified since verification! "
            f"Re-run verification pass before ingesting."
        )

    # Check for existing ingested chunks (F1.4 / --force safety guard)
    existing_count = cursor.execute(
        "SELECT COUNT(*) FROM knowledge_chunks WHERE manifest_id = ?",
        (manifest_id,),
    ).fetchone()[0]

    if existing_count > 0:
        if not force:
            conn.close()
            logger.info(
                f"Manifest '{manifest_id}' already has {existing_count} ingested chunks. "
                "Use --force to overwrite existing chunks."
            )
            return

        # Purge existing chunks and vectors if --force is enabled
        cursor.execute(
            "DELETE FROM knowledge_chunks_vec WHERE chunk_id LIKE ?",
            (f"chunk_{manifest_id}_%",),
        )
        cursor.execute("DELETE FROM knowledge_chunks WHERE manifest_id = ?", (manifest_id,))
        conn.commit()

    with open(paraphrase_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    chunker = MarkdownChunker()
    chunks = chunker.chunk_markdown(
        markdown_text=md_text,
        source_attribution=f"X4 Advisor Knowledge — {title}",
    )

    if not chunks:
        logger.warning(f"No chunks generated for manifest '{manifest_id}'.")
        conn.close()
        return

    embedder = OllamaEmbedder(endpoint=ollama_endpoint, model_name=embedding_model)

    now_str = datetime.now().isoformat()
    for idx, c in enumerate(chunks):
        chunk_id = f"chunk_{manifest_id}_{c.chunk_index:03d}"

        # Generate 1024-dim embedding
        vector = embedder.embed_text(c.content)
        import struct

        raw_vec_bytes = struct.pack(f"{len(vector)}f", *vector)

        # 1. Insert relational metadata chunk
        cursor.execute(
            """
            INSERT OR REPLACE INTO knowledge_chunks (
                id, manifest_id, heading_hierarchy, chunk_index, content, token_count,
                source_attribution, topic, related_entity_ids, game_version_scope, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                manifest_id,
                c.heading_hierarchy,
                c.chunk_index,
                c.content,
                c.word_count,  # Storing word_count as token approximation
                c.source_attribution,
                c.topic,
                None,
                "base_game",
                now_str,
            ),
        )

        # 2. Insert vector into sqlite-vec table
        cursor.execute(
            "INSERT INTO knowledge_chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, raw_vec_bytes),
        )

    conn.commit()
    conn.close()
    logger.info(f"Ingested {len(chunks)} chunks and vectors into database for manifest '{manifest_id}'.")


def main() -> None:
    """CLI subcommands entrypoint."""
    cfg = get_config(validate=False)

    parser = argparse.ArgumentParser(description="X4 Advisor Unstructured Knowledge Base Curation Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: register
    p_reg = subparsers.add_parser("register", help="Register candidate source in source_registry")
    p_reg.add_argument("--source-id", required=True, help="Unique source identifier (e.g. src_001)")
    p_reg.add_argument("--url", required=True, help="Source URL or origin reference")
    p_reg.add_argument("--title", required=True, help="Title of source guide/wiki page")
    p_reg.add_argument("--proposed-by", default="peter_manual", help="Discovery channel")
    p_reg.add_argument("--category", default="forum_guide", help="Source category")

    # Subcommand: verify
    p_ver = subparsers.add_parser("verify", help="Run dual-loop verification on C1 and C2 claims")
    p_ver.add_argument("--manifest-id", required=True, help="Manifest ID")
    p_ver.add_argument("--source-id", required=True, help="Registered source ID")
    p_ver.add_argument("--title", required=True, help="Title")
    p_ver.add_argument("--c1-path", required=True, type=Path, help="Path to initial C1 claims JSON")
    p_ver.add_argument("--c2-path", required=True, type=Path, help="Path to re-extracted C2 claims JSON")

    # Subcommand: approve
    p_app = subparsers.add_parser("approve", help="Explicit human approval of verified source manifest")
    p_app.add_argument("--manifest-id", required=True, help="Manifest ID to approve")

    # Subcommand: reset
    p_res = subparsers.add_parser("reset", help="Reset manifest status back to flagged_review and invalidate approval")
    p_res.add_argument("--manifest-id", required=True, help="Manifest ID to reset")

    # Subcommand: ingest
    p_ing = subparsers.add_parser("ingest", help="Chunk and embed approved paraphrased text into vector database")
    p_ing.add_argument("--manifest-id", required=True, help="Approved Manifest ID")
    p_ing.add_argument("--paraphrase-path", required=True, type=Path, help="Path to paraphrased Markdown P")
    p_ing.add_argument("--c1-path", required=True, type=Path, help="Path to initial C1 claims JSON (for hash integrity re-check)")
    p_ing.add_argument("--force", action="store_true", help="Force re-ingest and overwrite existing chunks")

    args = parser.parse_args()

    _ = cfg.sources_path  # Ensure data/sources directory exists on disk
    db_path = cfg.database_path

    if args.command == "register":
        register_source(
            db_path=db_path,
            source_id=args.source_id,
            url=args.url,
            title=args.title,
            proposed_by=args.proposed_by,
            category=args.category,
        )
    elif args.command == "verify":
        verify_source_claims(
            db_path=db_path,
            manifest_id=args.manifest_id,
            source_id=args.source_id,
            title=args.title,
            c1_path=args.c1_path,
            c2_path=args.c2_path,
        )
    elif args.command == "approve":
        approve_manifest(db_path=db_path, manifest_id=args.manifest_id)
    elif args.command == "reset":
        reset_manifest(db_path=db_path, manifest_id=args.manifest_id)
    elif args.command == "ingest":
        ingest_manifest(
            db_path=db_path,
            manifest_id=args.manifest_id,
            paraphrase_path=args.paraphrase_path,
            c1_path=args.c1_path,
            force=args.force,
            ollama_endpoint=cfg.ollama_endpoint,
            embedding_model=cfg.embedding_model,
        )


if __name__ == "__main__":
    main()
