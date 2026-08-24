"""Curation package exposing claim verification, Markdown chunking, and CLI tools."""

from x4_advisor.curation.chunker import MarkdownChunker
from x4_advisor.curation.claim_verifier import ClaimVerifier
from x4_advisor.curation.models import (
    DBVerificationResult,
    FidelityDiffResult,
    TextChunk,
    TypedClaim,
)

__all__ = [
    "ClaimVerifier",
    "MarkdownChunker",
    "TypedClaim",
    "FidelityDiffResult",
    "DBVerificationResult",
    "TextChunk",
]
