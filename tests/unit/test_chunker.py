"""Unit tests for MarkdownChunker heading-aware chunking and size bounds."""

import pytest

from x4_advisor.curation.chunker import MarkdownChunker


def test_chunk_markdown_heading_breadcrumbs():
    """Verifies chunker splits Markdown at headings and tracks breadcrumbs."""
    md = """# Mining Guide
## L-Class Ships
The Crane Vanguard is a large miner.

## M-Class Ships
The Magnetar Vanguard is a medium miner.
"""
    chunker = MarkdownChunker()
    chunks = chunker.chunk_markdown(md, source_attribution="X4 Wiki")

    assert len(chunks) == 2
    assert chunks[0].heading_hierarchy == "Mining Guide > L-Class Ships"
    assert "Crane Vanguard" in chunks[0].content
    assert chunks[1].heading_hierarchy == "Mining Guide > M-Class Ships"
    assert "Magnetar Vanguard" in chunks[1].content


def test_chunk_markdown_preamble():
    """Verifies text appearing before the first heading is chunked under 'Preamble'."""
    md = """This is preamble content before any heading.

# First Heading
Content inside first heading.
"""
    chunker = MarkdownChunker()
    chunks = chunker.chunk_markdown(md, source_attribution="X4 Wiki")

    assert len(chunks) == 2
    assert chunks[0].heading_hierarchy == "Preamble"
    assert "preamble content" in chunks[0].content


def test_chunk_markdown_paragraph_fallback():
    """Verifies oversized section text splits at paragraph boundaries."""
    para1 = "Word " * 200
    para2 = "Data " * 200
    md = f"""# Massive Section
{para1}

{para2}
"""
    chunker = MarkdownChunker(max_words=250)
    chunks = chunker.chunk_markdown(md, source_attribution="X4 Wiki")

    assert len(chunks) == 2
    assert chunks[0].heading_hierarchy == "Massive Section"
    assert chunks[1].heading_hierarchy == "Massive Section"
    assert chunks[0].chunk_index == 1
    assert chunks[1].chunk_index == 2


def test_chunk_markdown_deep_headings_treated_as_body():
    """Verifies level 4 headings (####) are treated as body content rather than chunk boundaries."""
    md = """# Parent Section
## Sub Section
#### Deep Heading
Deep heading text body.
"""
    chunker = MarkdownChunker()
    chunks = chunker.chunk_markdown(md, source_attribution="X4 Wiki")

    assert len(chunks) == 1
    assert chunks[0].heading_hierarchy == "Parent Section > Sub Section"
    assert "#### Deep Heading" in chunks[0].content
