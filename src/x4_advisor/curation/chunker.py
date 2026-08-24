"""Heading-aware Markdown chunking engine with size bounds and paragraph fallback."""

import logging
import re
from typing import List, Optional

from x4_advisor.curation.models import TextChunk

logger = logging.getLogger(__name__)

# Word count bounds approximating 300–500 tokens (assuming ~1.3 tokens/word)
MIN_WORD_BOUND = 230
MAX_WORD_BOUND = 385


class MarkdownChunker:
    """Chunks Markdown documents preserving heading hierarchy breadcrumbs and token boundaries."""

    def __init__(
        self,
        min_words: int = MIN_WORD_BOUND,
        max_words: int = MAX_WORD_BOUND,
    ) -> None:
        self.min_words = min_words
        self.max_words = max_words

    def chunk_markdown(
        self,
        markdown_text: str,
        source_attribution: str,
        topic: Optional[str] = None,
    ) -> List[TextChunk]:
        """Parses Markdown content and returns a list of size-bounded TextChunk instances."""
        if not markdown_text.strip():
            return []

        sections = self._parse_sections(markdown_text)
        chunks: List[TextChunk] = []
        chunk_idx = 1

        for breadcrumb, body_text in sections:
            if not body_text.strip():
                continue

            words = body_text.split()
            word_count = len(words)

            if word_count <= self.max_words:
                chunks.append(
                    TextChunk(
                        content=body_text.strip(),
                        heading_hierarchy=breadcrumb,
                        chunk_index=chunk_idx,
                        word_count=word_count,
                        source_attribution=source_attribution,
                        topic=topic,
                    )
                )
                chunk_idx += 1
            else:
                # Oversized section: split via paragraph fallback
                sub_bodies = self._split_paragraphs(body_text)
                for sub_text in sub_bodies:
                    sub_words = len(sub_text.split())
                    chunks.append(
                        TextChunk(
                            content=sub_text.strip(),
                            heading_hierarchy=breadcrumb,
                            chunk_index=chunk_idx,
                            word_count=sub_words,
                            source_attribution=source_attribution,
                            topic=topic,
                        )
                    )
                    chunk_idx += 1

        return chunks

    def _parse_sections(self, text: str) -> List[tuple[str, str]]:
        """Parses Markdown text into (heading_breadcrumb, section_content) pairs."""
        lines = text.splitlines()
        sections: List[tuple[str, str]] = []

        current_heading_stack: List[tuple[int, str]] = []  # (level, title)
        current_lines: List[str] = []
        preamble_lines: List[str] = []

        for line in lines:
            header_match = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
            if header_match:
                # Flush previous section
                if current_heading_stack:
                    breadcrumb = " > ".join(title for _, title in current_heading_stack)
                    sections.append((breadcrumb, "\n".join(current_lines)))
                    current_lines = []
                elif preamble_lines:
                    sections.append(("Preamble", "\n".join(preamble_lines)))
                    preamble_lines = []

                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Pop headings of equal or deeper level
                while current_heading_stack and current_heading_stack[-1][0] >= level:
                    current_heading_stack.pop()

                current_heading_stack.append((level, title))
            else:
                if current_heading_stack:
                    current_lines.append(line)
                else:
                    preamble_lines.append(line)

        # Flush final section
        if current_heading_stack and current_lines:
            breadcrumb = " > ".join(title for _, title in current_heading_stack)
            sections.append((breadcrumb, "\n".join(current_lines)))
        elif preamble_lines:
            sections.append(("Preamble", "\n".join(preamble_lines)))

        return sections

    def _split_paragraphs(self, text: str) -> List[str]:
        """Splits oversized section text at paragraph boundaries to fit word bounds."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return [text]

        chunks: List[str] = []
        curr_paras: List[str] = []
        curr_words = 0

        for p in paragraphs:
            p_words = len(p.split())
            if curr_words + p_words <= self.max_words:
                curr_paras.append(p)
                curr_words += p_words
            else:
                if curr_paras:
                    chunks.append("\n\n".join(curr_paras))
                curr_paras = [p]
                curr_words = p_words

        if curr_paras:
            chunks.append("\n\n".join(curr_paras))

        return chunks
