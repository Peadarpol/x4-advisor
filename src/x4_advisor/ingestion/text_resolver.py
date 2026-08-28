"""Text catalog localization resolver for X4 language files."""

import logging
from pathlib import Path
import re
from typing import Dict, Set, Tuple
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class TextResolver:
    """Parses X4 text catalog XML (e.g. t/0001-l044.xml) and resolves text references."""

    def __init__(self) -> None:
        self.raw_text_map: Dict[Tuple[int, int], str] = {}
        self.resolved_text_map: Dict[Tuple[int, int], str] = {}

    @property
    def text_map(self) -> Dict[Tuple[int, int], str]:
        """Provides backwards-compatible access to resolved entries."""
        return self.resolved_text_map

    def load_from_file(self, xml_path: Path) -> None:
        """Pass 1: Parses language catalog XML and loads all raw text entries without premature modification."""
        if not xml_path.exists():
            logger.warning(f"Text catalog file '{xml_path}' does not exist.")
            return

        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()

            for page_elem in root.findall("page"):
                try:
                    page_id = int(page_elem.attrib.get("id", "0"))
                except ValueError:
                    continue

                for t_elem in page_elem.findall("t"):
                    try:
                        t_id = int(t_elem.attrib.get("id", "0"))
                    except ValueError:
                        continue

                    raw_text = t_elem.text or ""
                    self.raw_text_map[(page_id, t_id)] = raw_text

            # Pass 2: Pre-resolve all loaded entries
            for key in list(self.raw_text_map.keys()):
                self._resolve_entry(key[0], key[1], set())

        except Exception as e:
            logger.error(f"Error parsing text catalog '{xml_path}': {e}")

    def _resolve_entry(self, page_id: int, t_id: int, visited: Set[Tuple[int, int]]) -> str:
        """Recursively resolves embedded references with cycle protection and cleans text output."""
        key = (page_id, t_id)
        if key in self.resolved_text_map:
            return self.resolved_text_map[key]
        if key in visited:
            return ""

        raw = self.raw_text_map.get(key, "")
        if not raw:
            return ""

        visited.add(key)
        raw_stripped = raw.strip()

        # 1. Paren-first title annotation wrapper: e.g. (Cerberus Vanguard){20101,10601}
        paren_first = re.match(r"^\(((?:[^()\\]|\\.)*)\)\s*\{(\d+)\s*,\s*(\d+)\}", raw_stripped)
        if paren_first:
            candidate = paren_first.group(1).replace(r"\(", "(").replace(r"\)", ")").strip()
            if candidate and not candidate.startswith("plea") and not candidate.startswith("Email") and not candidate.startswith("same as"):
                self.resolved_text_map[key] = candidate
                return candidate

        # 2. Ref-first with trailing comment: e.g. {20003,10001} {20402,1}(Grand Exchange I)
        ref_trailing = re.search(r"\{\d+\s*,\s*\d+\}.*?\(((?:[^()\\]|\\.)*)\)$", raw_stripped)
        if ref_trailing:
            candidate = ref_trailing.group(1).replace(r"\(", "(").replace(r"\)", ")").strip()
            if candidate and not candidate.startswith("plea") and not candidate.startswith("Email") and not candidate.startswith("same as"):
                self.resolved_text_map[key] = candidate
                return candidate

        # 3. Recursively substitute all embedded {page, id} references
        def _sub_ref(match: re.Match) -> str:
            p = int(match.group(1))
            t = int(match.group(2))
            return self._resolve_entry(p, t, visited.copy())

        had_refs = bool(re.search(r"\{\d+\s*,\s*\d+\}", raw_stripped))
        substituted = re.sub(r"\{(\d+)\s*,\s*(\d+)\}", _sub_ref, raw_stripped).strip()

        # 4. If string had refs and became fully wrapped in parens, extract inner title
        if had_refs:
            paren_full = re.fullmatch(r"\(((?:[^()\\]|\\.)*)\)", substituted)
            if paren_full:
                candidate = paren_full.group(1).replace(r"\(", "(").replace(r"\)", ")").strip()
                if candidate and not candidate.startswith("plea") and not candidate.startswith("Email") and not candidate.startswith("same as"):
                    substituted = candidate

        # 5. Unescape literal parens & clean whitespace
        cleaned = substituted.replace(r"\(", "(").replace(r"\)", ")")
        cleaned = re.sub(r"\\\\n|\n", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        self.resolved_text_map[key] = cleaned
        return cleaned

    def _clean_text(self, text: str) -> str:
        """Standalone cleaner for raw strings."""
        if not text:
            return ""
        cleaned = text.replace(r"\(", "(").replace(r"\)", ")")
        cleaned = re.sub(r"\\\\n|\n", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def resolve(self, text_ref: str, default: str = "") -> str:
        """Resolves a string reference like '{20101,10602}' or returns the raw text if not a reference."""
        if not text_ref:
            return default

        text_ref_str = str(text_ref).strip()
        match = re.fullmatch(r"\{(\d+)\s*,\s*(\d+)\}", text_ref_str)
        if match:
            page_id = int(match.group(1))
            t_id = int(match.group(2))
            if (page_id, t_id) in self.raw_text_map:
                return self._resolve_entry(page_id, t_id, set())
            return default

        # Embedded reference substitution if partial match
        if re.search(r"\{(\d+)\s*,\s*(\d+)\}", text_ref_str):
            def _sub_ref(m: re.Match) -> str:
                p = int(m.group(1))
                t = int(m.group(2))
                return self._resolve_entry(p, t, set())

            text_ref_str = re.sub(r"\{(\d+)\s*,\s*(\d+)\}", _sub_ref, text_ref_str)

        cleaned = self._clean_text(text_ref_str)
        return cleaned if cleaned else default
