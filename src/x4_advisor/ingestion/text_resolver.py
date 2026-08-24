"""Text catalog localization resolver for X4 language files."""

import logging
from pathlib import Path
import re
from typing import Dict, Tuple
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class TextResolver:
    """Parses X4 text catalog XML (e.g. t/0001-l044.xml) and resolves text references."""

    def __init__(self) -> None:
        self.text_map: Dict[Tuple[int, int], str] = {}

    def load_from_file(self, xml_path: Path) -> None:
        """Parses a language catalog XML file and populates the internal text map."""
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
                    cleaned = self._clean_text(raw_text)
                    if cleaned:
                        self.text_map[(page_id, t_id)] = cleaned

        except Exception as e:
            logger.error(f"Error parsing text catalog '{xml_path}': {e}")

    def _clean_text(self, text: str) -> str:
        """Cleans X4 text format strings (extracting human readable text from comments or references)."""
        if not text:
            return ""

        # If format is '(Cerberus Vanguard){20101,10601}', extract parenthesized text
        paren_match = re.match(r"^\((.*?)\)", text.strip())
        if paren_match:
            candidate = paren_match.group(1).strip()
            if candidate and not candidate.startswith("plea") and not candidate.startswith("Email"):
                return candidate

        # Remove trailing {page, id} references
        cleaned = re.sub(r"\{.*?\}", "", text).strip()
        # Remove leftover control markers
        cleaned = re.sub(r"\\\\n|\n", " ", cleaned).strip()
        return cleaned

    def resolve(self, text_ref: str, default: str = "") -> str:
        """Resolves a string reference like '{20101,10602}' or returns the raw text if not a reference."""
        if not text_ref:
            return default

        text_ref_str = str(text_ref).strip()
        match = re.search(r"\{(\d+)\s*,\s*(\d+)\}", text_ref_str)
        if match:
            page_id = int(match.group(1))
            t_id = int(match.group(2))

            if (page_id, t_id) in self.text_map:
                return self.text_map[(page_id, t_id)]

        # If it's not a reference format or reference not found, clean and return
        cleaned = self._clean_text(text_ref_str)
        return cleaned if cleaned else (text_ref_str if not text_ref_str.startswith("{") else default)
