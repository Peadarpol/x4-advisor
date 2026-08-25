"""Automated governance test asserting zero hardcoded model strings in operational logic."""

from pathlib import Path
import re

# Designated files and lines permitted to define default fallback constants
ALLOWLIST = {
    "config.py": {"gemma4:12b", "qwen3-embedding:0.6b"},
    "ollama_embedder.py": {"qwen3-embedding:0.6b"},
    "client.py": {"gemma4:12b"},
    "advisor_engine.py": {"gemma4:12b"},
}

MODEL_PATTERN = re.compile(r"['\"](gemma4:\w+|qwen3:\w+|granite4\.1:\w+|qwen3-embedding:\w+)['\"]")


def test_zero_unallowed_hardcoded_model_strings() -> None:
    """Scans all Python files in src/ to ensure model strings only appear in allowlisted default definitions."""
    src_dir = Path("src")
    assert src_dir.exists() and src_dir.is_dir()

    violations = []

    for py_file in src_dir.rglob("*.py"):
        rel_path = py_file.name
        content = py_file.read_text(encoding="utf-8")

        for line_num, line in enumerate(content.splitlines(), 1):
            matches = MODEL_PATTERN.findall(line)
            if not matches:
                continue

            allowed_set = ALLOWLIST.get(rel_path, set())
            for match in matches:
                if match not in allowed_set:
                    violations.append(f"{py_file}:{line_num} contains unallowed model string '{match}': {line.strip()}")

    assert not violations, "Found unallowed hardcoded model strings in src/:\n" + "\n".join(violations)
