"""Configuration management for X4 Advisor."""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


def _load_dotenv_if_present(env_path: Optional[Path] = None) -> None:
    """Loads key-value pairs from .env file without overriding existing environment variables."""
    path = env_path or Path(".env")
    if not path.is_file():
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("#"):
                    continue
                if "=" not in clean_line:
                    continue
                key, val = clean_line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as e:
        logger.debug("Failed to read .env file at %s: %s", path, e)


# Load .env once on import with standard precedence (real environment variables win)
_load_dotenv_if_present()


class ConfigError(ValueError):
    """Raised when application configuration is missing or invalid."""

    pass


class Config:
    """Application settings read from environment variables or defaults."""

    def __init__(self, validate: bool = True):
        self.x4_install_path_raw: Optional[str] = os.getenv("X4_INSTALL_PATH")
        self.database_path_str: str = os.getenv(
            "DATABASE_PATH", "data/db/x4_advisor.db"
        )
        self.ollama_endpoint: str = os.getenv(
            "OLLAMA_ENDPOINT", "http://localhost:11434"
        )
        self.model_name: Optional[str] = os.getenv("MODEL_NAME")
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "qwen3-embedding:0.6b"
        )

        if validate:
            self.validate_m1_config()

    @property
    def x4_install_path(self) -> Path:
        """Returns Path object for X4_INSTALL_PATH."""
        if not self.x4_install_path_raw:
            raise ConfigError("X4_INSTALL_PATH environment variable is not set.")
        return Path(self.x4_install_path_raw)

    @property
    def database_path(self) -> Path:
        """Returns Path object for DATABASE_PATH."""
        return Path(self.database_path_str)

    @property
    def sources_path(self) -> Path:
        """Returns Path object for data/sources directory, creating it if missing."""
        path = Path("data/sources")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def vector_relevance_threshold_is_default(self) -> bool:
        """Returns True if VECTOR_RELEVANCE_THRESHOLD is not set in environment."""
        return not bool(os.getenv("VECTOR_RELEVANCE_THRESHOLD"))

    @property
    def vector_relevance_threshold(self) -> float:
        """Minimum cosine similarity for vector retrieval results.

        Empirically calibrated to 0.50 to ensure robust recall for conceptual/procedural
        knowledge queries while maintaining clean separation above out-of-domain noise (<=0.41).
        """
        raw = os.getenv("VECTOR_RELEVANCE_THRESHOLD")
        if raw:
            return float(raw)
        return 0.50

    def validate_m1_config(self) -> None:
        """Validates configuration required for Milestone M1 (Structured Extraction)."""
        if not self.x4_install_path_raw:
            raise ConfigError(
                "X4_INSTALL_PATH is not set in environment or .env file. "
                "Please configure X4_INSTALL_PATH to point to your base X4 installation directory."
            )

        install_path = self.x4_install_path
        if not install_path.exists() or not install_path.is_dir():
            raise ConfigError(
                f"X4_INSTALL_PATH '{install_path}' does not exist or is not a directory."
            )

        root_cat = install_path / "01.cat"
        if not root_cat.exists():
            raise ConfigError(
                f"X4_INSTALL_PATH '{install_path}' does not appear to be an X4 installation directory "
                "(01.cat catalog file was not found)."
            )

        # Ensure parent directory of DATABASE_PATH exists/is writable
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ConfigError(
                f"DATABASE_PATH parent directory '{self.database_path.parent}' cannot be created or accessed: {e}"
            )

    def validate_m5_config(self, probe_ollama: bool = True) -> None:
        """Validates configuration required for Milestone M5 (Router + Synthesis).

        Args:
            probe_ollama: If True, performs a live reachability probe against Ollama
                to confirm the endpoint responds and the model is installed. Defaults
                to True for application startup; pass False in unit tests.
        """
        if not self.model_name or not self.model_name.strip():
            raise ConfigError(
                "MODEL_NAME environment variable is not set. "
                "Please configure MODEL_NAME (e.g. 'MODEL_NAME=gemma4:12b') in .env or environment."
            )

        if not self.ollama_endpoint or not self.ollama_endpoint.strip():
            raise ConfigError(
                "OLLAMA_ENDPOINT environment variable is not set. "
                "Please configure OLLAMA_ENDPOINT (e.g. 'http://localhost:11434')."
            )

        if not probe_ollama:
            return

        endpoint = self.ollama_endpoint.rstrip("/")
        tags_url = f"{endpoint}/api/tags"

        try:
            req = urllib.request.Request(tags_url, headers={"User-Agent": "x4-advisor"})
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status != 200:
                    raise ConfigError(
                        f"Ollama endpoint '{endpoint}' responded with HTTP {resp.status}."
                    )
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConfigError(
                f"Cannot reach Ollama endpoint at '{endpoint}'. Is the Ollama service running? Error: {e}"
            ) from e
        except Exception as e:
            raise ConfigError(
                f"Failed to query Ollama tags at '{tags_url}': {e}"
            ) from e

        models_list = data.get("models", [])
        installed_tags: List[str] = [m.get("name", "") for m in models_list if isinstance(m, dict)]

        target_model = self.model_name.strip().lower()

        # Check exact or prefix match against installed tags
        matched_tag: Optional[str] = None
        for tag in installed_tags:
            tag_lower = tag.lower()
            if tag_lower == target_model or tag_lower == f"{target_model}:latest" or tag_lower.startswith(f"{target_model}-"):
                matched_tag = tag
                break

        if not matched_tag:
            # Fallback: check if target is a prefix of any installed tag name (e.g. gemma4:12b -> gemma4:12b-instruct-q4_K_M)
            for tag in installed_tags:
                if tag.lower().startswith(target_model):
                    matched_tag = tag
                    break

        if not matched_tag:
            installed_str = ", ".join(f"'{t}'" for t in installed_tags) if installed_tags else "None"
            raise ConfigError(
                f"Configured MODEL_NAME '{self.model_name}' is not installed in Ollama. "
                f"Installed models: [{installed_str}]. "
                f"Please pull the model using 'ollama pull {self.model_name}'."
            )

        # Update model_name to exact resolved installed tag
        if matched_tag != self.model_name:
            logger.info(
                "Resolved MODEL_NAME '%s' to exact installed Ollama tag '%s'.",
                self.model_name,
                matched_tag,
            )
            self.model_name = matched_tag


def get_config(validate: bool = True) -> Config:
    """Returns validated Config instance."""
    return Config(validate=validate)

