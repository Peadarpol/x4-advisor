"""Configuration management for X4 Advisor."""

import os
from pathlib import Path
from typing import Optional


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


def get_config(validate: bool = True) -> Config:
    """Returns validated Config instance."""
    return Config(validate=validate)
