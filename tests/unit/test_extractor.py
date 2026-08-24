"""Unit tests for x4cat subprocess extractor wrapper and DLC boundary guard."""

from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from x4_advisor.ingestion.extractor import (
    DLCBoundaryError,
    ExtractorError,
    extract_catalog_files,
    validate_install_path,
)


def test_validate_install_path_valid(tmp_path: Path):
    """Validates that a directory containing 01.cat passes guard checks."""
    (tmp_path / "01.cat").touch()
    # Should not raise
    validate_install_path(tmp_path)


def test_validate_install_path_missing_root_cat(tmp_path: Path):
    """Validates that a directory without 01.cat raises DLCBoundaryError."""
    with pytest.raises(DLCBoundaryError, match="does not contain root catalog"):
        validate_install_path(tmp_path)


def test_validate_install_path_extensions_directory(tmp_path: Path):
    """Validates that pointing directly inside an extensions directory raises DLCBoundaryError."""
    ext_dir = tmp_path / "extensions" / "ego_dlc_split"
    ext_dir.mkdir(parents=True)
    (ext_dir / "01.cat").touch()

    with pytest.raises(DLCBoundaryError, match="violating base-game boundary"):
        validate_install_path(ext_dir)


@patch("subprocess.run")
def test_extract_catalog_files_command_construction(mock_run: MagicMock, tmp_path: Path):
    """Mocks subprocess.run to assert exact command vector passed to x4cat extract."""
    (tmp_path / "01.cat").touch()
    output_dir = tmp_path / "output"

    mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")

    extract_catalog_files(tmp_path, output_dir)

    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    cmd = args[0]

    assert cmd[0] == "x4cat"
    assert cmd[1] == "extract"
    assert cmd[2] == "-o"
    assert cmd[3] == str(output_dir)
    assert cmd[4] == "-p"
    assert cmd[5] == ""
    assert cmd[6] == "-g"
    assert cmd[7] == "*.xml"
    assert cmd[8] == str(tmp_path.resolve())
    assert kwargs.get("check") is True


@patch("subprocess.run")
def test_extract_catalog_files_x4cat_not_found(mock_run: MagicMock, tmp_path: Path):
    """Verifies ExtractorError is raised when x4cat is not installed on PATH."""
    (tmp_path / "01.cat").touch()
    mock_run.side_effect = FileNotFoundError()

    with pytest.raises(ExtractorError, match="x4cat executable was not found on PATH"):
        extract_catalog_files(tmp_path, tmp_path / "out")


@patch("subprocess.run")
def test_extract_catalog_files_called_process_error(mock_run: MagicMock, tmp_path: Path):
    """Verifies ExtractorError is raised on non-zero subprocess exit code."""
    (tmp_path / "01.cat").touch()
    mock_run.side_effect = subprocess.CalledProcessError(1, ["x4cat"], stderr="Fatal error")

    with pytest.raises(ExtractorError, match="x4cat extract failed with exit code 1"):
        extract_catalog_files(tmp_path, tmp_path / "out")


@patch("subprocess.run")
def test_extract_catalog_files_timeout(mock_run: MagicMock, tmp_path: Path):
    """Verifies ExtractorError is raised when subprocess times out."""
    (tmp_path / "01.cat").touch()
    mock_run.side_effect = subprocess.TimeoutExpired(["x4cat"], 300)

    with pytest.raises(ExtractorError, match="timed out after 300 seconds"):
        extract_catalog_files(tmp_path, tmp_path / "out")
