"""Tests for project setup: uv, CLI, and basic imports."""

import subprocess
import sys
from pathlib import Path

import sped


def test_version():
    """Package exposes __version__."""
    assert sped.__version__ == "0.1.0"


def test_cli_version():
    """`sped version` prints the correct version string."""
    result = subprocess.run(
        ["sped", "version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "sped v0.1.0" in result.stdout


def test_cli_info():
    """`sped info` runs without error and shows system info."""
    result = subprocess.run(
        ["sped", "info"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "sped System Info" in result.stdout
    assert "PyTorch" in result.stdout


def test_uv_lock_exists():
    """uv.lock is present for reproducible installs."""
    project_root = Path(__file__).resolve().parent.parent
    assert (project_root / "uv.lock").exists()


def test_pyproject_has_uv_section():
    """pyproject.toml declares dev dependency group."""
    import tomllib

    project_root = Path(__file__).resolve().parent.parent
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text())
    assert "dependency-groups" in pyproject
    assert "dev" in pyproject["dependency-groups"]
    assert "pytest" in str(pyproject["dependency-groups"]["dev"])


def test_sped_installed_as_editable():
    """sped is installed as an editable (local) package."""
    result = subprocess.run(
        [sys.executable, "-c", "import sped; print(sped.__file__)"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    # Should point to the local project directory
    assert "sped" in result.stdout
    assert result.stdout.strip().startswith("/Users/egesabanci/Desktop/sped")
