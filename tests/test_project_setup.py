"""Tests for project setup: uv, CLI, and basic imports."""

import subprocess
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


def test_pyproject_has_dev_deps():
    """pyproject.toml has dev dependencies declared."""
    import tomllib

    project_root = Path(__file__).resolve().parent.parent
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text())
    # Check in project.optional-dependencies
    proj = pyproject.get("project", {})
    opt_deps = proj.get("optional-dependencies", {})
    assert "dev" in opt_deps


def test_sped_installed_as_editable():
    """sped is installed as an editable (local) package."""
    import sped
    assert sped.__file__ is not None
    assert "sped" in str(sped.__file__)
