"""Tests for CLI commands (Phase 1 + Phase 6).

All tests that hit external APIs are marked with a timeout.
Network-dependent tests use @pytest.mark.network to allow filtering.
"""

import subprocess
import pytest


def _run_sped(*args: str, timeout: int = 10) -> subprocess.CompletedProcess:
    """Run a sped CLI command and return the result."""
    return subprocess.run(
        ["sped", *args],
        capture_output=True, text=True,
        timeout=timeout,
    )


# ── Top-level commands ────────────────────────────────────


def test_version():
    r = _run_sped("version")
    assert r.returncode == 0
    assert "sped v0.1.0" in r.stdout


def test_info():
    r = _run_sped("info")
    assert r.returncode == 0
    assert "sped System Info" in r.stdout
    assert "PyTorch" in r.stdout


# ── list subcommands ──────────────────────────────────────


def test_list_help():
    r = _run_sped("list", "--help")
    assert r.returncode == 0
    assert "models" in r.stdout.lower()


def test_list_models():
    """Only checks output structure, not full HF API query."""
    r = _run_sped("list", "models")
    assert r.returncode == 0


def test_list_adapters_no_adapters():
    r = _run_sped("list", "adapters")
    assert r.returncode == 0


def test_list_pairings():
    r = _run_sped("list", "pairings")
    assert r.returncode == 0
    assert "Draft" in r.stdout
    assert "Target" in r.stdout


# ── distil subcommands ────────────────────────────────────


def test_distil_help():
    r = _run_sped("distil", "--help")
    assert r.returncode == 0
    assert "run" in r.stdout.lower()


def test_distil_run_help():
    r = _run_sped("distil", "run", "--help")
    assert r.returncode == 0
    assert "--draft" in r.stdout
    assert "--target" in r.stdout


def test_distil_run_requires_draft():
    r = _run_sped("distil", "run", "--target", "test")
    assert r.returncode != 0


def test_distil_run_requires_target():
    r = _run_sped("distil", "run", "--draft", "test")
    assert r.returncode != 0


def test_distil_validate_help():
    r = _run_sped("distil", "validate", "--help")
    assert r.returncode == 0


# ── serve subcommands ─────────────────────────────────────


def test_serve_help():
    r = _run_sped("serve", "--help")
    assert r.returncode == 0


def test_serve_run_help():
    r = _run_sped("serve", "run", "--help")
    assert r.returncode == 0
    assert "--target" in r.stdout
    assert "--backend" in r.stdout


def test_serve_run_requires_target():
    r = _run_sped("serve", "run")
    assert r.returncode != 0


# ── experiment subcommands ────────────────────────────────


def test_experiment_help():
    r = _run_sped("experiment", "--help")
    assert r.returncode == 0
    assert "run" in r.stdout.lower()


def test_experiment_run_help():
    r = _run_sped("experiment", "run", "--help")
    assert r.returncode == 0
    assert "--target" in r.stdout


def test_experiment_auto_tune_help():
    r = _run_sped("experiment", "auto-tune", "--help")
    assert r.returncode == 0
    assert "--target" in r.stdout
