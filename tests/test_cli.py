"""Tests for CLI commands (Phase 1)."""

import subprocess
import sys
from pathlib import Path


def _run_sped(*args: str) -> subprocess.CompletedProcess:
    """Run a sped CLI command and return the result."""
    return subprocess.run(
        ["sped", *args],
        capture_output=True,
        text=True,
    )


# ── Top-level commands ────────────────────────────────────


def test_version():
    result = _run_sped("version")
    assert result.returncode == 0
    assert "sped v0.1.0" in result.stdout


def test_info():
    result = _run_sped("info")
    assert result.returncode == 0
    assert "sped System Info" in result.stdout
    assert "PyTorch" in result.stdout


# ── list subcommands ──────────────────────────────────────


def test_list_help():
    result = _run_sped("list", "--help")
    assert result.returncode == 0
    assert "models" in result.stdout.lower()
    assert "adapters" in result.stdout.lower()
    assert "pairings" in result.stdout.lower()


def test_list_models():
    result = _run_sped("list", "models")
    assert result.returncode == 0
    assert "Draft Models" in result.stdout
    assert "Target Models" in result.stdout
    assert "Qwen" in result.stdout
    assert "Llama" in result.stdout


def test_list_models_with_query():
    result = _run_sped("list", "models", "--query", "qwen")
    assert result.returncode == 0
    assert "Qwen" in result.stdout
    assert "Llama" not in result.stdout


def test_list_adapters_no_adapters():
    result = _run_sped("list", "adapters")
    assert result.returncode == 0
    assert "No adapters found" in result.stdout


def test_list_pairings():
    result = _run_sped("list", "pairings")
    assert result.returncode == 0
    assert "Draft" in result.stdout
    assert "Target" in result.stdout
    assert "Vocab Match" in result.stdout


# ── distil subcommands ────────────────────────────────────


def test_distil_help():
    result = _run_sped("distil", "--help")
    assert result.returncode == 0
    assert "run" in result.stdout.lower()
    assert "validate" in result.stdout.lower()


def test_distil_run_help():
    result = _run_sped("distil", "run", "--help")
    assert result.returncode == 0
    assert "--draft" in result.stdout
    assert "--target" in result.stdout
    assert "--dataset" in result.stdout
    assert "--lora-rank" in result.stdout
    assert "--epochs" in result.stdout
    assert "--output" in result.stdout


def test_distil_run_requires_draft():
    result = _run_sped("distil", "run", "--target", "test")
    assert result.returncode != 0
    assert "Error" in result.stderr or "Missing" in result.stderr


def test_distil_run_requires_target():
    result = _run_sped("distil", "run", "--draft", "test")
    assert result.returncode != 0
    assert "Error" in result.stderr or "Missing" in result.stderr


def test_distil_validate_help():
    result = _run_sped("distil", "validate", "--help")
    assert result.returncode == 0
    assert "--draft" in result.stdout
    assert "--target" in result.stdout
    assert "--num-prompts" in result.stdout


# ── serve subcommands ─────────────────────────────────────


def test_serve_help():
    result = _run_sped("serve", "--help")
    assert result.returncode == 0
    assert "run" in result.stdout.lower()


def test_serve_run_help():
    result = _run_sped("serve", "run", "--help")
    assert result.returncode == 0
    assert "--target" in result.stdout
    assert "--draft" in result.stdout
    assert "--draft-k" in result.stdout
    assert "--align" in result.stdout
    assert "--benchmark" in result.stdout


def test_serve_run_requires_target():
    result = _run_sped("serve", "run")
    assert result.returncode != 0
    assert "Error" in result.stderr or "required" in result.stderr


# ── experiment subcommands ────────────────────────────────


def test_experiment_help():
    result = _run_sped("experiment", "--help")
    assert result.returncode == 0
    assert "run" in result.stdout.lower()
    assert "auto-tune" in result.stdout.lower()


def test_experiment_run_help():
    result = _run_sped("experiment", "run", "--help")
    assert result.returncode == 0
    assert "--target" in result.stdout
    assert "--draft" in result.stdout
    assert "--draft-k-values" in result.stdout
    assert "--temperatures" in result.stdout
    assert "--align-strategies" in result.stdout
    assert "--output" in result.stdout


def test_experiment_auto_tune_help():
    result = _run_sped("experiment", "auto-tune", "--help")
    assert result.returncode == 0
    assert "--target" in result.stdout
    assert "--min-k" in result.stdout
    assert "--max-k" in result.stdout
    assert "--num-prompts" in result.stdout
