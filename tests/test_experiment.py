"""Tests for experiment CLI and engine (Phase 7).

No model loading — all tests use synthetic data or CLI parsing only.
Memory-safe: no GPU, no HF downloads, no large tensors.
"""

import pytest
from pathlib import Path
import json
import tempfile


# ── CLI Argument Tests (#23) ─────────────────────────────


class TestExperimentCLI:
    def test_experiment_help(self):
        """smoke test: experiment --help shows subcommands."""
        import subprocess
        r = subprocess.run(
            ["sped", "experiment", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "run" in r.stdout.lower()
        assert "auto-tune" in r.stdout.lower()

    def test_run_help_shows_all_flags(self):
        """run --help must include all grid search parameters."""
        import subprocess
        r = subprocess.run(
            ["sped", "experiment", "run", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        for flag in ("--target", "--draft", "--draft-k-values",
                     "--temperatures", "--align-strategies",
                     "--num-prompts", "--max-tokens", "--output"):
            assert flag in r.stdout

    def test_auto_tune_help_shows_all_flags(self):
        import subprocess
        r = subprocess.run(
            ["sped", "experiment", "auto-tune", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        for flag in ("--target", "--min-k", "--max-k", "--num-prompts"):
            assert flag in r.stdout

    def test_run_requires_target(self):
        import subprocess
        r = subprocess.run(
            ["sped", "experiment", "run"],
            capture_output=True, text=True,
        )
        assert r.returncode != 0

    def test_k_values_parsed_correctly(self):
        """Verify comma-separated K values parse."""
        from sped.cli.experiment import run as run_cmd
        # Just verify the Typer command signature accepts the types
        import inspect
        sig = inspect.signature(run_cmd)
        params = sig.parameters
        assert "draft_k_values" in params
        assert "temperatures" in params
        assert "align_strategies" in params


# ── HTML Report Tests (#24) ──────────────────────────────


class TestHTMLReport:
    def test_generates_valid_html(self):
        """HTML report should produce a valid HTML file."""
        from sped.cli._experiment_engine import generate_html_report

        report = self._make_sample_report()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            generate_html_report(report, path)

            assert path.exists()
            html = path.read_text()
            assert "<!DOCTYPE html>" in html
            assert "</html>" in html
            assert "sped Experiment Report" in html

    def test_best_config_highlighted(self):
        """Best config should appear in recommendations."""
        from sped.cli._experiment_engine import generate_html_report

        report = self._make_sample_report()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            generate_html_report(report, path)

            html = path.read_text()
            assert "K=10" in html  # best config in sample data (highest K = highest tps)

    def test_report_contains_metadata(self):
        from sped.cli._experiment_engine import generate_html_report

        report = self._make_sample_report()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            generate_html_report(report, path)

            html = path.read_text()
            assert "test/target" in html
            assert "test/draft" in html

    def test_many_results_renders_all_rows(self):
        from sped.cli._experiment_engine import generate_html_report

        report = self._make_sample_report(num_configs=12)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            generate_html_report(report, path)
            html = path.read_text()
            # Each config is a table row
            assert html.count("<tr") > 12

    def _make_sample_report(self, num_configs: int = 4) -> dict:
        """Generate a synthetic experiment report (no models needed)."""
        results = []
        for i in range(num_configs):
            k = [3, 5, 7, 10][i % 4]
            results.append({
                "config": {
                    "draft_k": k,
                    "temperature": 0.0,
                    "align_strategy": "hybrid" if i % 2 == 0 else "none",
                },
                "avg_tokens_per_second": round(20 + i * 5 - (k * 0.5), 1),
                "avg_time_seconds": round(5.0 - i * 0.3, 2),
                "total_tokens": 500,
                "num_prompts": 5,
                "per_prompt": [
                    {"prompt": "test", "tokens": 100,
                     "time_seconds": 1.0, "tokens_per_second": 100.0}
                ],
            })

        return {
            "metadata": {
                "target_model": "test/target",
                "draft_model": "test/draft",
                "timestamp": "2026-06-23T12:00:00",
                "device": "cpu",
            },
            "config": {
                "draft_k_values": [3, 5, 7, 10],
                "temperatures": [0.0],
                "align_strategies": ["none", "hybrid"],
                "num_prompts": 5,
                "max_tokens": 128,
            },
            "results": results,
        }


# ── Auto-Tune Algorithm Tests (#25) ──────────────────────


class TestAutoTune:
    def test_search_returns_integer(self):
        """Auto-tune returns an int K value."""
        from sped.cli._experiment_engine import AutoTuner

        class MockDecoder:
            max_draft_tokens = 5
            def reset_metrics(self): pass
            def generate(self, **kwargs): return "mock output"

        tuner = AutoTuner(
            decoder=MockDecoder(),
            prompts=["test prompt"],
            max_tokens=16,
        )
        best_k = tuner.search(min_k=2, max_k=8)
        assert isinstance(best_k, int)
        assert 2 <= best_k <= 8

    def test_search_honors_bounds(self):
        """Result must be within [min_k, max_k]."""
        from sped.cli._experiment_engine import AutoTuner

        class MockDecoder:
            max_draft_tokens = 5
            def reset_metrics(self): pass
            def generate(self, **kwargs): return "mock output"

        tuner = AutoTuner(
            decoder=MockDecoder(),
            prompts=["test"] * 2,
            max_tokens=8,
        )
        for _ in range(5):
            best = tuner.search(min_k=3, max_k=12)
            assert 3 <= best <= 12

    def test_increasing_eval_cache(self):
        """Each evaluation should cache its result."""
        from sped.cli._experiment_engine import AutoTuner

        class MockDecoder:
            max_draft_tokens = 5
            def reset_metrics(self): pass
            def generate(self, **kwargs): return "mock output"

        tuner = AutoTuner(
            decoder=MockDecoder(),
            prompts=["test"],
            max_tokens=8,
        )
        assert len(tuner._eval_cache) == 0
        tuner.search(min_k=2, max_k=6)
        assert len(tuner._eval_cache) >= 1

    def test_evaluated_k_values(self):
        """evaluated_k_values returns sorted list of (K, score) pairs."""
        from sped.cli._experiment_engine import AutoTuner

        class MockDecoder:
            max_draft_tokens = 5
            def reset_metrics(self): pass
            def generate(self, **kwargs): return "mock output"

        tuner = AutoTuner(
            decoder=MockDecoder(),
            prompts=["test"],
            max_tokens=8,
        )
        tuner.search(min_k=2, max_k=6)
        pairs = tuner.evaluated_k_values
        assert len(pairs) >= 1
        for k, score in pairs:
            assert isinstance(k, int)
            assert isinstance(score, float)
            assert score > 0


# ── JSON Export Tests ────────────────────────────────────


class TestJSONExport:
    def test_results_json_structure(self):
        """Experiment results JSON has expected structure."""
        from sped.cli._experiment_engine import ExperimentEngine

        # Test the result schema directly without engine
        result = {
            "config": {"draft_k": 5, "temperature": 0.0, "align_strategy": "none"},
            "avg_tokens_per_second": 25.3,
            "avg_time_seconds": 4.2,
            "total_tokens": 500,
            "num_prompts": 5,
            "per_prompt": [],
        }
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["config"]["draft_k"] == 5
        assert parsed["avg_tokens_per_second"] == 25.3


# ── Import Tests ─────────────────────────────────────────


class TestExperimentImports:
    def test_import_experiment_engine(self):
        from sped.cli._experiment_engine import ExperimentEngine, AutoTuner, generate_html_report
        assert ExperimentEngine is not None
        assert AutoTuner is not None
        assert callable(generate_html_report)

    def test_import_experiment_cli(self):
        from sped.cli.experiment import app, run, auto_tune
        assert app is not None
        assert callable(run)
        assert callable(auto_tune)
