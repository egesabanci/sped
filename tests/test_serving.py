"""Tests for inference serving backend (Phase 6)."""

import torch
import pytest


# ── Backend Config Tests ─────────────────────────────────


class TestBackendConfig:
    def test_default_config(self):
        from sped.serving import BackendConfig

        cfg = BackendConfig(model_id="test/model")
        assert cfg.model_id == "test/model"
        assert cfg.dtype == "auto"
        assert cfg.device == "auto"
        assert cfg.max_length == 8192

    def test_custom_config(self):
        from sped.serving import BackendConfig

        cfg = BackendConfig(
            model_id="test/model",
            device="cuda",
            dtype="bfloat16",
            max_length=4096,
            quantization="4bit",
        )
        assert cfg.device == "cuda"
        assert cfg.dtype == "bfloat16"
        assert cfg.max_length == 4096
        assert cfg.quantization == "4bit"


# ── Generation Result Tests ──────────────────────────────


class TestGenerationResult:
    def test_tokens_per_second(self):
        from sped.serving import GenerationResult

        r = GenerationResult(text="hello world", tokens=10, time_seconds=2.0)
        assert r.tokens_per_second == 5.0

    def test_zero_time(self):
        from sped.serving import GenerationResult

        r = GenerationResult(text="hello", tokens=5, time_seconds=0.0)
        assert r.tokens_per_second == 0.0


# ── HF Backend Tests (unit, no model loading) ────────────


class TestHFBackend:
    def test_init(self):
        from sped.serving.hf_backend import HFBackend

        backend = HFBackend()
        assert backend._model is None
        assert backend._tokenizer is None

    def test_resolve_device_auto_returns_cpu(self):
        from sped.serving.hf_backend import HFBackend

        device = HFBackend._resolve_device("auto")
        assert device in ("cuda", "mps", "cpu")

    def test_resolve_dtype(self):
        from sped.serving.hf_backend import HFBackend

        assert HFBackend._resolve_dtype("float32") == torch.float32
        assert HFBackend._resolve_dtype("float16") == torch.float16

    def test_close_clears_model(self):
        from sped.serving.hf_backend import HFBackend

        backend = HFBackend()
        backend.close()
        assert backend._model is None
        assert backend._tokenizer is None


# ── MLX Backend Tests (availability check only) ──────────


class TestMLXBackend:
    def test_init(self):
        from sped.serving.mlx_backend import MLXBackend

        backend = MLXBackend()
        assert backend._model is None
        assert backend._tokenizer is None

    def test_is_available_returns_bool(self):
        from sped.serving.mlx_backend import MLXBackend

        result = MLXBackend.is_available()
        assert isinstance(result, bool)

    def test_close_clears_model(self):
        from sped.serving.mlx_backend import MLXBackend

        backend = MLXBackend()
        backend.close()
        assert backend._model is None
        assert backend._tokenizer is None


# ── vLLM Backend Tests (import only) ─────────────────────


class TestVLLMBackend:
    def test_init(self):
        from sped.serving.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        assert backend._llm is None

    def test_get_logits_raises(self):
        from sped.serving.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        with pytest.raises(NotImplementedError):
            backend.get_logits(None)


# ── Serve CLI Tests ──────────────────────────────────────


class TestServeCLI:
    def test_serve_help(self):
        """Subprocess smoke test for serve help."""
        import subprocess
        result = subprocess.run(
            ["sped", "serve", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "run" in result.stdout.lower()

    def test_serve_run_help(self):
        """Subprocess smoke test for serve run help."""
        import subprocess
        result = subprocess.run(
            ["sped", "serve", "run", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--target" in result.stdout
        assert "--draft" in result.stdout
        assert "--backend" in result.stdout
        assert "--benchmark" in result.stdout
        assert "--quantization" in result.stdout

    def test_serve_run_requires_target(self):
        """serve run requires --target."""
        import subprocess
        result = subprocess.run(
            ["sped", "serve", "run"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


# ── Backend Resolution Tests ─────────────────────────────


class TestBackendResolution:
    def test_specific_backend(self):
        """Explicit backend choice is respected."""
        from sped.cli.serve import _resolve_backend

        assert _resolve_backend("hf") == "hf"
        assert _resolve_backend("mlx") == "mlx"

    def test_auto_returns_string(self):
        from sped.cli.serve import _resolve_backend

        result = _resolve_backend("auto")
        assert result in ("hf", "mlx", "vllm")

    def test_create_hf_backend(self):
        from sped.cli.serve import _create_backend
        from sped.serving.hf_backend import HFBackend

        backend = _create_backend("hf")
        assert isinstance(backend, HFBackend)


# ── Import Tests ─────────────────────────────────────────


class TestServingImports:
    def test_import_serving_modules(self):
        from sped.serving import InferenceBackend, BackendConfig, GenerationResult
        assert InferenceBackend is not None
        assert BackendConfig is not None
        assert GenerationResult is not None

    def test_import_hf_backend(self):
        from sped.serving.hf_backend import HFBackend
        assert HFBackend is not None

    def test_import_mlx_backend(self):
        from sped.serving.mlx_backend import MLXBackend
        assert MLXBackend is not None

    def test_import_vllm_backend(self):
        from sped.serving.vllm_backend import VLLMBackend
        assert VLLMBackend is not None
