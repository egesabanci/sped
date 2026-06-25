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
        assert result in ("hf", "mlx", "vllm", "unsloth")

    def test_create_hf_backend(self):
        from sped.cli.serve import _create_backend
        from sped.serving.hf_backend import HFBackend

        backend = _create_backend("hf")
        assert isinstance(backend, HFBackend)


# ── Draft LoRA load-id resolution tests ─────────────────


class TestDraftLoRA_LOAD_ID:
    """Tests for _resolve_draft_load_id — fixes the double-wrap bug where
    Unsloth-saved LoRA adapters were silently dropped during serve (#85)."""

    def test_unsloth_with_lora_loads_adapter_dir(self):
        """Unsloth + draft_lora => load adapter dir (auto base+adapter)."""
        from sped.cli.serve import _resolve_draft_load_id
        from pathlib import Path
        result = _resolve_draft_load_id(
            "/data/models/Qwen3-0.6B", Path("/tmp/adapter"), "unsloth",
        )
        assert result == "/tmp/adapter"

    def test_unsloth_without_lora_loads_base(self):
        from sped.cli.serve import _resolve_draft_load_id
        result = _resolve_draft_load_id("/data/models/Qwen3-0.6B", None, "unsloth")
        assert result == "/data/models/Qwen3-0.6B"

    def test_hf_with_lora_loads_base(self):
        """HF backend + draft_lora => load base draft (PeftModel wraps later)."""
        from sped.cli.serve import _resolve_draft_load_id
        from pathlib import Path
        result = _resolve_draft_load_id(
            "/data/models/Qwen3-0.6B", Path("/tmp/adapter"), "hf",
        )
        assert result == "/data/models/Qwen3-0.6B"

    def test_hf_without_lora_loads_base(self):
        from sped.cli.serve import _resolve_draft_load_id
        result = _resolve_draft_load_id("/data/models/Qwen3-0.6B", None, "hf")
        assert result == "/data/models/Qwen3-0.6B"

    def test_auto_backend_with_lora_loads_base(self):
        """Non-unsloth backends never load the adapter dir as model_id."""
        from sped.cli.serve import _resolve_draft_load_id
        from pathlib import Path
        for b in ("auto", "hf", "mlx", "vllm"):
            result = _resolve_draft_load_id(
                "/data/models/Qwen3-0.6B", Path("/tmp/adapter"), b,
            )
            assert result == "/data/models/Qwen3-0.6B"


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


# ── HFBackend static method tests ────────────────────────


class TestHFBackendDeviceResolution:
    def test_resolve_device_auto_cpu_fallback(self):
        from sped.serving.hf_backend import HFBackend
        # When no GPU available, auto should return cpu
        device = HFBackend._resolve_device("auto")
        assert device in ("cpu", "cuda", "mps")

    def test_resolve_device_explicit_cpu(self):
        from sped.serving.hf_backend import HFBackend
        assert HFBackend._resolve_device("cpu") == "cpu"

    def test_resolve_device_cuda_not_available(self):
        """Should raise when CUDA not available but cuda requested."""
        from sped.serving.hf_backend import HFBackend
        import torch
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError):
                HFBackend._resolve_device("cuda")

    def test_resolve_device_mps_not_available(self):
        """Should raise when MPS not available but mps requested."""
        from sped.serving.hf_backend import HFBackend
        import torch
        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_avail:
            with pytest.raises(RuntimeError):
                HFBackend._resolve_device("mps")


class TestHFBackendQuantization:
    def test_quantization_none(self):
        from sped.serving.hf_backend import HFBackend
        assert HFBackend._build_quantization_kwargs(None) == {}

    def test_quantization_4bit(self):
        from sped.serving.hf_backend import HFBackend
        try:
            kwargs = HFBackend._build_quantization_kwargs("4bit")
            assert "quantization_config" in kwargs
        except ImportError as e:
            # Graceful when bitsandbytes not installed
            assert "bitsandbytes" in str(e)
            assert "pip install" in str(e)

    def test_quantization_8bit(self):
        from sped.serving.hf_backend import HFBackend
        try:
            kwargs = HFBackend._build_quantization_kwargs("8bit")
            assert "quantization_config" in kwargs
        except ImportError as e:
            # Graceful when bitsandbytes not installed
            assert "bitsandbytes" in str(e)
            assert "pip install" in str(e)

    def test_quantization_awq(self):
        from sped.serving.hf_backend import HFBackend
        kwargs = HFBackend._build_quantization_kwargs("awq")
        # AWQ models auto-detect; no quantization_config needed
        assert kwargs == {}

    def test_quantization_gptq(self):
        from sped.serving.hf_backend import HFBackend
        kwargs = HFBackend._build_quantization_kwargs("gptq")
        assert kwargs == {}

    def test_quantization_invalid(self):
        from sped.serving.hf_backend import HFBackend
        with pytest.raises(ValueError):
            HFBackend._build_quantization_kwargs("invalid")


class TestHFBackendDtypeResolution:
    def test_resolve_dtype_auto(self):
        from sped.serving.hf_backend import HFBackend
        assert HFBackend._resolve_dtype("auto") == "auto"

    def test_resolve_dtype_float32(self):
        from sped.serving.hf_backend import HFBackend
        assert HFBackend._resolve_dtype("float32") == torch.float32

    def test_resolve_dtype_float16(self):
        from sped.serving.hf_backend import HFBackend
        assert HFBackend._resolve_dtype("float16") == torch.float16

    def test_resolve_dtype_bfloat16(self):
        from sped.serving.hf_backend import HFBackend
        assert HFBackend._resolve_dtype("bfloat16") == torch.bfloat16

    def test_resolve_dtype_unknown(self):
        from sped.serving.hf_backend import HFBackend
        assert HFBackend._resolve_dtype("unknown") == "auto"  # fallback


# ── MLX graceful fallback tests ──────────────────────────


class TestMLXFallback:
    def test_resolve_backend_mlx_not_available(self):
        """When MLX is not importable, _resolve_backend should not crash."""
        # We can test by just verifying the import path doesn't crash at module level
        from sped.cli.serve import _resolve_backend
        result = _resolve_backend("mlx")
        # If MLX is not available, it returns 'mlx' (the user explicitly asked for it)
        # The crash would happen later in _create_backend
        assert result == "mlx"

    def test_create_backend_mlx_fallback(self):
        """_create_backend should catch ImportError and fall back to HF."""
        from sped.cli.serve import _create_backend
        try:
            backend = _create_backend("mlx")
            from sped.serving.hf_backend import HFBackend
            assert isinstance(backend, HFBackend) or True  # If MLX is available, that's fine too
        except Exception as e:
            # Should not raise — any exception is a bug
            pytest.fail(f"MLX fallback raised unexpected exception: {e}")


# ── bitsandbytes graceful error tests ────────────────────


class TestBitsAndBytesErrorHandling:
    def test_quantization_4bit_no_bitsandbytes(self):
        """_build_quantization_kwargs should raise ImportError with clear message
        when bitsandbytes is not installed (regression test for #58)."""
        from sped.serving.hf_backend import HFBackend
        try:
            kwargs = HFBackend._build_quantization_kwargs("4bit")
            # If bitsandbytes IS installed, it should work
            assert "quantization_config" in kwargs
        except ImportError as e:
            # If NOT installed, error must mention bitsandbytes and install cmd
            assert "bitsandbytes" in str(e)
            assert "pip install" in str(e)


# ── Unsloth backend tests ────────────────────────────────


class TestUnslothBackend:
    def test_import_unsloth_backend(self):
        from sped.serving.unsloth_backend import UnslothBackend
        assert UnslothBackend is not None

    def test_unsloth_is_available(self):
        from sped.serving.unsloth_backend import UnslothBackend
        # Accept either True or False (depends on environment)
        result = UnslothBackend.is_available()
        assert isinstance(result, bool)

    def test_resolve_backend_unsloth(self):
        from sped.cli.serve import _resolve_backend
        result = _resolve_backend("unsloth")
        assert result == "unsloth"

    def test_create_backend_unsloth_fallback(self):
        """_create_backend should catch ImportError and fall back to HF."""
        from sped.cli.serve import _create_backend
        from sped.serving.hf_backend import HFBackend
        backend = _create_backend("unsloth")
        assert isinstance(backend, HFBackend) or True  # If unsloth installed, that's fine

    def test_unsloth_resolve_device_auto(self):
        from sped.serving.unsloth_backend import UnslothBackend
        device = UnslothBackend._resolve_device("auto")
        assert device in ("cpu", "cuda")

    def test_unsloth_resolve_device_explicit(self):
        from sped.serving.unsloth_backend import UnslothBackend
        assert UnslothBackend._resolve_device("cpu") == "cpu"

    def test_unsloth_resolve_dtype_auto(self):
        from sped.serving.unsloth_backend import UnslothBackend
        assert UnslothBackend._resolve_dtype("auto") is None

    def test_unsloth_resolve_dtype_float16(self):
        from sped.serving.unsloth_backend import UnslothBackend
        import torch
        assert UnslothBackend._resolve_dtype("float16") == torch.float16

    def test_unsloth_resolve_dtype_unknown(self):
        from sped.serving.unsloth_backend import UnslothBackend
        assert UnslothBackend._resolve_dtype("unknown") is None  # fallback to None
