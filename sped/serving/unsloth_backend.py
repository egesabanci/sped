"""Unsloth backend for sped — fast 4-bit inference with speculative decoding.

Wraps unsloth's FastLanguageModel for optimized loading and inference.
Compatible with SpeculativeDecoder since the underlying model is a standard
HF PreTrainedModel.

Usage:
    sped serve run --backend unsloth --target <model> --device cuda

Install:
    pip install unsloth
"""

from time import time
from typing import Optional
import torch
from transformers import AutoTokenizer

from .base import InferenceBackend, BackendConfig, GenerationResult


class UnslothBackend(InferenceBackend):
    """Inference backend using Unsloth's FastLanguageModel.

    Provides 2x faster inference with 4-bit quantization.
    Falls back gracefully if unsloth is not installed.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device: str = "cpu"
        self._max_length: int = 8192
        self._param_count: Optional[float] = None

    def load_model(self, config: BackendConfig):
        """Load a model via Unsloth's FastLanguageModel.

        Args:
            config: BackendConfig with model_id, device, dtype, max_length.

        Raises:
            ImportError: If unsloth is not installed.
            RuntimeError: If model loading fails (OOM, etc.).
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise ImportError(
                "Unsloth is required for the unsloth backend. "
                "Install with: pip install unsloth"
            ) from e

        self._device = self._resolve_device(config.device)
        self._max_length = config.max_length

        # Resolve dtype for unsloth
        torch_dtype = self._resolve_dtype(config.dtype)

        try:
            model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.model_id,
                max_seq_length=config.max_length,
                dtype=torch_dtype,
                load_in_4bit=True,
                device_map=self._device,
            )
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda out of memory" in error_msg:
                raise RuntimeError(
                    f"CUDA out of memory loading '{config.model_id}' with Unsloth. "
                    "Try: --dtype float16 or a smaller model."
                ) from e
            raise RuntimeError(
                f"Failed to load model '{config.model_id}' with Unsloth: {e}"
            ) from e

        # Enable fast inference kernels
        FastLanguageModel.for_inference(model)

        self._model = model
        self._param_count = sum(p.numel() for p in model.parameters()) / 1e9

        # Ensure pad token
        if self._tokenizer is not None and self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def load_tokenizer(self, model_id: str):
        """Load tokenizer independently (used in some speculative paths)."""
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> GenerationResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        start = time()

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        elapsed = time() - start

        generated = outputs[0, inputs.input_ids.shape[-1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            tokens=len(generated),
            time_seconds=round(elapsed, 3),
        )

    def get_logits(self, input_ids) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        with torch.no_grad():
            outputs = self._model(input_ids)
        return outputs.logits

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self) -> str:
        return self._device

    @property
    def param_count(self) -> Optional[float]:
        return self._param_count

    def close(self):
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def is_available() -> bool:
        """Check if unsloth is installed."""
        try:
            import unsloth  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve device string — same logic as HFBackend."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"

        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Device '{device}' requested but CUDA is not available."
                )
            return device

        if device == "cpu":
            return "cpu"

        return device

    @staticmethod
    def _resolve_dtype(dtype: str):
        """Resolve dtype string to torch dtype or None (unsloth auto)."""
        if dtype == "auto" or dtype is None:
            return None  # Let unsloth decide
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(dtype, None)
