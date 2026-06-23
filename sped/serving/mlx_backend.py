"""MLX backend for sped — Apple Silicon optimized inference.

Requires: pip install mlx-lm

MLX provides efficient inference on Apple Silicon (M1/M2/M3/M4) by
leveraging the unified memory architecture and Apple's Metal GPU.
"""

from time import time
from typing import Optional
from pathlib import Path

from .base import InferenceBackend, BackendConfig, GenerationResult


class MLXBackend(InferenceBackend):
    """Inference backend using MLX (Apple Silicon).

    Loads models via mlx-lm and provides logits for speculative decoding.
    Falls back to HF Transformers on non-Apple hardware.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._max_length = 8192

    def load_model(self, config: BackendConfig):
        self._max_length = config.max_length

        try:
            from mlx_lm import load as mlx_load
        except ImportError:
            raise ImportError(
                "MLX backend requires `mlx-lm`. Install with: uv pip install mlx-lm"
            )

        model_path = config.model_id

        # Try loading with mlx-lm
        try:
            self._model, self._tokenizer = mlx_load(model_path)
        except Exception:
            # Fallback: if the model path is a local directory with mlx weights
            model_path = Path(model_path)
            if model_path.exists():
                self._model, self._tokenizer = mlx_load(str(model_path))
            else:
                raise

    def load_tokenizer(self, model_id: str):
        from transformers import AutoTokenizer
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
        try:
            from mlx_lm import generate as mlx_generate
            import mlx.core as mx
        except ImportError:
            raise ImportError("MLX backend requires `mlx-lm`")

        start = time()

        if stream:
            # Streaming generation
            full_text = ""
            for chunk in mlx_generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=max_new_tokens,
                temp=temperature if temperature > 0 else 0.0,
                verbose=False,
                stream=True,
            ):
                full_text += chunk

            elapsed = time() - start
            tokens = len(self._tokenizer.encode(full_text))
            return GenerationResult(
                text=full_text,
                tokens=tokens,
                time_seconds=round(elapsed, 3),
            )
        else:
            text = mlx_generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=max_new_tokens,
                temp=temperature if temperature > 0 else 0.0,
                verbose=False,
            )
            elapsed = time() - start
            tokens = len(self._tokenizer.encode(text))
            return GenerationResult(
                text=text,
                tokens=tokens,
                time_seconds=round(elapsed, 3),
            )

    def get_logits(self, input_ids):
        """Get logits from MLX model for speculative decoding.

        Note: MLX speculative decoding integration requires converting
        between MLX and PyTorch tensors. This is a best-effort implementation.
        """
        try:
            import mlx.core as mx
        except ImportError:
            raise ImportError("MLX backend requires `mlx-lm`")

        # Convert PyTorch tensor to MLX array
        if hasattr(input_ids, "numpy"):
            mx_input = mx.array(input_ids.cpu().numpy())
        else:
            mx_input = input_ids

        # Forward pass through MLX model
        logits = self._model(mx_input)
        return logits

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def close(self):
        self._model = None
        self._tokenizer = None

    @staticmethod
    def is_available() -> bool:
        """Check if MLX is available and we're on Apple Silicon."""
        import platform
        if platform.processor() != "arm":
            return False
        try:
            import mlx_lm  # noqa
            return True
        except ImportError:
            return False
