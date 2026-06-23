"""Inference backend abstraction for sped.

Supports multiple backends:
- Hugging Face Transformers (default, works everywhere)
- MLX (Apple Silicon optimized)
- vLLM (production serving with continuous batching)
"""

from .base import InferenceBackend, BackendConfig, GenerationResult
from .hf_backend import HFBackend
from .mlx_backend import MLXBackend
from .vllm_backend import VLLMBackend

__all__ = [
    "InferenceBackend",
    "BackendConfig",
    "GenerationResult",
    "HFBackend",
    "MLXBackend",
    "VLLMBackend",
]
