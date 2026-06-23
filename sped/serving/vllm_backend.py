"""vLLM inference backend for sped — production-grade serving.

Requires: pip install vllm

Provides continuous batching, PagedAttention, and tensor parallelism
for high-throughput serving with speculative decoding support.
"""

from time import time
from typing import Optional

from .base import InferenceBackend, BackendConfig, GenerationResult


class VLLMBackend(InferenceBackend):
    """Inference backend using vLLM.

    Supports native speculative decoding via draft model matching.
    For cross-vocab speculation, use the HF backend with sped's
    alignment layer.
    """

    def __init__(self):
        self._llm = None
        self._tokenizer = None
        self._model = None  # Underlying HF model (for logits access)
        self._sampling_params = None

    def load_model(self, config: BackendConfig):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM backend requires `vllm`. Install with: uv pip install vllm"
            )

        # vLLM uses its own model loading
        self._llm = LLM(
            model=config.model_id,
            dtype=config.dtype if config.dtype != "auto" else "auto",
            max_model_len=config.max_length,
            gpu_memory_utilization=config.gpu_memory_utilization,
            kv_cache_dtype=config.kv_cache_dtype,
        )
        self._sampling_params = SamplingParams

        # Load HF tokenizer for compatibility
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

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
        params = self._sampling_params(
            temperature=temperature if temperature > 0 else 0.0,
            max_tokens=max_new_tokens,
        )
        start = time()
        outputs = self._llm.generate([prompt], params)
        elapsed = time() - start

        result = outputs[0]
        text = result.outputs[0].text
        tokens = len(result.outputs[0].token_ids)

        return GenerationResult(
            text=text,
            tokens=tokens,
            time_seconds=round(elapsed, 3),
        )

    def get_logits(self, input_ids):
        """vLLM does not expose raw logits directly.

        For speculative decoding with vLLM, use vLLM's native SD support
        or fall back to the HF backend for the target model.
        """
        raise NotImplementedError(
            "vLLM backend does not support raw logit access. "
            "Use vLLM's native speculative decoding or HF backend instead."
        )

    @property
    def model(self):
        return self._llm

    @property
    def tokenizer(self):
        return self._tokenizer

    def close(self):
        self._llm = None
        self._tokenizer = None
