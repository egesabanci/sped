"""Abstract inference backend interface for sped."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationResult:
    """Result from a generation call."""

    text: str
    tokens: int = 0
    time_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.time_seconds <= 0:
            return 0.0
        return self.tokens / self.time_seconds


@dataclass
class BackendConfig:
    """Configuration for loading a model on a backend."""

    model_id: str
    dtype: str = "auto"
    device: str = "auto"
    max_length: int = 8192
    gpu_memory_utilization: float = 0.9
    kv_cache_dtype: str = "auto"
    quantization: Optional[str] = None  # "4bit", "8bit", None


class InferenceBackend(ABC):
    """Abstract base class for inference backends.

    Each backend implements model loading, text generation, and
    logit access for speculative decoding.
    """

    @abstractmethod
    def load_model(self, config: BackendConfig):
        """Load a model. Called once at startup."""
        ...

    @abstractmethod
    def load_tokenizer(self, model_id: str):
        """Load the tokenizer for the model."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> GenerationResult:
        """Generate text. If stream=True, yields tokens incrementally."""
        ...

    @abstractmethod
    def get_logits(
        self,
        input_ids,
    ):
        """Get logits for a given input (used in speculative verification)."""
        ...

    @property
    @abstractmethod
    def model(self):
        """Access the underlying model (for speculative decoding)."""
        ...

    @property
    @abstractmethod
    def tokenizer(self):
        """Access the tokenizer."""
        ...

    def close(self):
        """Clean up resources."""
        pass
