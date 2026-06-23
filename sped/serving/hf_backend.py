"""Hugging Face Transformers backend for sped."""

from time import time
from typing import Optional, Generator
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .base import InferenceBackend, BackendConfig, GenerationResult


class HFBackend(InferenceBackend):
    """Inference backend using Hugging Face Transformers.

    Default backend — works everywhere, supports all model formats.
    """

    def __init__(self):
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._device: str = "cpu"
        self._max_length: int = 8192

    def load_model(self, config: BackendConfig):
        self._device = self._resolve_device(config.device)
        self._max_length = config.max_length

        # Resolve dtype
        torch_dtype = self._resolve_dtype(config.dtype)

        # Quantization
        quantization_kwargs = {}
        if config.quantization == "4bit":
            quantization_kwargs = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
            }
        elif config.quantization == "8bit":
            quantization_kwargs = {"load_in_8bit": True}

        self._model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch_dtype,
            device_map=self._device,
            **quantization_kwargs,
        )
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def load_tokenizer(self, model_id: str):
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
        if stream:
            return self._generate_streaming(prompt, max_new_tokens, temperature)

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

    def _generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        """Stream tokens one at a time using model.generate with streamer."""
        from transformers import TextStreamer

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
        with torch.no_grad():
            outputs = self._model(input_ids)
        return outputs.logits

    @property
    def model(self) -> PreTrainedModel:
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    @property
    def device(self) -> str:
        return self._device

    def close(self):
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype:
        mapping = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(dtype, "auto")
