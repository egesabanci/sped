"""Hugging Face Transformers backend for sped.

Supports CUDA, MPS, CPU auto-detection and AWQ/GPTQ/bitsandbytes quantization.
"""

from time import time
from typing import Optional, Generator
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
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
        self._param_count: Optional[float] = None
        self._quantization: Optional[str] = None

    def load_model(self, config: BackendConfig):
        self._device = self._resolve_device(config.device)
        self._max_length = config.max_length

        # Resolve dtype
        torch_dtype = self._resolve_dtype(config.dtype)

        # Build quantization config
        quantization_kwargs = self._build_quantization_kwargs(config.quantization)

        # Flash Attention 2: enable if flash-attn is installed
        fa2_kwargs = {}
        try:
            import flash_attn  # noqa: F401
            fa2_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

        self._model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch_dtype,
            device_map=self._device,
            trust_remote_code=True,
            **quantization_kwargs,
            **fa2_kwargs,
        )
        self._model.eval()

        # Track param count and quantization
        self._param_count = sum(p.numel() for p in self._model.parameters()) / 1e9
        self._quantization = config.quantization

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

    @property
    def param_count(self) -> Optional[float]:
        """Return parameter count in billions."""
        return self._param_count

    @property
    def quantization(self) -> Optional[str]:
        """Return quantization method used."""
        return self._quantization

    def close(self):
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve device string, trying CUDA -> MPS -> CPU.

        Catches errors at each step so 'auto' never crashes.
        """
        if device == "auto":
            # Try CUDA
            try:
                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
            # Try MPS
            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
            # Fallback to CPU
            return "cpu"

        # Explicit device — validate it exists
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Device '{device}' requested but CUDA is not available. "
                    "Use --device cpu or install PyTorch with CUDA support."
                )
            if device == "cuda":
                return "cuda"
            # cuda:N — validate index
            try:
                idx = int(device.split(":")[1])
                if idx >= torch.cuda.device_count():
                    raise RuntimeError(
                        f"CUDA device {idx} requested but only "
                        f"{torch.cuda.device_count()} devices available."
                    )
            except (IndexError, ValueError):
                raise RuntimeError(f"Invalid CUDA device spec: '{device}'. Use cuda:N or cuda.")

        if device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError(
                    "Device 'mps' requested but MPS is not available. "
                    "Use --device cpu."
                )

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

    @staticmethod
    def _build_quantization_kwargs(quantization: Optional[str]) -> dict:
        """Build quantization kwargs for model loading.

        Supports:
        - '4bit' / '8bit': bitsandbytes quantization
        - 'awq': AWQ (auto-detected from model config)
        - 'gptq': GPTQ (auto-detected from model config)
        - None: no quantization

        Raises a clear error if bitsandbytes is required but not installed.
        """
        if quantization is None:
            return {}

        if quantization in ("4bit", "8bit"):
            try:
                import bitsandbytes  # noqa: F401
            except ImportError:
                raise ImportError(
                    f"bitsandbytes is required for {quantization} quantization. "
                    "Install with: pip install bitsandbytes"
                )

            if quantization == "4bit":
                return {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    ),
                }
            elif quantization == "8bit":
                return {
                    "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                }
        elif quantization in ("awq", "gptq"):
            # AWQ/GPTQ models are auto-detected by Transformers from
            # their quantize_config.json. We just pass device_map.
            return {}
        else:
            raise ValueError(f"Unknown quantization: {quantization}")
