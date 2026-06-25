"""Shared 4-bit model load caching for Unsloth.

When loading a model with ``load_in_4bit=True``, Unsloth/bitsandbytes
quantizes bf16 weights to NF4 on-the-fly. For an 8B model this takes
~132s on every cold load. By saving the quantized weights to a
``{model_name}-4bit-cache`` directory, subsequent loads skip the
quantization step and complete in ~26s (5× faster).

Used by both ``sped distil run`` and ``sped serve --backend unsloth``.

Cache invalidation: the cache is keyed on the model path only. RoPE /
max_seq_length configuration is rebuilt at load time, so a cache created
with one ``max_seq_length`` can be reused with another. If the source
weights change (e.g. a new model revision), delete the cache directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Any


def _cache_dir_for(model_name: str, max_seq_length: Optional[int] = None) -> str:
    """Compute the 4-bit cache directory for a model.

    When ``max_seq_length`` is provided it is folded into the cache name
    so that incompatible RoPE caches are not reused. In practice Unsloth
    rebuilds the RoPE cache at load time so this is conservative; users
    who know their cache is seq-length independent can disable this via
    ``max_seq_length=None``.
    """
    base = model_name.rstrip("/")
    if max_seq_length is not None and max_seq_length > 0:
        return f"{base}-4bit-cache-{max_seq_length}"
    return f"{base}-4bit-cache"


def load_unsloth_model(
    model_name: str,
    *,
    max_seq_length: int = 4096,
    load_in_4bit: bool = True,
    device: str = "cuda",
    dtype: Any = None,
    cache_seq_key: bool = False,
    verbose: bool = False,
) -> Tuple[Any, Any]:
    """Load a model via ``FastLanguageModel`` with optional 4-bit caching.

    Args:
        model_name: HuggingFace model ID or local path.
        max_seq_length: Maximum sequence length for the model.
        load_in_4bit: If True, load with NF4 4-bit quantization (and cache).
        device: Device map string passed to ``FastLanguageModel``.
        dtype: Optional torch dtype override (None = let unsloth decide).
        cache_seq_key: If True, include ``max_seq_length`` in the cache
            directory name to avoid reusing a cache built with a different
            sequence length. Defaults to False because Unsloth rebuilds
            RoPE caches at load time.
        verbose: If True, print cache hit/miss messages.

    Returns:
        ``(model, tokenizer)`` tuple from ``FastLanguageModel.from_pretrained``.
    """
    from unsloth import FastLanguageModel  # noqa: WPS433

    if not load_in_4bit:
        # Full precision load (bf16) — no caching needed
        return FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=False,
            device_map=device,
        )

    cache_path = _cache_dir_for(
        model_name, max_seq_length if cache_seq_key else None,
    )
    cache = Path(cache_path)

    if cache.exists():
        if verbose:
            print(f"  (loading from 4-bit cache: {cache_path})")
        return FastLanguageModel.from_pretrained(
            model_name=cache_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=True,
            device_map=device,
        )

    if verbose:
        print(f"  (first load — quantizing to 4-bit, may take a few minutes)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
        device_map=device,
    )
    # Persist the quantized weights for fast reload next time
    try:
        model.save_pretrained(cache_path)
        tokenizer.save_pretrained(cache_path)
        if verbose:
            print(f"  Saved 4-bit cache to {cache_path}")
    except Exception:  # pragma: no cover — best-effort cache write
        pass
    return model, tokenizer