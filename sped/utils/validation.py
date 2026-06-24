"""Input validation utilities for sped CLI.

All validation runs before any model loading — fail fast with
clear error messages showing the bad value and acceptable range.
"""

import re
from pathlib import Path
from typing import Optional


VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}
VALID_BACKENDS = {"auto", "hf", "mlx", "vllm"}
VALID_ALIGNMENTS = {"auto", "none", "string", "probabilistic", "hybrid"}
VALID_OUTPUT_FORMATS = {"text", "json", "silent"}
VALID_LOG_LEVELS = {"debug", "info", "warn", "error"}
VALID_QUANTIZATIONS = {None, "4bit", "8bit", "awq", "gptq"}


def validate_draft_k(value: int) -> int:
    """Validate 1 ≤ draft_k ≤ max_new_tokens (upper bound checked later)."""
    if not isinstance(value, int) or value < 1:
        raise ValueError(
            f"Invalid --draft-k: {value}. Must be an integer ≥ 1."
        )
    if value > 50:
        raise ValueError(
            f"Invalid --draft-k: {value}. Maximum is 50."
        )
    return value


def validate_temperature(value: float) -> float:
    """Validate 0.0 ≤ temperature ≤ 2.0."""
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"Invalid --temperature: {value}. Must be a number."
        )
    if value < 0.0 or value > 2.0:
        raise ValueError(
            f"Invalid --temperature: {value}. Must be between 0.0 and 2.0."
        )
    return value


def validate_max_new_tokens(value: int) -> int:
    """Validate 1 ≤ max_new_tokens ≤ 4096."""
    if not isinstance(value, int) or value < 1:
        raise ValueError(
            f"Invalid --max-new-tokens: {value}. Must be an integer ≥ 1."
        )
    if value > 4096:
        raise ValueError(
            f"Invalid --max-new-tokens: {value}. Maximum is 4096."
        )
    return value


def validate_device(value: str) -> str:
    """Validate device string format."""
    if value in VALID_DEVICES:
        return value
    # Allow cuda:N pattern
    if re.match(r"^cuda:\d+$", value):
        return value
    raise ValueError(
        f"Invalid --device: '{value}'. Must be one of: "
        f"{', '.join(sorted(VALID_DEVICES))} or 'cuda:N'."
    )


def validate_backend(value: str) -> str:
    """Validate backend name."""
    if value not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid --backend: '{value}'. Must be one of: "
            f"{', '.join(sorted(VALID_BACKENDS))}."
        )
    return value


def validate_align(value: str) -> str:
    """Validate alignment strategy."""
    if value not in VALID_ALIGNMENTS:
        raise ValueError(
            f"Invalid --align: '{value}'. Must be one of: "
            f"{', '.join(sorted(VALID_ALIGNMENTS))}."
        )
    return value


def validate_output_format(value: str) -> str:
    """Validate output format."""
    if value not in VALID_OUTPUT_FORMATS:
        raise ValueError(
            f"Invalid --output: '{value}'. Must be one of: "
            f"{', '.join(sorted(VALID_OUTPUT_FORMATS))}."
        )
    return value


def validate_log_level(value: str) -> str:
    """Validate log level."""
    val = value.lower()
    if val not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid --log-level: '{value}'. Must be one of: "
            f"{', '.join(sorted(VALID_LOG_LEVELS))}."
        )
    return val


def validate_model_id(value: str) -> str:
    """Validate model ID or local path exists before loading."""
    if not value or not isinstance(value, str):
        raise ValueError("Model ID must be a non-empty string.")

    # Check if it's a local path
    path = Path(value)
    if path.exists():
        if not (path / "config.json").exists():
            raise ValueError(
                f"Local model path '{value}' exists but has no config.json. "
                "Is this a valid HuggingFace model directory?"
            )
        return value

    # Remote model: validate HF naming convention
    if not re.match(r"^[\w.-]+/[\w.-]+$", value):
        raise ValueError(
            f"Invalid model ID: '{value}'. "
            "Expected format: 'owner/model-name' (e.g., 'Qwen/Qwen3-0.6B') "
            "or a valid local path."
        )

    return value


def validate_timeout(value: Optional[int]) -> Optional[int]:
    """Validate timeout value."""
    if value is None:
        return None
    if not isinstance(value, (int, float)) or value < 1:
        raise ValueError(
            f"Invalid --timeout: {value}. Must be a positive number of seconds."
        )
    if value > 3600:
        raise ValueError(
            f"Invalid --timeout: {value}. Maximum is 3600 (1 hour)."
        )
    return int(value)


def validate_draft_k_against_max(draft_k: int, max_new_tokens: int):
    """Validate draft_k ≤ max_new_tokens."""
    if draft_k > max_new_tokens:
        raise ValueError(
            f"--draft-k ({draft_k}) cannot exceed --max-new-tokens ({max_new_tokens})."
        )
