"""Structured logging setup for sped.

Provides:
- Console logging with rich formatting
- File logging with rotation
- JSON output mode for machine-parseable results
- Memory/GPU reporting helpers
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO
from rich.logging import RichHandler


# Module-level logger holder
_logger: Optional[logging.Logger] = None
_json_output: Optional[TextIO] = None


def setup_logging(
    log_level: str = "info",
    log_file: Optional[str] = None,
    json_mode: bool = False,
    json_file: Optional[str] = None,
) -> logging.Logger:
    """Configure logging for the sped CLI.

    Args:
        log_level: One of debug, info, warn, error.
        log_file: Optional path to write structured logs.
        json_mode: If True, also write JSON-format output to stdout.
        json_file: If set, write JSON output to this path instead of stdout.

    Returns:
        Configured logger instance.
    """
    global _logger, _json_output

    level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger("sped")
    logger.setLevel(level)

    # Remove existing handlers to allow reconfiguration
    logger.handlers.clear()

    # Rich console handler
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=log_level == "debug",
        omit_repeated_times=False,
    )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = _resolve_log_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_path}")

    # JSON output
    if json_mode and json_file:
        json_path = Path(json_file)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        _json_output = open(json_path, "w", encoding="utf-8")
        logger.info(f"JSON output to {json_path}")
    elif json_mode:
        _json_output = sys.stdout

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Get the configured logger, creating a default one if needed."""
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


def write_json_output(data: dict):
    """Write a JSON object to the configured JSON output stream."""
    global _json_output
    if _json_output is not None and not _json_output.closed:
        json.dump(data, _json_output, indent=2, default=str)
        _json_output.write("\n")
        if _json_output is sys.stdout:
            _json_output.flush()


def close_json_output():
    """Close the JSON output file if one was opened."""
    global _json_output
    if _json_output is not None and _json_output is not sys.stdout:
        _json_output.close()
    _json_output = None


def _resolve_log_path(log_file: str) -> Path:
    """Resolve log file path. Supports ~ expansion and relative paths."""
    path = Path(log_file).expanduser()
    if path.is_absolute():
        return path
    # Default to ~/.sped/logs/
    return Path.home() / ".sped" / "logs" / path


def log_model_info(
    logger: logging.Logger,
    model_type: str,
    model_id: str,
    device: str,
    quantization: Optional[str] = None,
    param_count: Optional[float] = None,
    memory_estimate: Optional[str] = None,
):
    """Log model loading information in a structured way."""
    info = {
        "event": "model_loaded",
        "type": model_type,
        "model_id": model_id,
        "device": device,
    }
    if quantization:
        info["quantization"] = quantization
    if param_count:
        info["param_count_billions"] = round(param_count, 2)
    if memory_estimate:
        info["memory_estimate"] = memory_estimate

    msg = f"{model_type} model: {model_id} ({device})"
    if param_count:
        msg += f" [{param_count:.1f}B params]"
    if quantization:
        msg += f" | quant: {quantization}"
    logger.info(msg)
    write_json_output(info)


def log_generation_result(
    logger: logging.Logger,
    tokens: int,
    time_seconds: float,
    throughput: float,
    speedup: Optional[float] = None,
    acceptance_rate: Optional[float] = None,
    prompt: Optional[str] = None,
):
    """Log generation results in a structured way."""
    result = {
        "event": "generation_complete",
        "tokens": tokens,
        "time_seconds": round(time_seconds, 3),
        "throughput_tok_s": round(throughput, 2),
    }
    if speedup is not None:
        result["speedup_vs_vanilla"] = round(speedup, 3)
    if acceptance_rate is not None:
        result["acceptance_rate"] = round(acceptance_rate, 3)
    if prompt:
        result["prompt"] = prompt[:100]

    msg = f"{tokens} tokens in {time_seconds:.1f}s ({throughput:.1f} tok/s)"
    if speedup:
        msg += f" | {speedup:.2f}x speedup"
    if acceptance_rate:
        msg += f" | accept: {acceptance_rate:.1%}"
    logger.info(msg)
    write_json_output(result)
