"""Output formatting utilities for sped CLI.

Supports text (Rich tables), JSON (machine-parseable), and silent modes.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TextIO

from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


# Shared console instance
_console = Console()


def print_results(
    results: list[dict[str, Any]],
    format: str = "text",
    title: str = "Results",
    output_file: Optional[Path] = None,
):
    """Print generation results in the requested format.

    Args:
        results: List of result dictionaries.
        format: 'text', 'json', or 'silent'.
        title: Title for the table (text mode only).
        output_file: Optional path to also write output.
    """
    if format == "silent":
        return

    if format == "json":
        output = json.dumps(results, indent=2, default=str)
        _console.print(output)
        if output_file:
            output_file.write_text(output)
        return

    # Default: text
    if not results:
        _console.print("[yellow]No results.[/yellow]")
        return

    # Try to build a table from keys of first result
    first = results[0]
    table = Table(title=title, header_style="bold")
    for key in first:
        table.add_column(key.replace("_", " ").title(), style="cyan", no_wrap=False)

    for row in results:
        table.add_row(*[str(v) if v is not None else "\u2014" for v in row.values()])

    _console.print(table)

    if output_file:
        _console.print(f"[dim]Results saved to: {output_file}[/dim]")


def print_json(data: Any):
    """Print data as JSON to stdout."""
    json.dump(data, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def save_results_json(
    data: Any,
    path: Path,
    timestamp: bool = True,
) -> Path:
    """Save data as JSON to a file, optionally with timestamp.

    Args:
        data: Data to serialize.
        path: Output path (directory or file).
        timestamp: If True and path is a directory, create timestamped file.

    Returns:
        Path to the saved file.
    """
    path = Path(path)

    if path.is_dir() or (not path.suffix):
        path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path / f"sped_results_{ts}.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def print_model_info(
    model_id: str,
    params: Optional[float] = None,
    device: str = "cpu",
    quantization: Optional[str] = None,
    memory_mb: Optional[float] = None,
    format: str = "text",
):
    """Print model information."""
    if format == "json":
        info = {
            "model_id": model_id,
            "device": device,
        }
        if params:
            info["param_count_billions"] = round(params, 2)
        if quantization:
            info["quantization"] = quantization
        if memory_mb:
            info["memory_mb"] = round(memory_mb, 1)
        print_json(info)
        return

    if format == "silent":
        return

    desc = f"[bold cyan]{model_id}[/bold cyan]"
    details = f"  Device: {device}"
    if params:
        details += f"  |  {params:.1f}B params"
    if quantization:
        details += f"  |  Quant: {quantization}"
    if memory_mb:
        details += f"  |  ~{memory_mb:.0f} MB"

    rprint(Panel(f"{desc}\n{details}", expand=False))
