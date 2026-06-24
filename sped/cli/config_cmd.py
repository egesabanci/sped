"""sped config — YAML-based configuration management.

Supports three config layers:
1. Built-in defaults (hardcoded)
2. ~/.sped/config.yml (user-level, created by `config init`)
3. .sped.yml in project directory (project-level overrides)

CLI args always override config values.
"""

import os
from pathlib import Path
from typing import Any, Optional
import yaml

from rich import print as rprint
from rich.table import Table
from rich.panel import Panel

import typer


# ── Defaults ──────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "target": None,
    "draft": None,
    "backend": "auto",
    "device": "auto",
    "draft_k": 5,
    "max_new_tokens": 512,
    "temperature": 0.0,
    "align": "auto",
    "log_level": "info",
    "output_format": "text",
    "quantization": None,
}

CONFIG_DIR = Path.home() / ".sped"
USER_CONFIG_PATH = CONFIG_DIR / "config.yml"
PROJECT_CONFIG_NAME = ".sped.yml"


# ── Config loading ────────────────────────────────────────


def load_config() -> dict[str, Any]:
    """Load config from disk, merging layers.

    Resolution order (last wins): built-in -> user -> project
    """
    config = dict(DEFAULT_CONFIG)

    # User-level (~/.sped/config.yml)
    if USER_CONFIG_PATH.exists():
        with open(USER_CONFIG_PATH) as f:
            user_cfg = yaml.safe_load(f) or {}
        config.update(user_cfg)

    # Project-level (.sped.yml)
    project_path = _find_project_config()
    if project_path:
        with open(project_path) as f:
            proj_cfg = yaml.safe_load(f) or {}
        config.update(proj_cfg)

    return config


def save_config(config: dict[str, Any], path: Optional[Path] = None):
    """Save config to a YAML file."""
    save_path = path or USER_CONFIG_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(
            {k: v for k, v in config.items() if v is not None},
            f,
            default_flow_style=False,
            sort_keys=False,
        )


def _find_project_config() -> Optional[Path]:
    """Walk up from cwd looking for .sped.yml."""
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        candidate = parent / PROJECT_CONFIG_NAME
        if candidate.exists():
            return candidate
        # Stop at filesystem root
        if parent.parent == parent:
            break
    return None


def _get_config_path() -> Path:
    """Get the path to the user-level config file, creating dir if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return USER_CONFIG_PATH


# ── CLI helpers ────────────────────────────────────────────


def show_config():
    """Display the current effective configuration."""
    config = load_config()
    source = "~/.sped/config.yml"
    project_path = _find_project_config()
    if project_path:
        source += f" + {project_path}"

    rprint(Panel(f"[bold]Configuration[/bold] (from {source})", expand=False))
    table = Table(show_header=False, padding=(0, 2))
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for key, value in config.items():
        display = str(value) if value is not None else "[dim]not set[/dim]"
        table.add_row(key, display)
    rprint(table)
    rprint(f"\nConfig file: {USER_CONFIG_PATH}")


def set_config_value(key: str, value: str):
    """Set a single config value and save."""
    valid_keys = set(DEFAULT_CONFIG.keys())
    if key not in valid_keys:
        rprint(f"[red]Error:[/red] Unknown config key '{key}'.")
        rprint(f"Valid keys: {', '.join(sorted(valid_keys))}")
        return False

    # Type coercion
    default_val = DEFAULT_CONFIG[key]
    typed_value: Any = value
    if isinstance(default_val, int):
        try:
            typed_value = int(value)
        except ValueError:
            rprint(f"[red]Error:[/red] '{key}' must be an integer.")
            return False
    elif isinstance(default_val, float):
        try:
            typed_value = float(value)
        except ValueError:
            rprint(f"[red]Error:[/red] '{key}' must be a number.")
            return False
    elif isinstance(default_val, bool):
        typed_value = value.lower() in ("true", "yes", "1")

    config = load_config()
    config[key] = typed_value
    save_config(config)
    rprint(f"[green]OK[/green] Set {key} = {typed_value}")
    return True


def init_config(force: bool = False) -> bool:
    """Create default user config file if it doesn't exist."""
    path = _get_config_path()
    if path.exists() and not force:
        rprint(f"[yellow]Config already exists:[/yellow] {path}")
        rprint("Use --force to overwrite.")
        return False

    save_config(DEFAULT_CONFIG, path)
    rprint(f"[green]OK[/green] Created config: {path}")
    rprint("Edit it directly or use `sped config set <key> <value>`.")
    return True


# ── Typer CLI sub-commands ────────────────────────────────


app = typer.Typer(name="config", help="Manage configuration (YAML).", no_args_is_help=True)


@app.callback()
def callback():
    pass


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config."),
):
    """Create a default configuration file."""
    init_config(force=force)


@app.command()
def show():
    """Display the current configuration."""
    show_config()


@app.command(name="set")
def set_cmd(
    key: str = typer.Argument(..., help="Config key to set"),
    value: str = typer.Argument(..., help="Value to assign"),
):
    """Set a configuration value."""
    success = set_config_value(key, value)
    if not success:
        raise typer.Exit(code=1)
