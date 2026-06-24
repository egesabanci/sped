"""Main CLI application — registers all commands and sub-groups.

Features:
- All subcommands with auto-generated help
- Shell completion (bash/zsh/fish) via typer
- Global --output / --log-level flags on info/version
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

app = typer.Typer(
    name="sped",
    help="Universal speculative decoding for any model, any vocabulary.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=True,
)
console = Console()


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
):
    """sped - Universal Speculative Decoding Toolkit."""
    if version:
        from sped import __version__
        rprint(f"[bold green]sped[/bold green] v{__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        # Show help when no subcommand
        rprint(Panel.fit(
            "[bold cyan]sped[/bold cyan] — Universal Speculative Decoding\n\n"
            "Usage: sped <command> [options]\n\n"
            "Commands:\n"
            "  [bold]serve[/bold]       Run inference with speculative decoding\n"
            "  [bold]distil[/bold]      Distil a draft model via PEFT (LoRA)\n"
            "  [bold]experiment[/bold]  Run grid-search experiments and auto-tune\n"
            "  [bold]config[/bold]      Manage configuration (~/.sped/config.yml)\n"
            "  [bold]list[/bold]        List available models, adapters, and pairings\n"
            "  [bold]info[/bold]        Show system info and available hardware\n"
            "  [bold]version[/bold]     Show the sped version\n\n"
            "Run [bold]sped <command> --help[/bold] for detailed options.",
        ))


# ── Top-level commands ─────────────────────────────────────


@app.command()
def version():
    """Show the sped version."""
    from sped import __version__
    rprint(f"[bold green]sped[/bold green] v{__version__}")


@app.command()
def info(
    output: str = typer.Option("text", "--output", help="Output format: text or json"),
):
    """Show system info and available hardware."""
    import torch

    info_data = {
        "sped_version": __import__("sped").__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cpu_threads": torch.get_num_threads(),
    }

    if torch.cuda.is_available():
        info_data["gpu"] = torch.cuda.get_device_name(0)
        info_data["vram_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)

    if output == "json":
        import json as _json
        rprint(_json.dumps(info_data, indent=2))
        return

    rprint(Panel("[bold]sped System Info[/bold]", expand=False))
    rprint(f"  • sped:        [cyan]{info_data['sped_version']}[/cyan]")
    rprint(f"  • PyTorch:     [cyan]{info_data['torch_version']}[/cyan]")
    rprint(f"  • CUDA avail:  [cyan]{info_data['cuda_available']}[/cyan]")
    if torch.cuda.is_available():
        rprint(f"  • GPU:         [cyan]{info_data['gpu']}[/cyan]")
        rprint(f"  • VRAM:        [cyan]{info_data['vram_gb']} GB[/cyan]")
    rprint(f"  • MPS avail:   [cyan]{info_data['mps_available']}[/cyan]")
    rprint(f"  • CPU threads: [cyan]{info_data['cpu_threads']}[/cyan]")


@app.command()
def completion(shell: str = typer.Argument("bash", help="Shell type: bash, zsh, fish")):
    """Generate shell completion script."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "typer", "sped.cli.main", "utils", "completion", "--shell", shell],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        rprint(result.stdout)
    else:
        rprint(f"[red]Error generating completion: {result.stderr}[/red]")
        raise typer.Exit(code=1)


# ── Register sub-groups ────────────────────────────────────

from sped.cli.distil import app as distil_app
from sped.cli.serve import app as serve_app
from sped.cli.experiment import app as experiment_app
from sped.cli.list_cmd import app as list_app
from sped.cli.config_cmd import app as config_app

app.add_typer(serve_app, name="serve", help="Run inference with speculative decoding.")
app.add_typer(distil_app, name="distil", help="Distil a draft model via PEFT (LoRA).")
app.add_typer(experiment_app, name="experiment", help="Run grid-search experiments and auto-tune.")
app.add_typer(list_app, name="list", help="List available models, adapters, and pairings.")
app.add_typer(config_app, name="config", help="Manage configuration (YAML).")
