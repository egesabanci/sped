"""Main CLI application — registers all commands and sub-groups."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

app = typer.Typer(
    name="sped",
    help="Universal speculative decoding for any model, any vocabulary.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def callback():
    """sped - Universal Speculative Decoding Toolkit."""
    pass


# ── Top-level commands ─────────────────────────────────────


@app.command()
def version():
    """Show the sped version."""
    from sped import __version__
    rprint(f"[bold green]sped[/bold green] v{__version__}")


@app.command()
def info():
    """Show system info and available hardware."""
    import torch

    rprint(Panel("[bold]sped System Info[/bold]", expand=False))
    rprint(f"  • PyTorch:     [cyan]{torch.__version__}[/cyan]")
    rprint(f"  • CUDA avail:  [cyan]{torch.cuda.is_available()}[/cyan]")
    if torch.cuda.is_available():
        rprint(f"  • GPU:         [cyan]{torch.cuda.get_device_name(0)}[/cyan]")
        rprint(f"  • VRAM:        [cyan]{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB[/cyan]")
    rprint(f"  • CPU threads: [cyan]{torch.get_num_threads()}[/cyan]")


# ── Register sub-groups ────────────────────────────────────

from sped.cli.distil import app as distil_app
from sped.cli.serve import app as serve_app
from sped.cli.experiment import app as experiment_app
from sped.cli.list_cmd import app as list_app

app.add_typer(distil_app, name="distil", help="Distil a draft model via PEFT (LoRA).")
app.add_typer(serve_app, name="serve", help="Run inference with speculative decoding.")
app.add_typer(experiment_app, name="experiment", help="Run grid-search experiments and auto-tune.")
app.add_typer(list_app, name="list", help="List available models, adapters, and pairings.")
