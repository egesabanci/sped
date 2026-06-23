"""Main CLI application."""

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
