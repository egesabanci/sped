"""sped list — enumerate available models, adapters, and draft-target pairings."""

import typer
from pathlib import Path
from typing import Optional
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns

app = typer.Typer(
    name="list",
    help="List available models, adapters, and recommended pairings.",
    no_args_is_help=True,
)


@app.callback()
def callback():
    pass


@app.command()
def models(
    query: Optional[str] = typer.Option(
        None, "--query", "-q",
        help="Search term to filter models (e.g. 'qwen', 'llama')",
    ),
    limit: int = typer.Option(
        15, "--limit", "-l",
        help="Maximum number of models to show",
        min=1, max=50,
    ),
):
    """List popular draft and target models from Hugging Face."""
    from huggingface_hub import HfApi

    api = HfApi()
    rprint(f"[bold]Searching Hugging Face for models...[/bold]\n")

    # Recommended draft models (small, < 3B params)
    draft_candidates = [
        "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B",
        "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B",
        "google/gemma-2-2B", "google/gemma-2-9B",
        "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-small-8k-instruct",
        "HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "apple/OpenELM-270M", "apple/OpenELM-1.1B",
    ]

    # Recommended target models (large, > 7B params)
    target_candidates = [
        "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B", "meta-llama/Llama-3.1-405B",
        "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-72B",
        "google/gemma-2-27B",
        "mistralai/Mistral-7B-v0.3", "mistralai/Mixtral-8x7B-v0.1",
        "microsoft/Phi-3-medium-4k-instruct",
    ]

    def get_model_info(model_id):
        try:
            info = api.model_info(model_id, timeout=5)
            likes = getattr(info, "likes", 0) or 0
            return (model_id, likes)
        except Exception:
            return (model_id, 0)

    # Filter
    all_candidates = draft_candidates + target_candidates
    if query:
        all_candidates = [m for m in all_candidates if query.lower() in m.lower()]

    # Show drafts table
    draft_list = [m for m in all_candidates if m in draft_candidates][:limit]
    if draft_list:
        table = Table(title="Recommended Draft Models (< 3B params)", box=None)
        table.add_column("Model", style="cyan")
        table.add_column("Size hint", style="dim")
        for m in draft_list:
            size = m.split("-")[-1] if "-" in m else ""
            table.add_row(m, size)
        rprint(table)
        rprint()

    # Show targets table
    target_list = [m for m in all_candidates if m in target_candidates][:limit]
    if target_list:
        table = Table(title="Recommended Target Models (> 7B params)", box=None)
        table.add_column("Model", style="cyan")
        table.add_column("Size hint", style="dim")
        for m in target_list:
            size = m.split("-")[-1] if "-" in m else ""
            table.add_row(m, size)
        rprint(table)

    rprint(f"\n[dim]Tip: Use [bold]sped list adapters[/bold] to see saved LoRA adapters[/dim]")


@app.command()
def adapters(
    path: Optional[Path] = typer.Option(
        None, "--path",
        help="Custom path to scan for adapters (default: ./draft-lora)",
    ),
):
    """List saved LoRA adapters from local distillation runs."""
    search_paths = [Path("./draft-lora"), Path("./adapters")]
    if path:
        search_paths = [path]

    table = Table(title="Local LoRA Adapters", box=None)
    table.add_column("Location", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Command")

    found = False
    for sp in search_paths:
        if sp.exists() and (sp / "adapter_config.json").exists():
            found = True
            table.add_row(
                str(sp.resolve()),
                "✓ ready",
                f"sped serve --draft-lora {sp.resolve()}",
            )

    if found:
        rprint(table)
    else:
        rprint("[yellow]No adapters found.[/yellow]")
        rprint("  Run [bold]sped distil[/bold] first to train a LoRA adapter.")
        rprint(f"  Scanned: {', '.join(str(p) for p in search_paths)}")


@app.command()
def pairings():
    """Show recommended draft-target model pairings."""
    table = Table(title="Recommended Draft-Target Pairings", box=None)
    table.add_column("Draft", style="cyan")
    table.add_column("Target", style="magenta")
    table.add_column("Vocab Match", justify="center")
    table.add_column("Notes", style="dim")

    pairings = [
        ("Qwen/Qwen2.5-0.5B",  "Qwen/Qwen2.5-72B",     "✅", "Same family, best acceptance"),
        ("meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.1-70B", "✅", "Same family, great acceptance"),
        ("meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-405B", "✅", "Largest speedup potential"),
        ("google/gemma-2-2B",  "google/gemma-2-27B",   "✅", "Same family"),
        ("Qwen/Qwen2.5-0.5B",  "meta-llama/Llama-3.1-70B", "⚠️", "Cross-vocab, use --align hybrid"),
        ("microsoft/Phi-3-mini", "meta-llama/Llama-3.1-70B", "⚠️", "Cross-vocab, use --align hybrid"),
        ("HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-72B", "⚠️", "Cross-vocab, use --align hybrid"),
        ("TinyLlama/TinyLlama-1.1B", "meta-llama/Llama-3.1-70B", "⚠️", "Cross-vocab, use --align hybrid"),
    ]

    for draft, target, match, notes in pairings:
        table.add_row(draft, target, match, notes)

    rprint(table)
    rprint("\n[dim]💡 Same-vocab pairs need no alignment layer and achieve higher acceptance.[/dim]")
    rprint("[dim]   Cross-vocab pairs use the Intel/Weizmann alignment algorithms.[/dim]")
