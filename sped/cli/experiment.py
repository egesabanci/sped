"""sped experiment — automated grid-search experiments for speculative decoding.

Refactored to separate CLI (argument parsing) from experiment logic.
The core engine can be tested without loading any models.
"""

import typer
from pathlib import Path
from typing import Optional
from time import time
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from ._experiment_engine import ExperimentEngine, AutoTuner

app = typer.Typer(
    name="experiment",
    help="Run automated experiments to find optimal SD hyperparameters.",
    no_args_is_help=True,
)


@app.callback()
def callback():
    pass


@app.command()
def run(
    target: str = typer.Option(
        ..., "--target", "-t",
        help="Target model ID or path",
    ),
    draft: str = typer.Option(
        ..., "--draft", "-d",
        help="Draft model ID or path",
    ),
    draft_k_values: str = typer.Option(
        "3,5,7,10", "--draft-k-values",
        help="Comma-separated list of K values to test",
    ),
    temperatures: str = typer.Option(
        "0.0,0.7", "--temperatures",
        help="Comma-separated list of temperatures to test",
    ),
    align_strategies: str = typer.Option(
        "none,hybrid", "--align-strategies",
        help="Comma-separated list of alignment strategies",
    ),
    prompts_file: Optional[Path] = typer.Option(
        None, "--prompts", "-p",
        help="JSONL file with prompts (one per line). Uses defaults if omitted.",
        exists=True,
    ),
    num_prompts: int = typer.Option(
        10, "--num-prompts", "-n",
        help="Number of prompts to use",
        min=1, max=500,
    ),
    max_tokens: int = typer.Option(
        128, "--max-tokens", "-m",
        help="Max tokens to generate per prompt",
        min=16, max=2048,
    ),
    output_dir: Path = typer.Option(
        "./experiment-results", "--output", "-o",
        help="Output directory for experiment results and report",
    ),
    device: str = typer.Option(
        "auto", "--device",
        help="Device: auto, cuda, cpu, mps",
    ),
):
    """Run grid-search experiments across speculative decoding hyperparameters.

    Tests all combinations of draft K values, temperatures, and alignment
    strategies, then produces a comparison report with HTML + JSON output.
    """
    import torch
    import json
    from datetime import datetime
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sped.utils.tokenizer_utils import check_vocab_compatibility

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse hyperparameter lists
    k_list = [int(k.strip()) for k in draft_k_values.split(",") if k.strip()]
    temp_list = [float(t.strip()) for t in temperatures.split(",") if t.strip()]
    align_list = [s.strip() for s in align_strategies.split(",") if s.strip()]

    rprint(Panel.fit(
        f"[bold]Target:[/bold] {target}\n"
        f"[bold]Draft:[/bold]  {draft}\n"
        f"[bold]Grid:[/bold]   {len(k_list)} K × {len(temp_list)} T × {len(align_list)} align = "
        f"[bold]{len(k_list) * len(temp_list) * len(align_list)}[/bold] experiments\n"
        f"[bold]Device:[/bold] {device}",
        title="⚡ sped experiment",
    ))

    # Load models once
    rprint(f"\n[bold]Loading models...[/bold]")
    target_tokenizer = AutoTokenizer.from_pretrained(target)
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    target_model = AutoModelForCausalLM.from_pretrained(
        target, torch_dtype="auto", device_map=device
    )
    target_model.eval()
    rprint(f"  ✓ Target: [green]{_param_count(target_model):.1f}B[/green]")

    draft_tokenizer = AutoTokenizer.from_pretrained(draft)
    if draft_tokenizer.pad_token is None:
        draft_tokenizer.pad_token = draft_tokenizer.eos_token
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft, torch_dtype="auto", device_map=device
    )
    draft_model.eval()
    rprint(f"  ✓ Draft:  [green]{_param_count(draft_model):.1f}B[/green]")

    compat, overlap = check_vocab_compatibility(draft_tokenizer, target_tokenizer)
    rprint(f"  ✓ Vocab overlap: {overlap:.1%}")

    # Load prompts
    if prompts_file:
        with open(prompts_file) as f:
            prompts = [json.loads(line)["prompt"] if isinstance(json.loads(line), dict) else line.strip()
                       for line in f if line.strip()][:num_prompts]
    else:
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a Python function to compute fibonacci numbers.",
            "What are the main differences between TCP and UDP?",
            "Summarize the plot of Romeo and Juliet.",
            "What is machine learning?",
            "How does a transformer work?",
            "Write a haiku about programming.",
            "What is the meaning of life?",
            "Explain the concept of recursion.",
        ][:num_prompts]

    # Create engine
    engine = ExperimentEngine(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        draft_model=draft_model,
        draft_tokenizer=draft_tokenizer,
        device=device,
    )

    # Run experiments
    total_experiments = len(k_list) * len(temp_list) * len(align_list)
    rprint(f"\n[bold]Running {total_experiments} experiments over {len(prompts)} prompts each...[/bold]\n")

    results = []
    experiment_num = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Running experiments...", total=total_experiments)

        for k in k_list:
            for temp in temp_list:
                for align in align_list:
                    experiment_num += 1

                    result = engine.run_single_experiment(
                        draft_k=k,
                        temperature=temp,
                        align_strategy=align,
                        prompts=prompts,
                        max_tokens=max_tokens,
                    )
                    results.append(result)

                    progress.update(
                        task, advance=1,
                        description=f"[{experiment_num}/{total_experiments}] "
                        f"K={k} T={temp} align={align} — {result['avg_tokens_per_second']:.1f} tok/s",
                    )

    # Clean up model references explicitly to prevent memory leaks
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Generate report
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "metadata": {
            "target_model": target,
            "draft_model": draft,
            "timestamp": datetime.now().isoformat(),
            "device": device,
        },
        "config": {
            "draft_k_values": k_list,
            "temperatures": temp_list,
            "align_strategies": align_list,
            "num_prompts": num_prompts,
            "max_tokens": max_tokens,
        },
        "results": results,
    }

    # JSON export
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    rprint(f"\n[green]✓[/green] Results saved: [cyan]{json_path}[/cyan]")

    # Summary table
    _print_summary_table(results)

    # HTML report (#24)
    from ._experiment_engine import generate_html_report
    html_path = output_dir / "report.html"
    generate_html_report(report, html_path)
    rprint(f"[green]✓[/green] HTML report: [cyan]{html_path}[/cyan]")


@app.command()
def auto_tune(
    target: str = typer.Option(
        ..., "--target", "-t",
        help="Target model ID or path",
    ),
    draft: str = typer.Option(
        ..., "--draft", "-d",
        help="Draft model ID or path",
    ),
    min_k: int = typer.Option(
        2, "--min-k",
        help="Minimum draft K to evaluate",
        min=1, max=5,
    ),
    max_k: int = typer.Option(
        15, "--max-k",
        help="Maximum draft K to evaluate",
        min=5, max=30,
    ),
    temperature: float = typer.Option(
        0.0, "--temperature", "-T",
        help="Sampling temperature",
    ),
    align: str = typer.Option(
        "auto", "--align",
        help="Alignment strategy: auto, none, string, probabilistic, hybrid",
    ),
    num_prompts: int = typer.Option(
        5, "--num-prompts", "-n",
        help="Number of prompts to evaluate each K on",
        min=2, max=100,
    ),
    device: str = typer.Option(
        "auto", "--device",
        help="Device: auto, cuda, cpu, mps",
    ),
):
    """Automatically find the optimal draft K for a model pair.

    Uses golden-section search to converge quickly without a full grid.
    Only loads models once at startup.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sped.core.speculative_decoding import SpeculativeDecoder
    from sped.utils.tokenizer_utils import check_vocab_compatibility

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rprint(Panel.fit(
        f"[bold]Auto-tune draft K[/bold] — {draft} → {target}\n"
        f"  Searching [{min_k}..{max_k}]  align={align}  temp={temperature}",
        title="⚡ sped experiment auto-tune",
    ))

    # Load models once
    target_tokenizer = AutoTokenizer.from_pretrained(target)
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    target_model = AutoModelForCausalLM.from_pretrained(
        target, torch_dtype="auto", device_map=device
    )
    target_model.eval()

    draft_tokenizer = AutoTokenizer.from_pretrained(draft)
    if draft_tokenizer.pad_token is None:
        draft_tokenizer.pad_token = draft_tokenizer.eos_token
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft, torch_dtype="auto", device_map=device
    )
    draft_model.eval()

    # Resolve alignment
    if align == "auto":
        compat, _ = check_vocab_compatibility(draft_tokenizer, target_tokenizer)
        align = "none" if compat else "hybrid"

    vocab_aligner = None
    if align != "none":
        from sped.vocab_agnostic.alignment import VocabAligner
        vocab_aligner = VocabAligner(
            target_tokenizer=target_tokenizer,
            draft_tokenizer=draft_tokenizer,
            strategy=align,
        )

    # Create decoder once for all evaluations
    decoder = SpeculativeDecoder(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        draft_model=draft_model,
        draft_tokenizer=draft_tokenizer,
        vocab_aligner=vocab_aligner,
        max_draft_tokens=min_k,
        device=device,
    )

    # Use the same prompts for all evaluations (cached after first generation)
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to compute fibonacci numbers.",
    ][:num_prompts]

    tuner = AutoTuner(decoder=decoder, prompts=prompts, max_tokens=64)

    # Golden-section search
    best_k = tuner.search(min_k=min_k, max_k=max_k)

    # Cleanup
    del decoder, tuner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rprint(f"\n[bold green]🏆 Optimal K: {best_k}[/bold green]")
    rprint(f"  [dim]Use: [bold]sped serve --draft-k {best_k}[/bold][/dim]")


# ── Helpers ──────────────────────────────────────────────


def _param_count(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e9


def _print_summary_table(results: list[dict]):
    """Print a sorted results table with the best config highlighted."""
    table = Table(title="Experiment Results Summary")
    table.add_column("K", justify="right", style="cyan")
    table.add_column("Temp", justify="right")
    table.add_column("Align", style="magenta")
    table.add_column("Avg tok/s", justify="right")
    table.add_column("Avg time (s)", justify="right")

    best = max(results, key=lambda r: r["avg_tokens_per_second"])
    for r in sorted(results, key=lambda x: x["avg_tokens_per_second"], reverse=True):
        style = "bold green" if r == best else ""
        table.add_row(
            str(r["config"]["draft_k"]),
            str(r["config"]["temperature"]),
            str(r["config"]["align_strategy"]),
            str(r["avg_tokens_per_second"]),
            str(r["avg_time_seconds"]),
            style=style,
        )

    rprint(table)
    rprint(
        f"\n[bold green]🏆 Best:[/bold green] K={best['config']['draft_k']}  "
        f"T={best['config']['temperature']}  align={best['config']['align_strategy']}  "
        f"→  {best['avg_tokens_per_second']} tok/s"
    )
