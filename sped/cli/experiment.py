"""sped experiment — automated grid-search experiments for speculative decoding."""

import typer
from pathlib import Path
from typing import Optional
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

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
        20, "--num-prompts", "-n",
        help="Number of prompts to use from file or defaults",
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
    strategies, then produces a comparison report.
    """
    import torch
    import json
    from datetime import datetime
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sped.core.speculative_decoding import SpeculativeDecoder
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
        f"[bold]Grid:[/bold]   {len(k_list)} K values × {len(temp_list)} temps × {len(align_list)} align = "
        f"[bold]{len(k_list) * len(temp_list) * len(align_list)}[/bold] experiments\n"
        f"[bold]Device:[/bold] {device}",
        title="⚡ sped experiment",
    ))

    # Load models once (warm-start)
    rprint(f"\n[bold]Loading models (warm-start)[/bold]...")

    target_tokenizer = AutoTokenizer.from_pretrained(target)
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    target_model = AutoModelForCausalLM.from_pretrained(
        target, torch_dtype="auto", device_map=device
    )
    target_model.eval()
    rprint(f"  ✓ Target: [green]{sum(p.numel() for p in target_model.parameters()) / 1e9:.1f}B[/green]")

    draft_tokenizer = AutoTokenizer.from_pretrained(draft)
    if draft_tokenizer.pad_token is None:
        draft_tokenizer.pad_token = draft_tokenizer.eos_token
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft, torch_dtype="auto", device_map=device
    )
    draft_model.eval()
    rprint(f"  ✓ Draft:  [green]{sum(p.numel() for p in draft_model.parameters()) / 1e9:.1f}B[/green]")

    compat, overlap = check_vocab_compatibility(draft_tokenizer, target_tokenizer)
    rprint(f"  ✓ Vocab overlap: {overlap:.1%}")

    # Load prompts
    if prompts_file:
        with open(prompts_file) as f:
            prompts = [json.loads(line) for line in f if line.strip()]
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
        ]

    prompts = prompts[:num_prompts]

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
        task = progress.add_task(
            f"Running experiments...", total=total_experiments
        )

        for k in k_list:
            for temp in temp_list:
                for align in align_list:
                    experiment_num += 1

                    # Create vocab aligner if needed
                    vocab_aligner = None
                    if align != "none":
                        from sped.vocab_agnostic.alignment import VocabAligner
                        vocab_aligner = VocabAligner(
                            target_tokenizer=target_tokenizer,
                            draft_tokenizer=draft_tokenizer,
                            strategy=align,
                        )

                    decoder = SpeculativeDecoder(
                        target_model=target_model,
                        target_tokenizer=target_tokenizer,
                        draft_model=draft_model,
                        draft_tokenizer=draft_tokenizer,
                        vocab_aligner=vocab_aligner,
                        max_draft_tokens=k,
                        device=device,
                    )

                    from time import time
                    prompt_results = []
                    for prompt in prompts:
                        start = time()
                        output = decoder.generate(
                            prompt=prompt,
                            max_new_tokens=max_tokens,
                            temperature=temp,
                            verbose=False,
                        )
                        elapsed = time() - start

                        response = output[len(prompt):] if output.startswith(prompt) else output
                        tokens = len(target_tokenizer.encode(response))

                        prompt_results.append({
                            "prompt": prompt[:80],
                            "tokens": tokens,
                            "time_seconds": round(elapsed, 2),
                            "tokens_per_second": round(tokens / elapsed, 1) if elapsed > 0 else 0,
                        })

                    # Aggregate
                    avg_tps = sum(r["tokens_per_second"] for r in prompt_results) / len(prompt_results)
                    avg_time = sum(r["time_seconds"] for r in prompt_results) / len(prompt_results)
                    total_tokens = sum(r["tokens"] for r in prompt_results)

                    result = {
                        "config": {
                            "draft_k": k,
                            "temperature": temp,
                            "align_strategy": align,
                        },
                        "avg_tokens_per_second": round(avg_tps, 1),
                        "avg_time_seconds": round(avg_time, 2),
                        "total_tokens": total_tokens,
                        "num_prompts": len(prompts),
                        "per_prompt": prompt_results,
                    }
                    results.append(result)

                    progress.update(task, advance=1, description=f"[{experiment_num}/{total_experiments}] K={k} T={temp} align={align} — {avg_tps:.1f} tok/s")

    # Generate report
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
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

    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    rprint(f"\n[green]✓[/green] Results saved: [cyan]{json_path}[/cyan]")

    # Summary table
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
    rprint(f"\n[bold green]🏆 Best config:[/bold green] K={best['config']['draft_k']}  T={best['config']['temperature']}  "
           f"align={best['config']['align_strategy']}  →  {best['avg_tokens_per_second']} tok/s")

    # HTML report (placeholder — will be enhanced in Issue #24)
    html_path = output_dir / "report.html"
    _generate_html_report(report, html_path)
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
        10, "--num-prompts", "-n",
        help="Number of prompts to evaluate each K on",
        min=3, max=100,
    ),
    device: str = typer.Option(
        "auto", "--device",
        help="Device: auto, cuda, cpu, mps",
    ),
):
    """Automatically find the optimal draft K for a model pair.

    Uses golden-section search to converge quickly without a full grid.
    """
    import torch
    from time import time
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

    # Load models (once)
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

    # Resolve alignment strategy
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

    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to compute fibonacci numbers.",
        "What are the main differences between TCP and UDP?",
        "Summarize the plot of Romeo and Juliet.",
        "What is machine learning?",
        "How does a transformer work?",
        "Write a haiku about programming.",
        "Explain the concept of recursion.",
        "What is the meaning of life?",
    ][:num_prompts]

    # Evaluate a given K
    def evaluate_k(k: int) -> float:
        decoder = SpeculativeDecoder(
            target_model=target_model,
            target_tokenizer=target_tokenizer,
            draft_model=draft_model,
            draft_tokenizer=draft_tokenizer,
            vocab_aligner=vocab_aligner,
            max_draft_tokens=k,
            device=device,
        )
        times = []
        for prompt in prompts:
            start = time()
            decoder.generate(prompt=prompt, max_new_tokens=128, temperature=temperature)
            times.append(time() - start)
        return len(prompts) * 128 / sum(times)  # tokens per second

    # Simple golden-section search
    a, b = min_k, max_k
    phi = (5 ** 0.5 - 1) / 2  # golden ratio (~0.618)
    tol = 1

    rprint(f"\n[bold]Searching...[/bold]")
    rprint(f"  {'K':>4}  {'tok/s':>8}")

    cache = {}

    while b - a > tol:
        x1 = int(a + (1 - phi) * (b - a))
        x2 = int(a + phi * (b - a))

        x1 = max(x1, a + 1)
        x2 = min(x2, b - 1)

        if x1 not in cache:
            cache[x1] = evaluate_k(x1)
            rprint(f"  {x1:>4}  {cache[x1]:>8.1f}")
        if x2 not in cache:
            cache[x2] = evaluate_k(x2)
            rprint(f"  {x2:>4}  {cache[x2]:>8.1f}")

        if cache[x1] > cache[x2]:
            b = x2
        else:
            a = x1

    # Final evaluation at all candidates in range
    best_k = max(range(a, b + 1), key=lambda k: cache.get(k, evaluate_k(k)))
    if best_k not in cache:
        cache[best_k] = evaluate_k(best_k)

    rprint(f"\n[bold green]🏆 Optimal K: {best_k}[/bold green] ({cache[best_k]:.1f} tok/s)")
    rprint(f"  [dim]Use: [bold]sped serve --draft-k {best_k}[/bold][/dim]")


def _generate_html_report(report: dict, path: Path):
    """Generate a minimal HTML report from experiment results."""
    rows = ""
    best = max(report["results"], key=lambda r: r["avg_tokens_per_second"])

    for r in sorted(report["results"], key=lambda x: x["avg_tokens_per_second"], reverse=True):
        highlight = "style='background:#d4edda;font-weight:bold'" if r == best else ""
        rows += f"""
        <tr {highlight}>
            <td>{r['config']['draft_k']}</td>
            <td>{r['config']['temperature']}</td>
            <td>{r['config']['align_strategy']}</td>
            <td>{r['avg_tokens_per_second']}</td>
            <td>{r['avg_time_seconds']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>sped Experiment Report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 2rem; background: #f8f9fa; }}
  h1 {{ color: #2d3436; }}
  table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #2d3436; color: white; }}
  tr:hover {{ background: #f1f3f5; }}
  .meta {{ color: #636e72; margin: 1rem 0; }}
  .best {{ color: #00b894; font-weight: bold; }}
</style>
</head>
<body>
<h1>⚡ sped Experiment Report</h1>
<div class="meta">
  <strong>Target:</strong> {report['metadata']['target_model']}<br>
  <strong>Draft:</strong> {report['metadata']['draft_model']}<br>
  <strong>Device:</strong> {report['metadata']['device']}<br>
  <strong>Prompts:</strong> {report['config']['num_prompts']} &middot;
  <strong>Max tokens:</strong> {report['config']['max_tokens']}<br>
  <strong>Date:</strong> {report['metadata']['timestamp']}
</div>

<h2>Results</h2>
<table>
  <thead>
    <tr><th>K</th><th>Temp</th><th>Align</th><th>Avg tok/s</th><th>Avg time (s)</th></tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>

<h2>Recommendations</h2>
<p>The best configuration is <strong>K={best['config']['draft_k']}</strong>
at <strong>T={best['config']['temperature']}</strong> with
<strong>{best['config']['align_strategy']}</strong> alignment,
achieving <strong>{best['avg_tokens_per_second']} tok/s</strong>.</p>
</body>
</html>"""
    path.write_text(html)
