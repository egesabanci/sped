"""sped serve — interactive inference with speculative decoding.

Supports multiple backends (HF Transformers, MLX, vLLM) and modes:
- Interactive REPL with streaming output and speculation stats
- Single-prompt mode for quick generation
- Benchmark mode for automated speedup measurement
"""

import typer
from pathlib import Path
from typing import Optional
from time import time
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text
from rich.layout import Layout
from rich.live import Live

from sped.serving import BackendConfig
from sped.serving.hf_backend import HFBackend

app = typer.Typer(
    name="serve",
    help="Run inference with speculative decoding.",
    no_args_is_help=True,
)

console = Console()


@app.callback()
def callback():
    pass


@app.command()
def run(
    target: str = typer.Option(
        ..., "--target", "-t",
        help="Target model ID or path",
    ),
    draft: Optional[str] = typer.Option(
        None, "--draft", "-d",
        help="Draft model ID or path. If omitted, runs standard autoregressive.",
    ),
    draft_lora: Optional[Path] = typer.Option(
        None, "--draft-lora",
        help="Path to pre-trained LoRA adapter for the draft model",
        exists=True, file_okay=False, dir_okay=True,
    ),
    backend: str = typer.Option(
        "auto", "--backend", "-b",
        help="Inference backend: auto, hf, mlx, vllm, sglang",
    ),
    align: str = typer.Option(
        "auto", "--align",
        help="Vocabulary alignment strategy: auto, none, string, probabilistic, hybrid",
    ),
    draft_k: int = typer.Option(
        5, "--draft-k", "-k",
        help="Number of draft tokens per speculation step",
        min=1, max=20,
    ),
    temperature: float = typer.Option(
        0.0, "--temperature", "-T",
        help="Sampling temperature (0 = greedy)",
        min=0.0, max=5.0,
    ),
    max_new_tokens: int = typer.Option(
        512, "--max-new-tokens", "-n",
        help="Maximum tokens to generate per response",
        min=1, max=16384,
    ),
    device: str = typer.Option(
        "auto", "--device",
        help="Device: auto, cuda, cpu, mps",
    ),
    prompt: Optional[str] = typer.Option(
        None, "--prompt", "-p",
        help="Single prompt mode. Provide a prompt and exit.",
    ),
    benchmark: bool = typer.Option(
        False, "--benchmark",
        help="Run benchmark mode on standard prompts",
    ),
    quantization: Optional[str] = typer.Option(
        None, "--quantization", "-q",
        help="Quantization: 4bit, 8bit (HF backend only)",
    ),
):
    """Run inference with speculative decoding.

    Use --backend mlx for Apple Silicon optimized inference.
    Use --backend hf for standard PyTorch/Hugging Face inference.
    """
    import torch

    # Resolve backend
    resolved_backend = _resolve_backend(backend)
    rprint(Panel.fit(
        f"[bold]Target:[/bold] {target}\n"
        f"[bold]Draft:[/bold]  {draft or 'none (standard mode)'}\n"
        f"[bold]Backend:[/bold] {resolved_backend.upper()}  [bold]Device:[/bold] {device}\n"
        f"[bold]Draft K:[/bold] {draft_k}  [bold]Temp:[/bold] {temperature}",
        title="⚡ sped serve",
    ))

    # Load target model
    rprint(f"\n[bold]Loading target model[/bold]: [cyan]{target}[/cyan]")
    target_backend = _create_backend(resolved_backend)

    target_cfg = BackendConfig(
        model_id=target,
        device=device,
        quantization=quantization,
    )
    target_backend.load_model(target_cfg)
    target_model = target_backend.model
    target_tokenizer = target_backend.tokenizer
    rprint(f"  ✓ [green]Loaded[/green] via {resolved_backend} backend")

    # Load draft model
    draft_model = None
    draft_tokenizer = None
    vocab_aligner = None

    if draft is not None:
        rprint(f"[bold]Loading draft model[/bold]: [cyan]{draft}[/cyan]")
        draft_backend = _create_backend(resolved_backend)
        draft_cfg = BackendConfig(
            model_id=draft if draft_lora is None else str(draft_lora),
            device=device,
            quantization=quantization,
        )
        draft_backend.load_model(draft_cfg)

        draft_model = draft_backend.model
        draft_tokenizer = draft_backend.tokenizer

        # Load LoRA adapter if specified
        if draft_lora is not None:
            try:
                from peft import PeftModel
                draft_model = PeftModel.from_pretrained(draft_model, str(draft_lora))
                rprint(f"  ✓ [green]LoRA adapter loaded[/green] from {draft_lora}")
            except Exception as e:
                rprint(f"  [yellow]⚠ LoRA load failed: {e}[/yellow]")

        rprint(f"  ✓ [green]Draft loaded[/green]")

        # Vocab compatibility check
        from sped.utils.tokenizer_utils import check_vocab_compatibility
        compat, overlap = check_vocab_compatibility(draft_tokenizer, target_tokenizer)
        if compat:
            rprint(f"  ✓ Vocab match ({overlap:.1%})")
            align = "none"
        else:
            rprint(f"  ⚠ Vocabs differ ({overlap:.1%}) — using [bold]{align}[/bold]")
            if align == "auto":
                align = "hybrid"

        if align != "none":
            from sped.vocab_agnostic.alignment import VocabAligner
            vocab_aligner = VocabAligner(
                target_tokenizer=target_tokenizer,
                draft_tokenizer=draft_tokenizer,
                strategy=align,
                target_model=target_model,
            )

    # Create speculative decoder
    from sped.core.speculative_decoding import SpeculativeDecoder

    decoder = SpeculativeDecoder(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        draft_model=draft_model,
        draft_tokenizer=draft_tokenizer,
        vocab_aligner=vocab_aligner,
        max_draft_tokens=draft_k,
        device=device,
    )

    # Run selected mode
    if benchmark:
        _run_benchmark(decoder, target_tokenizer, draft is not None)
    elif prompt is not None:
        _run_single(decoder, prompt, max_new_tokens, temperature)
    else:
        _run_repl(decoder, max_new_tokens, temperature)


# ── Backend resolution ───────────────────────────────────


def _resolve_backend(backend: str) -> str:
    """Resolve 'auto' to the best available backend."""
    if backend != "auto":
        return backend

    # Auto-detect: prefer MLX on Apple Silicon, HF everywhere else
    try:
        from sped.serving.mlx_backend import MLXBackend
        if MLXBackend.is_available():
            return "mlx"
    except ImportError:
        pass

    return "hf"


def _create_backend(backend: str):
    """Create an inference backend instance."""
    if backend == "mlx":
        from sped.serving.mlx_backend import MLXBackend
        return MLXBackend()
    elif backend == "vllm":
        # vLLM backend — attempt import, raise if not installed
        try:
            from sped.serving.vllm_backend import VLLMBackend
            return VLLMBackend()
        except ImportError:
            rprint("[yellow]vLLM not installed, falling back to HF[/yellow]")
            return HFBackend()
    else:
        return HFBackend()


# ── Generation helpers ───────────────────────────────────


def _run_single(decoder, prompt: str, max_new_tokens: int, temperature: float):
    """Single-prompt generation."""
    rprint(f"\n[bold]Prompt:[/bold] {prompt}\n")
    rprint("[bold]Response:[/bold]")

    start = time()
    output = decoder.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    elapsed = time() - start

    response = output[len(prompt):] if output.startswith(prompt) else output
    rprint(f"{response}\n")

    tokens = len(decoder.target_tokenizer.encode(response))
    metrics = decoder.get_metrics()

    stats = Table.grid(padding=(0, 2))
    stats.add_column()
    stats.add_column()
    stats.add_row("[dim]Tokens[/dim]", str(tokens))
    stats.add_row("[dim]Time[/dim]", f"{elapsed:.1f}s")
    stats.add_row("[dim]Throughput[/dim]", f"{tokens / max(elapsed, 0.01):.1f} tok/s")

    if metrics.get("speedup_vs_vanilla"):
        stats.add_row("[dim]Speedup[/dim]", f"[green]{metrics['speedup_vs_vanilla']}x[/green]")
    if metrics.get("acceptance_rate", 0) > 0:
        stats.add_row("[dim]Accept rate[/dim]", f"{metrics['acceptance_rate']:.1%}")

    rprint(stats)


def _run_repl(decoder, max_new_tokens: int, temperature: float):
    """Interactive REPL with streaming."""
    from time import time

    rprint(f"\n[bold green]Interactive mode[/bold green] — type /help for commands\n")

    while True:
        try:
            prompt = Prompt.ask("[bold]»[/bold]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Bye![/yellow]")
            break

        if not prompt:
            continue

        if prompt.startswith("/"):
            _handle_command(prompt[1:], decoder)
            continue

        start = time()
        output = decoder.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        elapsed = time() - start

        response = output[len(prompt):] if output.startswith(prompt) else output
        console.print(f"{response}")
        tokens = len(decoder.target_tokenizer.encode(response))
        metrics = decoder.get_metrics()

        stat_line = (
            f"[dim]─ {tokens} tokens in {elapsed:.1f}s "
            f"({tokens / max(elapsed, 0.01):.1f} tok/s)"
        )
        if metrics.get("acceptance_rate", 0) > 0:
            stat_line += f" | accept rate: {metrics['acceptance_rate']:.1%}"
        if metrics.get("speedup_vs_vanilla"):
            stat_line += f" | [green]{metrics['speedup_vs_vanilla']}x speedup[/green]"

        console.print(f"{stat_line}[/dim]\n")
        decoder.reset_metrics()


def _handle_command(cmd: str, decoder):
    """Handle REPL slash commands."""
    parts = cmd.strip().split()
    command = parts[0].lower()

    if command in ("exit", "quit", "q"):
        console.print("[yellow]Bye![/yellow]")
        raise SystemExit(0)

    elif command in ("help", "h", "?"):
        console.print(Panel.fit(
            "[bold]/help[/bold]   — show this help\n"
            "[bold]/stats[/bold]  — show cumulative generation stats\n"
            "[bold]/exit[/bold]   — quit\n"
            "[bold]/clear[/bold]  — clear screen",
            title="Commands",
        ))

    elif command == "stats":
        metrics = decoder.get_metrics()
        if metrics["total_steps"] == 0:
            console.print("[yellow]No generations yet.[/yellow]")
            return

        table = Table(title="Generation Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total steps", str(metrics["total_steps"]))
        table.add_row("Tokens generated", str(metrics["total_tokens_generated"]))
        table.add_row("Acceptance rate", f"{metrics['acceptance_rate']:.1%}")
        table.add_row("Avg tokens/step", f"{metrics['avg_tokens_per_step']:.2f}")
        table.add_row("Avg tok/s", f"{metrics['avg_tokens_per_second']:.1f}")
        if metrics.get("speedup_vs_vanilla"):
            table.add_row("Speedup vs vanilla", f"[green]{metrics['speedup_vs_vanilla']}x[/green]")
        table.add_row("Total time", f"{metrics['total_time_seconds']:.1f}s")

        if "time_breakdown" in metrics:
            tb = metrics["time_breakdown"]
            table.add_row("Draft time", f"{tb.get('draft_pct', 0)}%")
            table.add_row("Verify time", f"{tb.get('verify_pct', 0)}%")
            table.add_row("Sampling time", f"{tb.get('sampling_pct', 0)}%")

        console.print(table)

    elif command == "clear":
        import os
        os.system("cls" if os.name == "nt" else "clear")

    else:
        console.print(f"[red]Unknown command:[/red] /{command}. Type /help.")


# ── Benchmark mode (#21) ─────────────────────────────────


def _run_benchmark(decoder, tokenizer, has_draft: bool):
    """Run benchmark comparing speculative vs standard generation."""
    from time import time
    import json
    from datetime import datetime

    benchmarks = [
        "What is the capital of France?",
        "Explain quantum computing in three sentences.",
        "Write a Python function to compute fibonacci numbers.",
        "What are the main differences between TCP and UDP?",
        "Summarize the theory of relativity.",
    ]

    rprint(f"\n[bold]Running benchmark on {len(benchmarks)} prompts...[/bold]\n")

    results = []
    total_spec_time = 0
    total_standard_time = 0
    total_tokens = 0

    for i, prompt in enumerate(benchmarks):
        rprint(f"  [{i+1}/{len(benchmarks)}] {prompt[:60]}...")

        # Measure speculative generation
        decoder.reset_metrics()
        start = time()
        spec_output = decoder.generate(
            prompt=prompt, max_new_tokens=128, temperature=0.0,
        )
        spec_elapsed = time() - start
        spec_response = spec_output[len(prompt):] if spec_output.startswith(prompt) else spec_output
        spec_tokens = len(tokenizer.encode(spec_response))
        metrics = decoder.get_metrics()

        # Measure standard generation (no draft)
        if has_draft:
            from sped.core.speculative_decoding import SpeculativeDecoder
            std_decoder = SpeculativeDecoder(
                target_model=decoder.target_model,
                target_tokenizer=decoder.target_tokenizer,
                max_draft_tokens=decoder.max_draft_tokens,
                device=decoder.device,
            )
            start = time()
            std_output = std_decoder.generate(
                prompt=prompt, max_new_tokens=128, temperature=0.0,
            )
            std_elapsed = time() - start
        else:
            std_elapsed = spec_elapsed  # no draft = same

        speedup = round(std_elapsed / max(spec_elapsed, 0.001), 2) if has_draft else 1.0

        results.append({
            "prompt": prompt,
            "spec_tokens": spec_tokens,
            "spec_time": round(spec_elapsed, 3),
            "spec_tok_s": round(spec_tokens / max(spec_elapsed, 0.001), 1),
            "std_time": round(std_elapsed, 3),
            "speedup": speedup,
            "acceptance_rate": round(metrics.get("acceptance_rate", 0), 3),
            "avg_tokens_per_step": round(metrics.get("avg_tokens_per_step", 0), 2),
        })
        total_spec_time += spec_elapsed
        total_standard_time += std_elapsed
        total_tokens += spec_tokens

    # Summary table
    table = Table(title="Benchmark Results", header_style="bold")
    table.add_column("Prompt", style="cyan", no_wrap=False)
    table.add_column("Tokens", justify="right")
    table.add_column("Spec (s)", justify="right")
    table.add_column("Std (s)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Accept", justify="right")

    for r in results:
        speedup_style = "green" if r["speedup"] > 1.5 else "yellow" if r["speedup"] > 1.0 else "red"
        table.add_row(
            r["prompt"][:40],
            str(r["spec_tokens"]),
            str(r["spec_time"]),
            str(r["std_time"]),
            f"[{speedup_style}]{r['speedup']}x[/{speedup_style}]",
            f"{r['acceptance_rate']:.0%}" if r["acceptance_rate"] > 0 else "—",
        )

    avg_speedup = total_standard_time / max(total_spec_time, 0.001) if has_draft else 1.0
    table.add_row(
        "[bold]Total/Avg[/bold]",
        str(total_tokens),
        f"{total_spec_time:.1f}",
        f"{total_standard_time:.1f}",
        f"[bold green]{avg_speedup:.2f}x[/bold green]",
        "",
        style="bold",
    )
    rprint(table)

    # JSON export
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "has_draft": has_draft,
            "draft_k": decoder.max_draft_tokens,
        },
        "summary": {
            "total_tokens": total_tokens,
            "total_spec_time": round(total_spec_time, 3),
            "total_standard_time": round(total_standard_time, 3),
            "avg_speedup": round(avg_speedup, 3),
        },
        "per_prompt": results,
    }

    json_path = Path("benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    rprint(f"\n[dim]Results saved to: {json_path}[/dim]")
