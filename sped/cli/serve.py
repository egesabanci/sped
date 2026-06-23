"""sped serve — interactive inference with speculative decoding.

Supports multiple backends (HF Transformers, MLX, vLLM) and modes:
- Interactive REPL with streaming output and speculation stats
- Single-prompt mode for quick generation
- Benchmark mode for automated speedup measurement (memory-safe)
"""

import gc
import typer
from pathlib import Path
from typing import Optional
from time import time
import json
from datetime import datetime

from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.prompt import Prompt

from sped.serving import BackendConfig
from sped.serving.hf_backend import HFBackend

app = typer.Typer(name="serve", help="Run inference with speculative decoding.", no_args_is_help=True)
console = Console()


@app.callback()
def callback():
    pass


@app.command()
def run(
    target: str = typer.Option(..., "--target", "-t", help="Target model ID or path"),
    draft: Optional[str] = typer.Option(None, "--draft", "-d", help="Draft model ID or path. Omit for standard autoregressive."),
    draft_lora: Optional[Path] = typer.Option(None, "--draft-lora", help="Path to LoRA adapter", exists=True, file_okay=False, dir_okay=True),
    backend: str = typer.Option("auto", "--backend", "-b", help="Backend: auto, hf, mlx, vllm"),
    align: str = typer.Option("auto", "--align", help="Alignment: auto, none, string, probabilistic, hybrid"),
    draft_k: int = typer.Option(5, "--draft-k", "-k", help="Draft tokens per step", min=1, max=20),
    temperature: float = typer.Option(0.0, "--temperature", "-T", help="Sampling temperature (0=greedy)", min=0.0, max=5.0),
    max_new_tokens: int = typer.Option(512, "--max-new-tokens", "-n", help="Max tokens per response", min=1, max=16384),
    device: str = typer.Option("auto", "--device", help="Device: auto, cuda, cpu, mps"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Single prompt mode"),
    benchmark: bool = typer.Option(False, "--benchmark", help="Run benchmark mode"),
    quantization: Optional[str] = typer.Option(None, "--quantization", "-q", help="Quantization: 4bit, 8bit (HF only)"),
):
    """Run inference with speculative decoding."""
    import torch

    # Resolve backend — MLX can't be used with speculative decoding
    resolved_backend = _resolve_backend(backend, has_draft=(draft is not None))
    if draft is not None and resolved_backend == "mlx":
        rprint("  [yellow]MLX backend doesn't support speculative decoding. Forcing HF backend.[/yellow]")
        resolved_backend = "hf"

    rprint(Panel.fit(
        f"[bold]Target:[/bold] {target}\n"
        f"[bold]Draft:[/bold]  {draft or 'none (standard mode)'}\n"
        f"[bold]Backend:[/bold] {resolved_backend.upper()}  [bold]Device:[/bold] {device}\n"
        f"[bold]Draft K:[/bold] {draft_k}  [bold]Temp:[/bold] {temperature}",
        title="sped serve",
    ))

    # Load target
    rprint(f"\n[bold]Loading target model[/bold]: [cyan]{target}[/cyan]")
    target_backend = _create_backend(resolved_backend)
    target_backend.load_model(BackendConfig(model_id=target, device=device, quantization=quantization))
    target_model = target_backend.model
    target_tokenizer = target_backend.tokenizer
    rprint(f"  \u2713 Loaded via {resolved_backend} backend")

    # Load draft
    draft_model = draft_tokenizer = vocab_aligner = None
    if draft is not None:
        rprint(f"[bold]Loading draft model[/bold]: [cyan]{draft}[/cyan]")
        draft_backend = _create_backend(resolved_backend)
        draft_backend.load_model(BackendConfig(model_id=draft if draft_lora is None else str(draft_lora), device=device, quantization=quantization))
        draft_model = draft_backend.model
        draft_tokenizer = draft_backend.tokenizer

        if draft_lora is not None:
            try:
                from peft import PeftModel
                draft_model = PeftModel.from_pretrained(draft_model, str(draft_lora))
                rprint(f"  \u2713 LoRA loaded from {draft_lora}")
            except Exception as e:
                rprint(f"  [yellow]Warning: LoRA load failed: {e}[/yellow]")

        rprint("  \u2713 Draft loaded")

        from sped.utils.tokenizer_utils import check_vocab_compatibility
        compat, overlap = check_vocab_compatibility(draft_tokenizer, target_tokenizer)
        if compat:
            rprint(f"  \u2713 Vocab match ({overlap:.1%})")
            align = "none"
        else:
            rprint(f"  \u26a0 Vocabs differ ({overlap:.1%}) \u2014 using {align}")
            if align == "auto":
                align = "hybrid"

        if align != "none":
            from sped.vocab_agnostic.alignment import VocabAligner
            vocab_aligner = VocabAligner(target_tokenizer=target_tokenizer, draft_tokenizer=draft_tokenizer, strategy=align, target_model=target_model)

    # Create decoder
    from sped.core.speculative_decoding import SpeculativeDecoder
    decoder = SpeculativeDecoder(
        target_model=target_model, target_tokenizer=target_tokenizer,
        draft_model=draft_model, draft_tokenizer=draft_tokenizer,
        vocab_aligner=vocab_aligner, max_draft_tokens=draft_k, device=device,
    )

    if benchmark:
        _run_benchmark(decoder, target_tokenizer, draft is not None, max_new_tokens=min(max_new_tokens, 32))
    elif prompt is not None:
        _run_single(decoder, prompt, max_new_tokens, temperature)
    else:
        _run_repl(decoder, max_new_tokens, temperature)


# ── Backend resolution ───────────────────────────────────


def _resolve_backend(backend: str, has_draft: bool = False) -> str:
    """Resolve 'auto' to best available backend.

    When speculative decoding is used (has_draft=True), never auto-select
    MLX \u2014 the SpeculativeDecoder requires HF model/tokenizer interfaces.
    """
    if backend != "auto":
        return backend
    if has_draft:
        return "hf"
    try:
        from sped.serving.mlx_backend import MLXBackend
        if MLXBackend.is_available():
            return "mlx"
    except ImportError:
        pass
    return "hf"


def _create_backend(backend: str):
    if backend == "mlx":
        from sped.serving.mlx_backend import MLXBackend
        return MLXBackend()
    elif backend == "vllm":
        try:
            from sped.serving.vllm_backend import VLLMBackend
            return VLLMBackend()
        except ImportError:
            rprint("[yellow]vLLM not installed, falling back to HF[/yellow]")
            return HFBackend()
    return HFBackend()


# ── Generation helpers ───────────────────────────────────


def _run_single(decoder, prompt: str, max_new_tokens: int, temperature: float):
    rprint(f"\n[bold]Prompt:[/bold] {prompt}\n[bold]Response:[/bold]")
    start = time()
    output = decoder.generate(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    elapsed = time() - start
    response = output[len(prompt):] if output.startswith(prompt) else output
    rprint(f"{response}\n")

    tokens = len(decoder.target_tokenizer.encode(response))
    metrics = decoder.get_metrics()
    stats = Table.grid(padding=(0, 2))
    stats.add_column(); stats.add_column()
    stats.add_row("Tokens", str(tokens))
    stats.add_row("Time", f"{elapsed:.1f}s")
    stats.add_row("Throughput", f"{tokens / max(elapsed, 0.01):.1f} tok/s")
    if metrics.get("speedup_vs_vanilla"):
        stats.add_row("Speedup", f"[green]{metrics['speedup_vs_vanilla']}x[/green]")
    if metrics.get("acceptance_rate", 0) > 0:
        stats.add_row("Accept rate", f"{metrics['acceptance_rate']:.1%}")
    rprint(stats)


def _run_repl(decoder, max_new_tokens: int, temperature: float):
    rprint(f"\n[bold green]Interactive mode[/bold green] \u2014 type /help for commands\n")
    while True:
        try:
            prompt_str = Prompt.ask("[bold]\u00bb[/bold]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Bye![/yellow]")
            break
        if not prompt_str:
            continue
        if prompt_str.startswith("/"):
            _handle_command(prompt_str[1:], decoder)
            continue

        start = time()
        output = decoder.generate(prompt=prompt_str, max_new_tokens=max_new_tokens, temperature=temperature)
        elapsed = time() - start
        response = output[len(prompt_str):] if output.startswith(prompt_str) else output
        console.print(f"{response}")
        tokens = len(decoder.target_tokenizer.encode(response))
        metrics = decoder.get_metrics()
        line = f"[dim]\u2014 {tokens} tokens in {elapsed:.1f}s ({tokens / max(elapsed, 0.01):.1f} tok/s)"
        if metrics.get("acceptance_rate", 0) > 0:
            line += f" | accept rate: {metrics['acceptance_rate']:.1%}"
        if metrics.get("speedup_vs_vanilla"):
            line += f" | [green]{metrics['speedup_vs_vanilla']}x speedup[/green]"
        console.print(f"{line}[/dim]\n")
        decoder.reset_metrics()


def _handle_command(cmd: str, decoder):
    parts = cmd.strip().split()
    command = parts[0].lower()
    if command in ("exit", "quit", "q"):
        console.print("[yellow]Bye![/yellow]")
        raise SystemExit(0)
    elif command in ("help", "h", "?"):
        console.print(Panel.fit("/help   \u2014 show this help\n/stats  \u2014 show cumulative stats\n/exit   \u2014 quit\n/clear  \u2014 clear screen", title="Commands"))
    elif command == "stats":
        metrics = decoder.get_metrics()
        if metrics["total_steps"] == 0:
            console.print("[yellow]No generations yet.[/yellow]")
            return
        table = Table(title="Generation Statistics")
        table.add_column("Metric", style="cyan"); table.add_column("Value", justify="right")
        table.add_row("Total steps", str(metrics["total_steps"]))
        table.add_row("Tokens generated", str(metrics["total_tokens_generated"]))
        table.add_row("Acceptance rate", f"{metrics['acceptance_rate']:.1%}")
        table.add_row("Avg tokens/step", f"{metrics['avg_tokens_per_step']:.2f}")
        table.add_row("Avg tok/s", f"{metrics['avg_tokens_per_second']:.1f}")
        if metrics.get("speedup_vs_vanilla"):
            table.add_row("Speedup vs vanilla", f"[green]{metrics['speedup_vs_vanilla']}x[/green]")
        table.add_row("Total time", f"{metrics['total_time_seconds']:.1f}s")
        console.print(table)
    elif command == "clear":
        import os
        os.system("cls" if os.name == "nt" else "clear")
    else:
        console.print(f"[red]Unknown:[/red] /{command}. Type /help.")


# ── Benchmark mode (memory-safe) ─────────────────────────


def _run_benchmark(decoder, tokenizer, has_draft: bool, max_new_tokens: int = 16):
    """Run benchmark comparing speculative vs standard generation.

    Memory-safe: short generations (default 16 tokens), gc between prompts,
    no duplicate decoder allocations, capped at 32 tokens max.
    """
    benchmarks = [
        "What is the capital of France?",
        "Explain quantum computing in three sentences.",
        "Write a Python function to compute fibonacci numbers.",
        "What are the main differences between TCP and UDP?",
        "Summarize the theory of relativity.",
    ]
    max_new_tokens = min(max_new_tokens, 32)  # hard cap
    rprint(f"\n[bold]Running benchmark on {len(benchmarks)} prompts...[/bold]")
    rprint(f"  (max {max_new_tokens} tokens each, capped for safety)\n")

    results = []
    total_spec_time = 0.0
    total_tokens = 0

    for i, prompt_text in enumerate(benchmarks):
        rprint(f"  [{i+1}/{len(benchmarks)}] {prompt_text[:60]}...")

        # Speculative generation
        decoder.reset_metrics()
        start = time()
        try:
            spec_output = decoder.generate(prompt=prompt_text, max_new_tokens=max_new_tokens, temperature=0.0)
        except Exception as e:
            rprint(f"  [red]Speculation failed: {e}[/red]")
            spec_output = prompt_text
        spec_elapsed = time() - start
        spec_response = spec_output[len(prompt_text):] if spec_output.startswith(prompt_text) else spec_output
        spec_tokens = len(tokenizer.encode(spec_response) if hasattr(tokenizer, 'encode') else spec_response.split())
        metrics = decoder.get_metrics()

        # Standard generation via a new decoder (no draft)
        if has_draft:
            std_decoder = decoder.__class__(
                target_model=decoder.target_model, target_tokenizer=decoder.target_tokenizer,
                max_draft_tokens=decoder.max_draft_tokens, device=decoder.device,
            )
            start = time()
            try:
                std_output = std_decoder.generate(prompt=prompt_text, max_new_tokens=max_new_tokens, temperature=0.0)
            except Exception as e:
                rprint(f"  [red]Standard generation failed: {e}[/red]")
                std_output = prompt_text
            std_elapsed = time() - start
            del std_decoder
        else:
            std_elapsed = spec_elapsed

        speedup = round(std_elapsed / max(spec_elapsed, 0.001), 2) if has_draft else 1.0
        results.append({
            "prompt": prompt_text, "spec_tokens": spec_tokens,
            "spec_time": round(spec_elapsed, 3), "spec_tok_s": round(spec_tokens / max(spec_elapsed, 0.001), 1),
            "std_time": round(std_elapsed, 3), "speedup": speedup,
            "acceptance_rate": round(metrics.get("acceptance_rate", 0), 3),
            "avg_tokens_per_step": round(metrics.get("avg_tokens_per_step", 0), 2),
        })
        total_spec_time += spec_elapsed
        total_tokens += spec_tokens

        # Memory cleanup between prompts
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    total_standard_time = sum(r["std_time"] for r in results) if has_draft else total_spec_time
    avg_speedup = total_standard_time / max(total_spec_time, 0.001) if has_draft else 1.0

    table = Table(title="Benchmark Results", header_style="bold")
    table.add_column("Prompt", style="cyan", no_wrap=False)
    table.add_column("Tokens", justify="right")
    table.add_column("Spec (s)", justify="right")
    table.add_column("Std (s)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Accept", justify="right")

    for r in results:
        s_style = "green" if r["speedup"] > 1.5 else "yellow" if r["speedup"] > 1.0 else "red"
        table.add_row(r["prompt"][:40], str(r["spec_tokens"]), str(r["spec_time"]), str(r["std_time"]), f"[{s_style}]{r['speedup']}x[/{s_style}]", f"{r['acceptance_rate']:.0%}" if r["acceptance_rate"] > 0 else "\u2014")

    table.add_row("[bold]Total/Avg[/bold]", str(total_tokens), f"{total_spec_time:.1f}", f"{total_standard_time:.1f}", f"[bold green]{avg_speedup:.2f}x[/bold green]", "", style="bold")
    rprint(table)

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {"has_draft": has_draft, "draft_k": decoder.max_draft_tokens, "max_new_tokens": max_new_tokens},
        "summary": {"total_tokens": total_tokens, "total_spec_time": round(total_spec_time, 3), "total_standard_time": round(total_standard_time, 3), "avg_speedup": round(avg_speedup, 3)},
        "per_prompt": results,
    }
    json_path = Path("benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    rprint(f"\n[dim]Results saved to: {json_path}[/dim]")
