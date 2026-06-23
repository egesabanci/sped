"""sped serve — interactive inference with speculative decoding."""

import typer
from pathlib import Path
from typing import Optional
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout

app = typer.Typer(
    name="serve",
    help="Run inference with speculative decoding.",
    no_args_is_help=True,
)


@app.callback()
def callback():
    pass


@app.command()
def run(
    target: str = typer.Option(
        ..., "--target", "-t",
        help="Target model ID or path (e.g. meta-llama/Llama-3.1-70B)",
    ),
    draft: Optional[str] = typer.Option(
        None, "--draft", "-d",
        help="Draft model ID or path (e.g. Qwen/Qwen2.5-0.5B). "
             "If omitted, runs standard autoregressive (no speculation).",
    ),
    draft_lora: Optional[Path] = typer.Option(
        None, "--draft-lora",
        help="Path to pre-trained LoRA adapter for the draft model",
        exists=True,
        file_okay=False,
        dir_okay=True,
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
        help="Single prompt mode (non-interactive). Provide a prompt and exit.",
    ),
    benchmark: bool = typer.Option(
        False, "--benchmark",
        help="Run benchmark mode on standard prompts",
    ),
):
    """Run interactive or single-prompt inference with speculative decoding.

    If --draft is provided, uses speculative decoding. Otherwise falls back
    to standard autoregressive generation for comparison.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sped.core.speculative_decoding import SpeculativeDecoder
    from sped.utils.tokenizer_utils import check_vocab_compatibility

    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rprint(Panel.fit(
        f"[bold]Target:[/bold] {target}\n"
        f"[bold]Draft:[/bold]  {draft or 'none (standard mode)'}\n"
        f"[bold]Device:[/bold] {device}  [bold]Draft K:[/bold] {draft_k}  [bold]Temp:[/bold] {temperature}",
        title="⚡ sped serve",
    ))

    # Load target model
    rprint(f"\n[bold]Loading target model[/bold]: [cyan]{target}[/cyan]")
    target_tokenizer = AutoTokenizer.from_pretrained(target)
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    target_model = AutoModelForCausalLM.from_pretrained(
        target, torch_dtype="auto", device_map=device
    )
    target_model.eval()
    rprint(f"  ✓ [green]{sum(p.numel() for p in target_model.parameters()) / 1e9:.1f}B[/green] params")

    # Load draft model (optional)
    draft_model = None
    draft_tokenizer = None
    vocab_aligner = None

    if draft is not None:
        rprint(f"[bold]Loading draft model[/bold]: [cyan]{draft}[/cyan]")
        from peft import PeftModel

        draft_tokenizer = AutoTokenizer.from_pretrained(draft if draft_lora is None else draft_lora)
        if draft_tokenizer.pad_token is None:
            draft_tokenizer.pad_token = draft_tokenizer.eos_token

        base_draft = AutoModelForCausalLM.from_pretrained(
            draft, torch_dtype="auto", device_map=device
        )
        if draft_lora is not None:
            draft_model = PeftModel.from_pretrained(base_draft, str(draft_lora))
            rprint(f"  ✓ [green]LoRA adapter loaded from[/green] {draft_lora}")
        else:
            draft_model = base_draft
        draft_model.eval()
        rprint(f"  ✓ [green]{sum(p.numel() for p in draft_model.parameters()) / 1e9:.1f}B[/green] params")

        # Check vocabulary compatibility
        compat, overlap = check_vocab_compatibility(draft_tokenizer, target_tokenizer)
        if compat:
            rprint(f"  ✓ Vocabularies match ({overlap:.1%} overlap) — no alignment needed")
            align = "none"
        else:
            rprint(f"  ⚠ Vocabularies differ ({overlap:.1%} overlap) — using [bold]{align}[/bold] alignment")
            if align == "auto":
                align = "hybrid"

        # Build vocab aligner if needed
        if align != "none":
            from sped.vocab_agnostic.alignment import VocabAligner
            vocab_aligner = VocabAligner(
                target_tokenizer=target_tokenizer,
                draft_tokenizer=draft_tokenizer,
                strategy=align,
            )
            rprint(f"  ✓ Vocab aligner created (strategy: {align})")

    # Create decoder
    decoder = SpeculativeDecoder(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        draft_model=draft_model,
        draft_tokenizer=draft_tokenizer,
        vocab_aligner=vocab_aligner,
        max_draft_tokens=draft_k,
        device=device,
    )

    # Benchmark mode
    if benchmark:
        _run_benchmark(decoder, target_tokenizer, device)
        return

    # Single prompt mode
    if prompt is not None:
        _generate_and_show(decoder, prompt, max_new_tokens, temperature)
        return

    # Interactive REPL
    _run_repl(decoder, max_new_tokens, temperature)


def _generate_and_show(decoder, prompt: str, max_new_tokens: int, temperature: float):
    """Generate a single response and print it."""
    from time import time

    rprint(f"\n[bold]Prompt:[/bold] {prompt}\n")
    rprint("[bold]Response:[/bold]")

    start = time()
    output = decoder.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        verbose=False,
    )
    elapsed = time() - start

    # Extract only the generated part (strip the prompt)
    response = output[len(prompt):] if output.startswith(prompt) else output
    rprint(f"{response}\n")

    tokens = len(decoder.target_tokenizer.encode(response))
    rprint(f"[dim]─ {tokens} tokens in {elapsed:.1f}s ({tokens/elapsed:.1f} tok/s)[/dim]")


def _run_repl(decoder, max_new_tokens: int, temperature: float):
    """Run interactive REPL."""
    from time import time

    rprint(f"\n[bold green]Interactive mode[/bold green] — type your prompts or /help for commands\n")

    while True:
        try:
            prompt = input("[bold]»[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            rprint("\n[yellow]Bye![/yellow]")
            break

        if not prompt:
            continue

        if prompt.startswith("/"):
            _handle_command(prompt, decoder)
            continue

        start = time()
        output = decoder.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=False,
        )
        elapsed = time() - start

        response = output[len(prompt):] if output.startswith(prompt) else output
        rprint(f"{response}")
        tokens = len(decoder.target_tokenizer.encode(response))
        rprint(f"[dim]─ {tokens} tokens in {elapsed:.1f}s ({tokens/elapsed:.1f} tok/s)[/dim]\n")


def _handle_command(cmd: str, decoder):
    """Handle REPL commands."""
    cmd = cmd.lower().strip()
    if cmd in ("/exit", "/quit", "/q"):
        rprint("[yellow]Bye![/yellow]")
        raise SystemExit(0)
    elif cmd in ("/help", "/h", "/?"):
        rprint(Panel.fit(
            "[bold]/help[/bold]  — show this help\n"
            "[bold]/stats[/bold] — show cumulative generation stats\n"
            "[bold]/exit[/bold]  — quit\n"
            "[bold]/clear[/bold] — clear conversation",
            title="Commands",
        ))
    elif cmd == "/stats":
        rprint("[yellow]Stats will be available after full metrics collector implementation[/yellow]")
    elif cmd == "/clear":
        import os
        os.system("cls" if os.name == "nt" else "clear")
    else:
        rprint(f"[red]Unknown command:[/red] {cmd}. Type /help for commands.")


def _run_benchmark(decoder, tokenizer, device):
    """Run a quick benchmark on standard prompts."""
    from time import time
    import json

    benchmarks = [
        "What is the capital of France?",
        "Explain quantum computing in three sentences.",
        "Write a Python function to compute fibonacci numbers.",
        "What are the main differences between TCP and UDP?",
    ]

    rprint(f"\n[bold]Running benchmark on {len(benchmarks)} prompts...[/bold]\n")
    results = []

    for i, prompt in enumerate(benchmarks):
        rprint(f"  [{i+1}/{len(benchmarks)}] {prompt[:60]}...")

        start = time()
        output = decoder.generate(prompt=prompt, max_new_tokens=128, temperature=0.0)
        elapsed = time() - start

        response = output[len(prompt):] if output.startswith(prompt) else output
        tokens = len(tokenizer.encode(response))
        results.append({
            "prompt": prompt,
            "tokens": tokens,
            "time_seconds": round(elapsed, 2),
            "tokens_per_second": round(tokens / elapsed, 1) if elapsed > 0 else 0,
        })

    # Summary table
    table = Table(title="Benchmark Results")
    table.add_column("Prompt", style="cyan")
    table.add_column("Tokens", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("tok/s", justify="right")

    total_tokens = 0
    total_time = 0
    for r in results:
        table.add_row(r["prompt"][:40], str(r["tokens"]), str(r["time_seconds"]), str(r["tokens_per_second"]))
        total_tokens += r["tokens"]
        total_time += r["time_seconds"]

    table.add_row(
        "[bold]Total[/bold]",
        str(total_tokens),
        f"{total_time:.1f}",
        f"{total_tokens / max(total_time, 0.1):.1f}",
        style="bold",
    )

    rprint(table)
