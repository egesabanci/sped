"""sped distil — PEFT distillation of a draft model to a target model."""

import typer
from pathlib import Path
from typing import Optional
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from rich.panel import Panel

app = typer.Typer(
    name="distil",
    help="Distil a draft model to a target model using PEFT (LoRA).",
    no_args_is_help=True,
)


@app.callback()
def callback():
    pass


@app.command()
def run(
    draft: str = typer.Option(
        ..., "--draft", "-d",
        help="Draft model ID or path (e.g. Qwen/Qwen2.5-0.5B)",
    ),
    target: str = typer.Option(
        ..., "--target", "-t",
        help="Target model ID or path (e.g. meta-llama/Llama-3.1-70B)",
    ),
    dataset: str = typer.Option(
        ..., "--dataset",
        help="Hugging Face dataset ID or local save_to_disk path",
    ),
    text_column: str = typer.Option(
        "text", "--text-column",
        help="Column name containing text in the dataset",
    ),
    split: str = typer.Option(
        "auto", "--split",
        help="Dataset split to use: auto, train_sft, train_gen, train, or custom name",
    ),
    lora_rank: int = typer.Option(
        8, "--lora-rank", "-r",
        help="LoRA rank (r)",
        min=1, max=64,
    ),
    lora_alpha: int = typer.Option(
        16, "--lora-alpha", "-a",
        help="LoRA alpha scaling",
        min=1, max=128,
    ),
    epochs: int = typer.Option(
        3, "--epochs", "-e",
        help="Number of distillation epochs",
        min=1, max=100,
    ),
    batch_size: int = typer.Option(
        4, "--batch-size", "-b",
        help="Training batch size per GPU",
        min=1, max=256,
    ),
    learning_rate: float = typer.Option(
        5e-5, "--learning-rate", "-lr",
        help="Learning rate",
    ),
    max_length: int = typer.Option(
        512, "--max-length", "-ml",
        help="Maximum token length for training sequences",
        min=64, max=16384,
    ),
    temperature: float = typer.Option(
        1.0, "--temperature", "-T",
        help="Distillation temperature (higher = softer targets)",
        min=0.1, max=10.0,
    ),
    output: Path = typer.Option(
        "./draft-lora", "--output", "-o",
        help="Output directory for LoRA adapter weights",
    ),
    device: str = typer.Option(
        "auto", "--device",
        help="Device to use: auto, cuda, cpu, mps",
    ),
    backend: str = typer.Option(
        "auto", "--backend",
        help="Backend: auto, hf, unsloth",
    ),
    draft_dtype: str = typer.Option(
        "bf16", "--draft-dtype",
        help="Draft model precision: bf16 (full) or 4bit (quantized)",
    ),
    # ── Training tuning parameters (issues #67, #78) ─────────────────
    gradient_accumulation_steps: int = typer.Option(
        1, "--grad-accum",
        help="Gradient accumulation steps (effective batch = batch_size × this)",
        min=1, max=32,
    ),
    warmup_steps: int = typer.Option(
        -1, "--warmup-steps",
        help="LR warmup steps (default -1 = auto: 5%% of total steps)",
        min=-1,
    ),
    max_grad_norm: float = typer.Option(
        1.0, "--max-grad-norm",
        help="Gradient clipping max norm",
        min=0.0,
    ),
    mixed_precision: Optional[str] = typer.Option(
        None, "--mixed-precision",
        help="Mixed precision: bf16, fp16, or none (auto-detect by default)",
    ),
    # ── On-policy generation (issues #67) ────────────────────────────
    on_policy_regen_every: int = typer.Option(
        200, "--on-policy-regen-every",
        help="Regenerate on-policy data every N steps (0 to disable)",
        min=0,
    ),
    on_policy_tokens_per_prompt: int = typer.Option(
        64, "--on-policy-tokens",
        help="Continuation tokens generated per prompt (on-policy)",
        min=1, max=512,
    ),
    on_policy_gen_temp: float = typer.Option(
        0.7, "--on-policy-temp",
        help="Generation temperature for on-policy data",
        min=0.0, max=2.0,
    ),
    on_policy_fraction: float = typer.Option(
        0.25, "--on-policy-fraction",
        help="Fraction of on-policy buffer to regenerate per cycle",
        min=0.05, max=1.0,
    ),
    # ── Validation (issues #67, #77) ─────────────────────────────────
    validation_split: float = typer.Option(
        0.05, "--validation-split",
        help="Fraction of dataset for acceptance-rate validation (0 to disable)",
        min=0.0, max=0.5,
    ),
    val_prompts: int = typer.Option(
        20, "--val-prompts",
        help="Number of prompts for validation",
        min=1, max=100,
    ),
    val_draft_k: int = typer.Option(
        5, "--val-draft-k",
        help="Draft K (speculation width) for validation",
        min=1, max=20,
    ),
    val_max_new_tokens: int = typer.Option(
        32, "--val-max-new-tokens",
        help="Max new tokens generated per validation prompt",
        min=1, max=256,
    ),
    # ── Logging & checkpointing (issues #67, #75, #79) ───────────────
    log_every_steps: int = typer.Option(
        10, "--log-every",
        help="Log metrics every N steps",
        min=1,
    ),
    save_every_steps: int = typer.Option(
        500, "--save-every",
        help="Save checkpoint every N steps (requires --checkpoint-dir)",
        min=1,
    ),
    checkpoint_dir: Optional[Path] = typer.Option(
        None, "--checkpoint-dir",
        help="Directory for checkpoints (enables checkpointing)",
    ),
    resume_from: Optional[Path] = typer.Option(
        None, "--resume-from",
        help="Path to a checkpoint directory to resume from",
    ),
):
    """Run PEFT distillation to align a draft model to a target model.

    Uses on-policy DistillSpec: generates continuations with the draft model,
    then minimizes KL divergence against target model logits via LoRA.
    """
    import torch
    from datasets import load_dataset
    from sped.distillation.distillspec import DistillSpec

    rprint(Panel.fit(
        f"[bold]DistillSpec[/bold] — {draft} → {target}\n"
        f"  LoRA rank={lora_rank}  epochs={epochs}  batch={batch_size}  lr={learning_rate}\n"
        f"  backend={backend}  max_length={max_length}  validation_split={validation_split}",
        title="⚡ sped distil",
    ))

    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve backend
    if backend == "auto":
        try:
            import unsloth  # noqa: F401
            backend = "unsloth"
        except ImportError:
            backend = "hf"

    # Resolve mixed precision string to None for accelerator
    mp_value = mixed_precision
    if mp_value is not None and mp_value.lower() in ("none", "no", ""):
        mp_value = None

    if backend == "unsloth":
        from sped.utils.unsloth_cache import load_unsloth_model

        # Resolve draft dtype flag
        draft_4bit = draft_dtype == "4bit"
        if draft_4bit:
            rprint(f"[yellow]  Draft: 4-bit quantized[/yellow]")
        else:
            rprint(f"[dim]  Draft: full bf16 precision[/dim]")

        # Load models
        rprint(f"\n[bold]Loading target model (Unsloth 4-bit)[/bold]: [cyan]{target}[/cyan]")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Loading target model...", total=None)
            target_model, target_tokenizer = load_unsloth_model(
                target, max_seq_length=max_length, load_in_4bit=True,
                device=device, verbose=True,
            )
            target_model.eval()

        rprint(f"\n[bold]Loading draft model (Unsloth)[/bold]: [cyan]{draft}[/cyan]")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Loading draft model...", total=None)
            draft_model, draft_tokenizer = load_unsloth_model(
                draft, max_seq_length=max_length, load_in_4bit=draft_4bit,
                device=device, verbose=True,
            )
    else:
        # Standard HF loading
        from transformers import AutoModelForCausalLM, AutoTokenizer

        rprint(f"\n[bold]Loading target model[/bold]: [cyan]{target}[/cyan]")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Loading target model...", total=None)
            target_tokenizer = AutoTokenizer.from_pretrained(target)
            target_model = AutoModelForCausalLM.from_pretrained(
                target, torch_dtype="auto", device_map=device
            )
            target_model.eval()

        rprint(f"\n[bold]Loading draft model[/bold]: [cyan]{draft}[/cyan]")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Loading draft model...", total=None)
            draft_tokenizer = AutoTokenizer.from_pretrained(draft)
            draft_model = AutoModelForCausalLM.from_pretrained(
                draft, torch_dtype="auto", device_map=device
            )

    rprint(f"  ✓ Target: [green]{sum(p.numel() for p in target_model.parameters()) / 1e9:.1f}B[/green] params")
    rprint(f"  ✓ Draft:  [green]{sum(p.numel() for p in draft_model.parameters()) / 1e9:.1f}B[/green] params")

    # Load dataset
    rprint(f"\n[bold]Loading dataset[/bold]: [cyan]{dataset}[/cyan]" + (
        f" [dim](split={split})[/dim]" if split != "auto" else ""
    ))
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Loading dataset...", total=None)
        dataset_obj = _load_dataset(dataset, split)
    rprint(f"  ✓ Dataset loaded: [green]{len(dataset_obj):,}[/green] examples")

    # Initialize DistillSpec
    rprint(f"\n[bold]Initializing DistillSpec with LoRA rank={lora_rank}...[/bold]")
    distiller = DistillSpec(
        draft_model=draft_model,
        draft_tokenizer=draft_tokenizer,
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=device,
        backend=backend,
    )

    # Run distillation
    rprint(f"\n[bold]Starting distillation for {epochs} epochs...[/bold]")
    output.mkdir(parents=True, exist_ok=True)

    trained_model = distiller.distill(
        dataset=dataset_obj,
        text_column=text_column,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=epochs,
        max_length=max_length,
        temperature=temperature,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        mixed_precision=mp_value,
        on_policy_regenerate_every=on_policy_regen_every,
        on_policy_tokens_per_prompt=on_policy_tokens_per_prompt,
        on_policy_gen_temp=on_policy_gen_temp,
        on_policy_fraction=on_policy_fraction,
        validation_split=validation_split,
        val_prompts=val_prompts,
        val_draft_k=val_draft_k,
        val_max_new_tokens=val_max_new_tokens,
        checkpoint_dir=checkpoint_dir,
        save_every_steps=save_every_steps,
        log_every_steps=log_every_steps,
        resume_from=resume_from,
    )

    # Save LoRA adapter
    rprint(f"\n[bold]Saving LoRA adapter[/bold] → [cyan]{output}[/cyan]")
    trained_model.save_pretrained(str(output))
    draft_tokenizer.save_pretrained(str(output))

    rprint(f"\n[green]✓ Distillation complete![/green]")
    rprint(f"  Adapter saved to: [cyan]{output.resolve()}[/cyan]")
    rprint(f"  Use: [bold]sped serve --draft-lora {output.resolve()}[/bold]")


@app.command()
def validate(
    draft: str = typer.Option(
        ..., "--draft", "-d",
        help="Draft model ID, path, or LoRA adapter directory",
    ),
    target: str = typer.Option(
        ..., "--target", "-t",
        help="Target model ID or path",
    ),
    num_prompts: int = typer.Option(
        100, "--num-prompts", "-n",
        help="Number of validation prompts",
        min=10, max=1000,
    ),
    draft_k: int = typer.Option(
        5, "--draft-k", "-k",
        help="Number of draft tokens per speculation step",
        min=1, max=20,
    ),
    backend: str = typer.Option(
        "auto", "--backend",
        help="Backend: auto, hf, unsloth",
    ),
    draft_lora: Optional[Path] = typer.Option(
        None, "--draft-lora",
        help="LoRA adapter path to apply on top of the draft model",
    ),
    target_4bit: bool = typer.Option(
        False, "--target-4bit",
        help="(Unsloth only) Load target in 4-bit quantization",
    ),
):
    """Validate a trained draft adapter by measuring acceptance rate."""
    import torch
    from rich.progress import Progress, SpinnerColumn, TextColumn

    rprint(Panel.fit(
        f"[bold]Validation[/bold] — {draft} → {target}",
        title="⚡ sped distil validate",
    ))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve backend
    resolved_backend = backend
    if resolved_backend == "auto":
        try:
            import unsloth  # noqa: F401
            resolved_backend = "unsloth"
        except ImportError:
            resolved_backend = "hf"

    if resolved_backend == "unsloth":
        from unsloth import FastLanguageModel
        from sped.utils.unsloth_cache import load_unsloth_model

        rprint(f"\n[bold]Loading target (Unsloth)[/bold]: [cyan]{target}[/cyan]")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as p:
            p.add_task("Loading target...", total=None)
            target_model, target_tokenizer = load_unsloth_model(
                target, max_seq_length=4096, load_in_4bit=target_4bit,
                device=device, verbose=True,
            )
            FastLanguageModel.for_inference(target_model)

        rprint(f"\n[bold]Loading draft (Unsloth)[/bold]: [cyan]{draft}[/cyan]")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as p:
            p.add_task("Loading draft...", total=None)
            # When a LoRA adapter is provided, load the adapter dir directly:
            # FastLanguageModel auto-loads the base model + applies the LoRA,
            # returning a PeftModel. Wrapping again with PeftModel.from_pretrained
            # would double-wrap and silently drop the adapter weights.
            draft_load_id = str(draft_lora) if draft_lora is not None else draft
            draft_model, draft_tokenizer = load_unsloth_model(
                draft_load_id, max_seq_length=4096, load_in_4bit=False,
                device=device, verbose=True,
            )
            if draft_lora is not None and not hasattr(draft_model, "peft_config"):
                # Fallback: backend did not auto-apply the adapter
                from peft import PeftModel
                draft_model = PeftModel.from_pretrained(draft_model, str(draft_lora))
            FastLanguageModel.for_inference(draft_model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        rprint(f"\n[bold]Loading target[/bold]: [cyan]{target}[/cyan]")
        target_tokenizer = AutoTokenizer.from_pretrained(target)
        target_model = AutoModelForCausalLM.from_pretrained(
            target, torch_dtype="auto", device_map=device
        )
        target_model.eval()

        rprint(f"\n[bold]Loading draft[/bold]: [cyan]{draft}[/cyan]")
        draft_tokenizer = AutoTokenizer.from_pretrained(draft)
        base_draft = AutoModelForCausalLM.from_pretrained(
            draft, torch_dtype="auto", device_map=device
        )
        lora_path = draft_lora if draft_lora is not None else draft
        try:
            draft_model = PeftModel.from_pretrained(base_draft, lora_path)
        except Exception:
            draft_model = base_draft

    # Build prompts
    prompts = _build_validation_prompts(num_prompts)

    total_accepted = 0
    total_drafted = 0

    for prompt in prompts:
        inputs = draft_tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            draft_ids = draft_model.generate(
                **inputs, max_new_tokens=draft_k, do_sample=True, temperature=0.7
            )
        draft_tokens = draft_ids[0, inputs.input_ids.shape[-1]:]

        if len(draft_tokens) == 0:
            continue

        # Target verification
        target_inputs = target_tokenizer(prompt, return_tensors="pt").to(device)
        combined = torch.cat([target_inputs.input_ids, draft_tokens.unsqueeze(0).to(target_model.device)], dim=-1)
        with torch.no_grad():
            outputs = target_model(combined)
        target_logits = outputs.logits[0, target_inputs.input_ids.shape[-1]-1:-1, :]

        draft_logits = draft_model(combined).logits[0, target_inputs.input_ids.shape[-1]-1:-1, :]

        # Simple acceptance check (greedy)
        draft_probs = torch.softmax(draft_logits, dim=-1)
        target_probs = torch.softmax(target_logits, dim=-1)

        for i, tok in enumerate(draft_tokens):
            total_drafted += 1
            if target_probs[i, tok] >= draft_probs[i, tok]:
                total_accepted += 1
            else:
                p = torch.rand(1).item()
                if p < (target_probs[i, tok] / draft_probs[i, tok]).item():
                    total_accepted += 1

    rate = total_accepted / max(total_drafted, 1)
    rprint(f"\n[bold]Acceptance Rate:[/bold] [green]{rate:.1%}[/green] ({total_accepted}/{total_drafted})")
    rprint(f"  Draft K: {draft_k}")
    if rate > 0.6:
        rprint(f"  [green]✓ Good alignment — ready for deployment[/green]")
    elif rate > 0.4:
        rprint(f"  [yellow]○ Moderate alignment — consider more distillation epochs[/yellow]")
    else:
        rprint(f"  [red]✗ Poor alignment — increase epochs or LoRA rank[/red]")


# ── Helpers ─────────────────────────────────────────────────────────────


def _load_dataset(dataset: str, split: str):
    """Load a dataset from disk or HuggingFace, honoring the split flag.

    Args:
        dataset: Local save_to_disk path or HuggingFace dataset ID.
        split: "auto" (preference order) or an explicit split name.

    Returns:
        A datasets.Dataset (single split).
    """
    from pathlib import Path as _Path
    from datasets import load_dataset as _load, load_from_disk, DatasetDict as _DatasetDict

    dataset_path = _Path(dataset)
    # A single Dataset save creates dataset_info.json; a DatasetDict
    # save creates dataset_dict.json. Detect either.
    is_local = dataset_path.exists() and (
        (dataset_path / "dataset_info.json").exists()
        or (dataset_path / "dataset_dict.json").exists()
    )
    if is_local:
        loaded = load_from_disk(dataset)
        if isinstance(loaded, _DatasetDict):
            if split != "auto":
                if split not in loaded:
                    raise ValueError(
                        f"Split '{split}' not found. Available: {list(loaded.keys())}"
                    )
                return loaded[split]
            # Auto-detect the first known SFT/gen split, fallback to first
            for _preferred in ("train_sft", "train_gen", "train", "sft"):
                if _preferred in loaded:
                    return loaded[_preferred]
            return loaded[list(loaded.keys())[0]]
        return loaded
    else:
        # Remote HuggingFace dataset
        if split == "auto":
            return _load(dataset, split="train")
        return _load(dataset, split=split)


def _build_validation_prompts(num_prompts: int) -> list[str]:
    """Build a list of validation prompts from ultrachat_200k."""
    from datasets import load_dataset
    try:
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train")
        prompts = [
            dataset[i]["messages"][0]["content"]
            for i in range(min(num_prompts, len(dataset)))
        ]
        return prompts
    except Exception:
        # Offline fallback to generic prompts
        return [
            "Hello, how are you today?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about the ocean.",
            "What is the capital of France?",
            "Tell me a joke about programming.",
        ] * (num_prompts // 5 + 1)