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
        help="Hugging Face dataset ID for distillation (e.g. HuggingFaceH4/ultrachat_200k)",
    ),
    text_column: str = typer.Option(
        "text", "--text-column",
        help="Column name containing text in the dataset",
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
        min=64, max=8192,
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
        "auto", "--backend", "-b",
        help="Backend: auto, hf, unsloth",
    ),
):
    """Run PEFT distillation to align a draft model to a target model.

    Uses on-policy DistillSpec: generates continuations with the draft model,
    then minimizes KL divergence against target model logits via LoRA.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sped.distillation.distillspec import DistillSpec

    rprint(Panel.fit(
        f"[bold]DistillSpec[/bold] — {draft} → {target}\n"
        f"  LoRA rank={lora_rank}  epochs={epochs}  batch={batch_size}  lr={learning_rate}",
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

    if backend == "unsloth":
        from unsloth import FastLanguageModel

        # Load target model via Unsloth
        rprint(f"\n[bold]Loading target model (Unsloth)[/bold]: [cyan]{target}[/cyan]")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Loading target model...", total=None)
            target_model, target_tokenizer = FastLanguageModel.from_pretrained(
                model_name=target,
                max_seq_length=max_length,
                dtype=None,
                load_in_4bit=True,
                device_map=device,
            )
            target_model.eval()

        # Load draft model via Unsloth
        rprint(f"\n[bold]Loading draft model (Unsloth)[/bold]: [cyan]{draft}[/cyan]")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Loading draft model...", total=None)
            draft_model, draft_tokenizer = FastLanguageModel.from_pretrained(
                model_name=draft,
                max_seq_length=max_length,
                dtype=None,
                load_in_4bit=True,
                device_map=device,
            )
    else:
        # Standard HF loading
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
    rprint(f"\n[bold]Loading dataset[/bold]: [cyan]{dataset}[/cyan]")
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Loading dataset...", total=None)
        dataset_obj = load_dataset(dataset, split="train")
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
):
    """Validate a trained draft adapter by measuring acceptance rate."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    rprint(Panel.fit(
        f"[bold]Validation[/bold] — {draft} → {target}",
        title="⚡ sped distil validate",
    ))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load target
    target_tokenizer = AutoTokenizer.from_pretrained(target)
    target_model = AutoModelForCausalLM.from_pretrained(
        target, torch_dtype="auto", device_map=device
    )
    target_model.eval()

    # Load draft (possibly with LoRA adapter)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft)
    base_draft = AutoModelForCausalLM.from_pretrained(
        draft, torch_dtype="auto", device_map=device
    )
    try:
        draft_model = PeftModel.from_pretrained(base_draft, draft)
    except Exception:
        draft_model = base_draft

    # Minimal validation: measure acceptance rate
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train")
    prompts = [dataset[i]["messages"][0]["content"] for i in range(min(num_prompts, len(dataset)))]

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
