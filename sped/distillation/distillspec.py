"""DistillSpec: knowledge distillation to align a draft model with a target."""

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from accelerate import Accelerator


class DistillSpec:
    """Aligns a small draft model to a target model via on-policy KL distillation.

    Uses LoRA (PEFT) for efficient training — only a small fraction of
    the draft model's parameters are updated.
    """

    def __init__(
        self,
        draft_model: PreTrainedModel,
        draft_tokenizer: PreTrainedTokenizer,
        target_model: PreTrainedModel,
        target_tokenizer: PreTrainedTokenizer,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        device: str = "auto",
    ):
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.device = device

        # Wrap draft with LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        self.draft_model = get_peft_model(draft_model, lora_config)
        self.draft_model.print_trainable_parameters()

    def distill(
        self,
        dataset: Dataset,
        text_column: str = "text",
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        max_length: int = 512,
        temperature: float = 1.0,
    ):
        """Run on-policy distillation.

        The draft model generates sequences (on-policy), and we minimize
        KL divergence between draft and target logits.

        Args:
            dataset: Hugging Face Dataset with a text column.
            text_column: Column name containing prompt/text.
            batch_size: Training batch size.
            learning_rate: Learning rate for LoRA optimizer.
            num_epochs: Number of distillation epochs.
            max_length: Max token length.
            temperature: Distillation temperature.
        """
        accelerator = Accelerator()

        optimizer = torch.optim.AdamW(
            self.draft_model.parameters(), lr=learning_rate
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        self.draft_model, optimizer, dataloader = accelerator.prepare(
            self.draft_model, optimizer, dataloader
        )

        self.target_model.eval()
        self.draft_model.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in dataloader:
                texts = batch[text_column]
                # Tokenize with draft tokenizer
                inputs = self.draft_tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(accelerator.device)

                # Forward pass through draft (on-policy generation)
                with torch.no_grad():
                    draft_gen = self.draft_model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=True,
                        temperature=0.7,
                    )

                # Get logits from both models
                with torch.no_grad():
                    target_logits = self.target_model(draft_gen).logits

                draft_outputs = self.draft_model(draft_gen)
                draft_logits = draft_outputs.logits

                # KL divergence loss
                loss = self._kl_divergence(
                    draft_logits, target_logits, temperature
                )

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} — KL loss: {avg_loss:.4f}")

        return self.draft_model

    def _kl_divergence(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Compute KL divergence between student and teacher distributions."""
        student_probs = torch.log_softmax(
            student_logits / temperature, dim=-1
        )
        teacher_probs = torch.softmax(
            teacher_logits / temperature, dim=-1
        )
        kl = torch.nn.functional.kl_div(
            student_probs, teacher_probs, reduction="batchmean", log_target=False
        )
        return kl * (temperature ** 2)  # Scale by T^2 per distillation literature
