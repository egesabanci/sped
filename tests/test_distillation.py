"""Tests for PEFT distillation module (Phase 4).

Note: Full training loop tests are marked 'slow' and can be run with:
    pytest tests/test_distillation.py -v --run-slow
"""

from pathlib import Path
import tempfile
import torch
import pytest


# ── DistillSpec Init Tests ───────────────────────────────


class TestDistillSpecInit:
    def test_init_with_models(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sped.distillation.distillspec import DistillSpec

        model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        distiller = DistillSpec(
            draft_model=model,
            draft_tokenizer=tokenizer,
            target_model=model,
            target_tokenizer=tokenizer,
            lora_rank=4,
        )
        assert distiller.draft_model is not None
        assert distiller.target_model is not None

    def test_lora_applied(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sped.distillation.distillspec import DistillSpec

        model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        distiller = DistillSpec(
            draft_model=model,
            draft_tokenizer=tokenizer,
            target_model=model,
            target_tokenizer=tokenizer,
            lora_rank=4,
        )
        lora_params = sum(
            p.numel() for n, p in distiller.draft_model.named_parameters()
            if "lora" in n
        )
        assert lora_params > 0

    def test_detect_attention_modules(self):
        from transformers import AutoModelForCausalLM
        from sped.distillation.distillspec import DistillSpec

        model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        modules = DistillSpec._detect_attention_modules(model)
        assert len(modules) > 0
        assert isinstance(modules, list)

    def test_kl_divergence(self):
        from sped.distillation.distillspec import DistillSpec

        logits = torch.randn(2, 4, 10)
        kl = DistillSpec._kl_divergence(logits, logits, temperature=1.0)
        assert kl.item() >= -1e-5  # near-zero due to floating point


# ── On-Policy Generation Tests (#14) ─────────────────────


class TestOnPolicyGeneration:
    def test_generate_on_policy_small(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sped.distillation.distillspec import DistillSpec

        model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        distiller = DistillSpec(
            draft_model=model,
            draft_tokenizer=tokenizer,
            target_model=model,
            target_tokenizer=tokenizer,
            lora_rank=4,
        )

        prompts = ["Hello world", "Test prompt"]
        sequences = distiller._generate_on_policy(
            prompts,
            gen_temperature=0.7,
            gen_tokens_per_prompt=5,  # small for speed
        )
        assert sequences is not None
        assert sequences.shape[0] == len(prompts)
        assert sequences.shape[1] > 0

    def test_generate_empty_prompts(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sped.distillation.distillspec import DistillSpec

        model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        distiller = DistillSpec(
            draft_model=model,
            draft_tokenizer=tokenizer,
            target_model=model,
            target_tokenizer=tokenizer,
            lora_rank=4,
        )

        sequences = distiller._generate_on_policy(
            [], gen_temperature=0.7, gen_tokens_per_prompt=5,
        )
        assert sequences is not None


# ── Training Loop Tests (#15) ────────────────────────────


class TestTrainingLoop:
    @pytest.mark.slow
    def test_distill_runs(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import Dataset
        from sped.distillation.distillspec import DistillSpec

        model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        distiller = DistillSpec(
            draft_model=model,
            draft_tokenizer=tokenizer,
            target_model=model,
            target_tokenizer=tokenizer,
            lora_rank=4,
        )

        data = Dataset.from_list([
            {"text": "Hello world this is a test."},
            {"text": "Another test prompt for distillation."},
        ] * 3)

        trained = distiller.distill(
            dataset=data,
            text_column="text",
            batch_size=2,
            num_epochs=1,
            max_length=32,
            log_every_steps=5,
            validation_split=0.0,
        )
        assert trained is not None


# ── Validation Tests (#16) ───────────────────────────────


class TestAcceptanceValidation:
    def test_extract_text(self):
        from sped.distillation.distillspec import DistillSpec

        assert DistillSpec._extract_text("hello") == "hello"
        assert DistillSpec._extract_text({"content": "hi"}) == "hi"
        assert DistillSpec._extract_text({"text": "hey"}) == "hey"
        chat = {"messages": [{"role": "user", "content": "Hello"}]}
        assert DistillSpec._extract_text(chat) == "Hello"
        result = DistillSpec._extract_text({})
        assert isinstance(result, str)

    @pytest.mark.slow
    def test_measure_acceptance_rate(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sped.distillation.distillspec import DistillSpec

        model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        distiller = DistillSpec(
            draft_model=model,
            draft_tokenizer=tokenizer,
            target_model=model,
            target_tokenizer=tokenizer,
            lora_rank=4,
        )

        metrics = distiller.measure_acceptance_rate(
            prompts=["Hello world"],
            draft_k=2,
            temperature=0.0,
        )
        assert "acceptance_rate" in metrics
        assert "avg_tokens_per_step" in metrics


# ── Save/Load Tests ─────────────────────────────────────


class TestSaveLoad:
    def test_save_adapter(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sped.distillation.distillspec import DistillSpec

        model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        distiller = DistillSpec(
            draft_model=model,
            draft_tokenizer=tokenizer,
            target_model=model,
            target_tokenizer=tokenizer,
            lora_rank=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test-adapter"
            distiller.save_adapter(save_path)
            assert (save_path / "adapter_config.json").exists()
            assert (save_path / "adapter_model.safetensors").exists() or \
                   any((save_path / f).exists() for f in ["adapter_model.bin"])


# ── Import Tests ─────────────────────────────────────────


class TestDistillationImports:
    def test_import_distillspec(self):
        from sped.distillation import DistillSpec
        assert DistillSpec is not None
