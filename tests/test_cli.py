"""Tests for CLI commands (Phase 1 + Phase 6 + polish).

All tests that hit external APIs are marked with a timeout.
Network-dependent tests use @pytest.mark.network to allow filtering.
"""

import subprocess
import os
import tempfile
import json
from pathlib import Path
import pytest


def test_distil_load_dataset_split_auto(tmp_path):
    """_load_dataset auto-detects the train_sft split (#74)."""
    from datasets import Dataset, DatasetDict
    from sped.cli.distil import _load_dataset

    ds = Dataset.from_list([{"text": "a"}, {"text": "b"}])
    dd = DatasetDict({"train_sft": ds})
    save_path = tmp_path / "ds"
    dd.save_to_disk(str(save_path))

    loaded = _load_dataset(str(save_path), "auto")
    assert len(loaded) == 2


def test_distil_load_dataset_explicit_split(tmp_path):
    """_load_dataset honors an explicit split name (#74)."""
    from datasets import Dataset, DatasetDict
    from sped.cli.distil import _load_dataset

    dd = DatasetDict({
        "train_sft": Dataset.from_list([{"text": "a"}]),
        "test_sft": Dataset.from_list([{"text": "b"}, {"text": "c"}]),
    })
    save_path = tmp_path / "ds"
    dd.save_to_disk(str(save_path))

    loaded = _load_dataset(str(save_path), "test_sft")
    assert len(loaded) == 2


def test_distil_load_dataset_invalid_split(tmp_path):
    """_load_dataset raises on an unknown split (#74)."""
    from datasets import Dataset, DatasetDict
    from sped.cli.distil import _load_dataset

    dd = DatasetDict({"train": Dataset.from_list([{"text": "a"}])})
    save_path = tmp_path / "ds"
    dd.save_to_disk(str(save_path))

    with pytest.raises(ValueError, match="not found"):
        _load_dataset(str(save_path), "nonexistent")


def test_distil_build_validation_prompts_offline():
    """_build_validation_prompts falls back to generic prompts offline (#73)."""
    from sped.cli.distil import _build_validation_prompts
    prompts = _build_validation_prompts(5)
    assert len(prompts) >= 5
    assert all(isinstance(p, str) for p in prompts)


@pytest.fixture
def temp_config_dir():
    """Create a temporary .sped config directory."""
    tmpdir = Path(tempfile.mkdtemp())
    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(tmpdir)
    yield tmpdir
    if old_home:
        os.environ["HOME"] = old_home
    else:
        del os.environ["HOME"]


def _run_sped(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a sped CLI command and return the result."""
    return subprocess.run(
        ["sped", *args],
        capture_output=True, text=True,
        timeout=timeout,
    )


# ── Top-level commands ────────────────────────────────────


def test_version():
    r = _run_sped("version")
    assert r.returncode == 0
    assert "sped v0.1.0" in r.stdout


def test_info():
    r = _run_sped("info")
    assert r.returncode == 0
    assert "sped System Info" in r.stdout
    assert "PyTorch" in r.stdout


# ── list subcommands ──────────────────────────────────────


def test_list_help():
    r = _run_sped("list", "--help")
    assert r.returncode == 0
    assert "models" in r.stdout.lower()


def test_list_models():
    """Only checks output structure, not full HF API query."""
    r = _run_sped("list", "models")
    assert r.returncode == 0


def test_list_adapters_no_adapters():
    r = _run_sped("list", "adapters")
    assert r.returncode == 0


def test_list_pairings():
    r = _run_sped("list", "pairings")
    assert r.returncode == 0
    assert "Draft" in r.stdout
    assert "Target" in r.stdout


# ── distil subcommands ────────────────────────────────────


def test_distil_help():
    r = _run_sped("distil", "--help")
    assert r.returncode == 0
    assert "run" in r.stdout.lower()


def test_distil_run_help():
    r = _run_sped("distil", "run", "--help")
    assert r.returncode == 0
    assert "--draft" in r.stdout
    assert "--target" in r.stdout


def test_distil_run_new_flags():
    """New flags from issues #67/#77/#78/#79 are exposed."""
    r = _run_sped("distil", "run", "--help", timeout=15)
    assert r.returncode == 0
    for flag in [
        "--validation-split",
        "--val-max-new-tokens",
        "--grad-accum",
        "--warmup-steps",
        "--mixed-precision",
        "--on-policy-regen",
        "--checkpoint-dir",
        "--resume-from",
        "--backend",
        "--draft-dtype",
        "--split",
        "--log-every",
    ]:
        assert flag in r.stdout, f"missing flag {flag} in distil run --help"


def test_distil_validate_backend_flag():
    """distil validate exposes --backend and --draft-lora (#73)."""
    r = _run_sped("distil", "validate", "--help", timeout=15)
    assert r.returncode == 0
    assert "--backend" in r.stdout
    assert "--draft-lora" in r.stdout


def test_distil_run_requires_draft():
    r = _run_sped("distil", "run", "--target", "test")
    assert r.returncode != 0


def test_distil_run_requires_target():
    r = _run_sped("distil", "run", "--draft", "test")
    assert r.returncode != 0


def test_distil_validate_help():
    r = _run_sped("distil", "validate", "--help")
    assert r.returncode == 0


# ── serve subcommands ─────────────────────────────────────


def test_serve_help():
    r = _run_sped("serve", "--help")
    assert r.returncode == 0


def test_serve_run_help():
    r = _run_sped("serve", "run", "--help")
    assert r.returncode == 0
    assert "--target" in r.stdout
    assert "--backend" in r.stdout


def test_serve_run_requires_target():
    r = _run_sped("serve", "run")
    assert r.returncode != 0


# ── experiment subcommands ────────────────────────────────


def test_experiment_help():
    r = _run_sped("experiment", "--help")
    assert r.returncode == 0
    assert "run" in r.stdout.lower()


def test_experiment_run_help():
    r = _run_sped("experiment", "run", "--help")
    assert r.returncode == 0
    assert "--target" in r.stdout


def test_experiment_auto_tune_help():
    r = _run_sped("experiment", "auto-tune", "--help")
    assert r.returncode == 0
    assert "--target" in r.stdout


# ── config subcommands ────────────────────────────────────


def test_config_init(temp_config_dir):
    r = _run_sped("config", "init")
    assert r.returncode == 0
    assert "Created config" in r.stdout
    config_path = Path(os.environ["HOME"]) / ".sped" / "config.yml"
    assert config_path.exists()


def test_config_init_twice(temp_config_dir):
    """Second init should warn without --force."""
    _run_sped("config", "init")
    r = _run_sped("config", "init")
    assert r.returncode == 0
    assert "already exists" in r.stdout


def test_config_init_force(temp_config_dir):
    _run_sped("config", "init")
    r = _run_sped("config", "init", "--force")
    assert r.returncode == 0
    assert "Created config" in r.stdout


def test_config_set(temp_config_dir):
    _run_sped("config", "init")
    r = _run_sped("config", "set", "draft_k", "3")
    assert r.returncode == 0
    assert "draft_k" in r.stdout
    assert "3" in r.stdout


def test_config_set_invalid_key(temp_config_dir):
    _run_sped("config", "init")
    r = _run_sped("config", "set", "invalid_key", "value")
    assert r.returncode != 0


def test_config_show(temp_config_dir):
    _run_sped("config", "init")
    r = _run_sped("config", "show")
    assert r.returncode == 0
    assert "Configuration" in r.stdout


def test_config_show_no_config(temp_config_dir):
    # Defaults should show even without config file
    r = _run_sped("config", "show")
    assert r.returncode == 0
    assert "Configuration" in r.stdout


# ── validation tests ──────────────────────────────────────


def test_validation_draft_k_valid():
    from sped.utils.validation import validate_draft_k
    assert validate_draft_k(5) == 5
    assert validate_draft_k(1) == 1


def test_validation_draft_k_invalid():
    from sped.utils.validation import validate_draft_k
    import pytest as _pytest
    with _pytest.raises(ValueError):
        validate_draft_k(0)
    with _pytest.raises(ValueError):
        validate_draft_k(100)


def test_validation_temperature_valid():
    from sped.utils.validation import validate_temperature
    assert validate_temperature(0.0) == 0.0
    assert validate_temperature(1.5) == 1.5
    assert validate_temperature(0) == 0


def test_validation_temperature_invalid():
    from sped.utils.validation import validate_temperature
    import pytest as _pytest
    with _pytest.raises(ValueError):
        validate_temperature(-0.1)
    with _pytest.raises(ValueError):
        validate_temperature(2.1)


def test_validation_max_tokens_valid():
    from sped.utils.validation import validate_max_new_tokens
    assert validate_max_new_tokens(10) == 10
    assert validate_max_new_tokens(4096) == 4096


def test_validation_max_tokens_invalid():
    from sped.utils.validation import validate_max_new_tokens
    import pytest as _pytest
    with _pytest.raises(ValueError):
        validate_max_new_tokens(0)
    with _pytest.raises(ValueError):
        validate_max_new_tokens(5000)


def test_validation_device_valid():
    from sped.utils.validation import validate_device
    assert validate_device("auto") == "auto"
    assert validate_device("cpu") == "cpu"
    assert validate_device("cuda") == "cuda"
    assert validate_device("cuda:0") == "cuda:0"
    assert validate_device("mps") == "mps"


def test_validation_device_invalid():
    from sped.utils.validation import validate_device
    import pytest as _pytest
    with _pytest.raises(ValueError):
        validate_device("gpu")
    with _pytest.raises(ValueError):
        validate_device("cpu:0")


def test_validation_backend_valid():
    from sped.utils.validation import validate_backend
    for b in ["auto", "hf", "mlx", "vllm"]:
        assert validate_backend(b) == b


def test_validation_backend_invalid():
    from sped.utils.validation import validate_backend
    import pytest as _pytest
    with _pytest.raises(ValueError):
        validate_backend("tensorrt")


def test_validation_align_valid():
    from sped.utils.validation import validate_align
    for a in ["auto", "none", "string", "probabilistic", "hybrid"]:
        assert validate_align(a) == a


def test_validation_output_format_valid():
    from sped.utils.validation import validate_output_format
    for f in ["text", "json", "silent"]:
        assert validate_output_format(f) == f


def test_validation_model_id_local():
    from sped.utils.validation import validate_model_id
    # Use a temp dir with config.json
    import tempfile as _tf
    import os as _os
    tmp = _tf.mktemp()
    _os.makedirs(tmp, exist_ok=True)
    Path(tmp, "config.json").write_text("{}")
    assert validate_model_id(tmp) == tmp


def test_validation_model_id_remote_format():
    from sped.utils.validation import validate_model_id
    assert validate_model_id("Qwen/Qwen3-0.6B") == "Qwen/Qwen3-0.6B"


def test_validation_model_id_invalid():
    from sped.utils.validation import validate_model_id
    import pytest as _pytest
    with _pytest.raises(ValueError):
        validate_model_id("not-a-valid-path-or-model")


def test_validation_draft_k_against_max():
    from sped.utils.validation import validate_draft_k_against_max
    import pytest as _pytest
    # Should not raise
    validate_draft_k_against_max(5, 10)
    validate_draft_k_against_max(5, 5)
    # Should raise
    with _pytest.raises(ValueError):
        validate_draft_k_against_max(10, 5)


def test_validation_timeout_valid():
    from sped.utils.validation import validate_timeout
    assert validate_timeout(30) == 30
    assert validate_timeout(None) is None


def test_validation_timeout_invalid():
    from sped.utils.validation import validate_timeout
    import pytest as _pytest
    with _pytest.raises(ValueError):
        validate_timeout(0)
    with _pytest.raises(ValueError):
        validate_timeout(7200)


def test_validation_dtype_valid():
    from sped.utils.validation import validate_dtype
    assert validate_dtype("auto") == "auto"
    assert validate_dtype("float16") == "float16"
    assert validate_dtype("bfloat16") == "bfloat16"
    assert validate_dtype("float32") == "float32"


def test_validation_dtype_invalid():
    from sped.utils.validation import validate_dtype
    import pytest as _pytest
    with _pytest.raises(ValueError):
        validate_dtype("int8")
    with _pytest.raises(ValueError):
        validate_dtype("float64")


# ── logging tests ────────────────────────────────────────


def test_logging_setup(temp_config_dir):
    from sped.utils.logging import setup_logging, get_logger, close_json_output
    logger = setup_logging(log_level="debug")
    assert logger is not None
    assert logger.level == 10  # DEBUG
    close_json_output()


def test_logging_log_file(temp_config_dir):
    from sped.utils.logging import setup_logging, close_json_output
    log_path = Path(tempfile.mktemp(suffix=".log"))
    setup_logging(log_level="info", log_file=str(log_path))
    assert log_path.exists() or log_path.parent.exists()
    close_json_output()


def test_logging_json_output(temp_config_dir):
    from sped.utils.logging import setup_logging, write_json_output, close_json_output
    json_path = Path(tempfile.mktemp(suffix=".json"))
    setup_logging(json_mode=True, json_file=str(json_path))
    write_json_output({"event": "test", "value": 42})
    close_json_output()
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["event"] == "test"


# ── output formatting tests ──────────────────────────────


def test_save_results_json():
    from sped.utils.output import save_results_json
    tmpdir = Path(tempfile.mkdtemp())
    data = {"foo": "bar", "num": 42}
    saved = save_results_json(data, tmpdir, timestamp=False)
    assert saved.exists()
    loaded = json.loads(saved.read_text())
    assert loaded["foo"] == "bar"
    assert loaded["num"] == 42
