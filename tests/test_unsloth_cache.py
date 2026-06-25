"""Tests for the shared 4-bit Unsloth model load caching utility (#68, #80)."""

import pytest
from pathlib import Path


class TestCacheDirFor:
    def test_basic_cache_name(self):
        from sped.utils.unsloth_cache import _cache_dir_for
        assert _cache_dir_for("/data/models/Qwen3-8B") == "/data/models/Qwen3-8B-4bit-cache"

    def test_trailing_slash_stripped(self):
        from sped.utils.unsloth_cache import _cache_dir_for
        assert _cache_dir_for("/data/models/Qwen3-8B/") == "/data/models/Qwen3-8B-4bit-cache"

    def test_seq_key_included(self):
        from sped.utils.unsloth_cache import _cache_dir_for
        result = _cache_dir_for("/data/models/Qwen3-8B", max_seq_length=4096)
        assert result == "/data/models/Qwen3-8B-4bit-cache-4096"

    def test_seq_key_none_excluded(self):
        from sped.utils.unsloth_cache import _cache_dir_for
        result = _cache_dir_for("/data/models/Qwen3-8B", max_seq_length=None)
        assert result == "/data/models/Qwen3-8B-4bit-cache"

    def test_hf_model_id(self):
        from sped.utils.unsloth_cache import _cache_dir_for
        assert _cache_dir_for("Qwen/Qwen3-8B") == "Qwen/Qwen3-8B-4bit-cache"


class TestLoadUnslothModelImport:
    def test_import_from_utils(self):
        from sped.utils import load_unsloth_model
        assert callable(load_unsloth_model)

    def test_import_from_module(self):
        from sped.utils.unsloth_cache import load_unsloth_model
        assert callable(load_unsloth_model)


class TestLoadUnslothModelBehavior:
    def test_load_without_unsloth_raises_importerror(self):
        """If unsloth is not installed, loading should raise ImportError."""
        from sped.utils.unsloth_cache import load_unsloth_model
        try:
            import unsloth  # noqa: F401
            pytest.skip("unsloth is installed — cannot test ImportError path")
        except ImportError:
            with pytest.raises(ImportError):
                load_unsloth_model("dummy/model")