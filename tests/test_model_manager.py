"""Unit tests for model_manager.py — pytest."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import model_manager


class TestModelsDict:
    def test_expected_labels(self):
        labels = list(model_manager.MODELS.keys())
        assert "tiny (75 MB)" in labels
        assert "base (150 MB)" in labels
        assert "small (500 MB)" in labels
        assert "medium (1.5 GB)" in labels
        assert "large-v3 (3 GB)" in labels

    def test_expected_model_ids(self):
        values = list(model_manager.MODELS.values())
        assert "tiny" in values
        assert "base" in values
        assert "small" in values
        assert "medium" in values
        assert "large-v3" in values

    def test_five_models(self):
        assert len(model_manager.MODELS) == 5

    def test_sizes_list_matches_models(self):
        assert model_manager.SIZES == list(model_manager.MODELS.keys())

    def test_default_size_exists(self):
        assert model_manager.DEFAULT_SIZE in model_manager.MODELS

    def test_default_size_is_small(self):
        assert model_manager.DEFAULT_SIZE == "small (500 MB)"

    def test_model_values_are_strings(self):
        for key, val in model_manager.MODELS.items():
            assert isinstance(val, str)
            assert len(val) > 0


class TestGetModelPath:
    def test_label_to_model_id(self):
        assert model_manager.get_model_path("tiny (75 MB)") == "tiny"
        assert model_manager.get_model_path("base (150 MB)") == "base"
        assert model_manager.get_model_path("small (500 MB)") == "small"
        assert model_manager.get_model_path("medium (1.5 GB)") == "medium"
        assert model_manager.get_model_path("large-v3 (3 GB)") == "large-v3"

    def test_unknown_label_returns_input(self):
        assert model_manager.get_model_path("unknown-model") == "unknown-model"


class TestIsModelCached:
    def test_nonexistent_cache_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, "HF_CACHE", str(tmp_path / "nonexistent"))
        assert model_manager.is_model_cached("small (500 MB)") is False

    def test_empty_cache_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, "HF_CACHE", str(tmp_path))
        assert model_manager.is_model_cached("small (500 MB)") is False

    def test_model_dir_without_snapshots_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, "HF_CACHE", str(tmp_path))
        model_dir = tmp_path / "models--Systran--faster-whisper-small"
        model_dir.mkdir()
        assert model_manager.is_model_cached("small (500 MB)") is False

    def test_model_dir_with_empty_snapshots_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, "HF_CACHE", str(tmp_path))
        model_dir = tmp_path / "models--Systran--faster-whisper-small"
        model_dir.mkdir()
        (model_dir / "snapshots").mkdir()
        assert model_manager.is_model_cached("small (500 MB)") is False

    def test_model_dir_with_snapshots_returns_true(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, "HF_CACHE", str(tmp_path))
        model_dir = tmp_path / "models--Systran--faster-whisper-small"
        model_dir.mkdir()
        snapshots = model_dir / "snapshots"
        snapshots.mkdir()
        (snapshots / "abc123").mkdir()
        assert model_manager.is_model_cached("small (500 MB)") is True

    def test_large_v3_cache_check(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, "HF_CACHE", str(tmp_path))
        model_dir = tmp_path / "models--Systran--faster-whisper-large-v3"
        model_dir.mkdir()
        snapshots = model_dir / "snapshots"
        snapshots.mkdir()
        (snapshots / "def456").mkdir()
        assert model_manager.is_model_cached("large-v3 (3 GB)") is True

    def test_all_models_cached_check(self, tmp_path, monkeypatch):
        """Verify cache check works for every model label."""
        monkeypatch.setattr(model_manager, "HF_CACHE", str(tmp_path))
        for label, model_id in model_manager.MODELS.items():
            model_dir = tmp_path / f"models--Systran--faster-whisper-{model_id}"
            model_dir.mkdir(exist_ok=True)
            snapshots = model_dir / "snapshots"
            snapshots.mkdir(exist_ok=True)
            (snapshots / "snap1").mkdir(exist_ok=True)
            assert model_manager.is_model_cached(label) is True, f"Cache check failed for {label}"
