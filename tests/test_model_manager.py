"""Unit tests for model_manager.py — pytest."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import model_manager


class TestModelsDict:
    def test_expected_labels(self):
        labels = list(model_manager.MODELS.keys())
        assert "mlx-whisper tiny" in labels
        assert "mlx-whisper small" in labels
        assert "mlx-whisper medium" in labels
        assert "mlx-whisper large-v3" in labels
        assert "mlx-whisper large-v3-turbo" in labels
        assert "mlx-whisper distil-large-v3" in labels
        assert "mlx-whisper medical-v1" in labels

    def test_expected_model_ids(self):
        values = list(model_manager.MODELS.values())
        assert ("mlx", "mlx-community/whisper-tiny") in values
        assert ("mlx", "mlx-community/whisper-small-mlx") in values
        assert ("mlx", "mlx-community/whisper-medium-mlx") in values
        assert ("mlx", "mlx-community/whisper-large-v3-mlx") in values
        assert ("mlx", "mlx-community/whisper-large-v3-turbo") in values
        assert ("mlx", "mlx-community/distil-whisper-large-v3") in values
        assert ("mlx", "Crystalcareai/Whisper-Medicalv1") in values

    def test_seven_models(self):
        assert len(model_manager.MODELS) == 7

    def test_sizes_list_matches_models(self):
        assert model_manager.SIZES == list(model_manager.MODELS.keys())

    def test_default_size_exists(self):
        assert model_manager.DEFAULT_SIZE in model_manager.MODELS

    def test_default_size_is_mlx_small(self):
        assert model_manager.DEFAULT_SIZE == "mlx-whisper small"

    def test_model_values_are_tuples(self):
        for key, val in model_manager.MODELS.items():
            assert isinstance(val, tuple), f"{key} value is not a tuple"
            assert len(val) == 2
            assert val[0] == "mlx"
            assert len(val[1]) > 0


class TestGetModelInfo:
    def test_label_to_model_info(self):
        assert model_manager.get_model_info("mlx-whisper tiny") == ("mlx", "mlx-community/whisper-tiny")
        assert model_manager.get_model_info("mlx-whisper small") == ("mlx", "mlx-community/whisper-small-mlx")
        assert model_manager.get_model_info("mlx-whisper medium") == ("mlx", "mlx-community/whisper-medium-mlx")
        assert model_manager.get_model_info("mlx-whisper large-v3") == ("mlx", "mlx-community/whisper-large-v3-mlx")
        assert model_manager.get_model_info("mlx-whisper large-v3-turbo") == ("mlx", "mlx-community/whisper-large-v3-turbo")
        assert model_manager.get_model_info("mlx-whisper distil-large-v3") == ("mlx", "mlx-community/distil-whisper-large-v3")
        assert model_manager.get_model_info("mlx-whisper medical-v1") == ("mlx", "Crystalcareai/Whisper-Medicalv1")

    def test_unknown_label_returns_mlx_fallback(self):
        assert model_manager.get_model_info("unknown-model") == ("mlx", "unknown-model")


class TestIsModelCached:
    def test_nonexistent_cache_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, "HF_CACHE", str(tmp_path / "nonexistent"))
        assert model_manager.is_model_cached("mlx-whisper small") is False

    def test_empty_cache_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, "HF_CACHE", str(tmp_path))
        assert model_manager.is_model_cached("mlx-whisper small") is False

    def test_model_dir_without_snapshots_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, "HF_CACHE", str(tmp_path))
        model_dir = tmp_path / "models--mlx-community--whisper-small-mlx"
        model_dir.mkdir()
        assert model_manager.is_model_cached("mlx-whisper small") is False

    def test_model_dir_with_empty_snapshots_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, "HF_CACHE", str(tmp_path))
        model_dir = tmp_path / "models--mlx-community--whisper-small-mlx"
        model_dir.mkdir()
        (model_dir / "snapshots").mkdir()
        assert model_manager.is_model_cached("mlx-whisper small") is False

    def test_model_dir_with_snapshots_returns_true(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, "HF_CACHE", str(tmp_path))
        model_dir = tmp_path / "models--mlx-community--whisper-small-mlx"
        model_dir.mkdir()
        snapshots = model_dir / "snapshots"
        snapshots.mkdir()
        (snapshots / "abc123").mkdir()
        assert model_manager.is_model_cached("mlx-whisper small") is True

    def test_large_v3_cache_check(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, "HF_CACHE", str(tmp_path))
        model_dir = tmp_path / "models--mlx-community--whisper-large-v3-mlx"
        model_dir.mkdir()
        snapshots = model_dir / "snapshots"
        snapshots.mkdir()
        (snapshots / "def456").mkdir()
        assert model_manager.is_model_cached("mlx-whisper large-v3") is True

    def test_all_models_cached_check(self, tmp_path, monkeypatch):
        """Verify cache check works for every model label."""
        monkeypatch.setattr(model_manager, "HF_CACHE", str(tmp_path))
        for label, (backend, model_id) in model_manager.MODELS.items():
            expected_dir = f"models--{model_id.replace('/', '--')}"
            model_dir = tmp_path / expected_dir
            model_dir.mkdir(exist_ok=True)
            snapshots = model_dir / "snapshots"
            snapshots.mkdir(exist_ok=True)
            (snapshots / "snap1").mkdir(exist_ok=True)
            assert model_manager.is_model_cached(label) is True, f"Cache check failed for {label}"
