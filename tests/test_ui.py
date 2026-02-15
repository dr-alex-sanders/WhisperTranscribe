"""Tkinter event simulation tests for WhisperTranscribe UI."""

import os
import sys
import tkinter as tk
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def app():
    """Create the app with mocked audio devices."""
    with patch("app.sd") as mock_sd:
        mock_sd.query_devices.return_value = [
            {"name": "Test Mic", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000.0},
            {"name": "Test Speaker", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 16000.0},
        ]
        mock_sd.default.device = [0, 1]
        mock_sd.RawInputStream = MagicMock()

        root = tk.Tk()
        root.withdraw()  # hide window during tests

        from app import WhisperTranscribe
        wt = WhisperTranscribe(root)
        yield wt

        root.destroy()


class TestWindowSetup:
    def test_window_title(self, app):
        assert app.root.title() == "WhisperTranscribe"

    def test_window_geometry(self, app):
        assert app.root.minsize() is not None

    def test_initial_state_not_recording(self, app):
        assert app.recording is False
        assert app.paused is False
        assert app.playing is False


class TestModelDropdown:
    def test_default_is_mlx_small(self, app):
        assert app.size_var.get() == "mlx-whisper small"

    def test_dropdown_has_seven_models(self, app):
        from model_manager import SIZES
        assert len(SIZES) == 7

    def test_dropdown_labels_are_mlx(self, app):
        from model_manager import SIZES
        for label in SIZES:
            assert label.startswith("mlx-whisper")


class TestModelSwitching:
    """Test that changing model dropdown updates _loaded_key and triggers reload."""

    def test_switch_to_tiny(self, app):
        app.size_var.set("mlx-whisper tiny")
        assert app.size_var.get() == "mlx-whisper tiny"

    def test_switch_to_medium(self, app):
        app.size_var.set("mlx-whisper medium")
        assert app.size_var.get() == "mlx-whisper medium"

    def test_switch_to_large(self, app):
        app.size_var.set("mlx-whisper large-v3")
        assert app.size_var.get() == "mlx-whisper large-v3"

    def test_switch_triggers_model_reload(self, app):
        """Switching model should require reload — _loaded_key won't match."""
        mock_model = MagicMock()
        with patch.object(app, "_begin_audio_capture"):
            app._on_model_loaded("mlx-whisper small", "mlx", mock_model, None)
        assert app._loaded_key == "mlx-whisper small"

        # Now switch to medium — key no longer matches
        app.size_var.set("mlx-whisper medium")
        assert app._loaded_key != app.size_var.get()

    def test_same_model_no_reload(self, app):
        """Same model selected — should not need reload."""
        mock_model = MagicMock()
        with patch.object(app, "_begin_audio_capture"):
            app._on_model_loaded("mlx-whisper small", "mlx", mock_model, None)

        app.size_var.set("mlx-whisper small")
        assert app._loaded_key == app.size_var.get()

    def test_model_load_passes_correct_id(self, app):
        """Verify that get_model_info resolves label to (backend, model_id)."""
        import model_manager
        label = "mlx-whisper medium"
        backend, model_id = model_manager.get_model_info(label)
        assert backend == "mlx"
        assert model_id == "mlx-community/whisper-medium-mlx"

    def test_all_models_resolve_to_mlx(self, app):
        """Every dropdown label should map to an MLX backend."""
        import model_manager
        for label in model_manager.SIZES:
            backend, model_id = model_manager.get_model_info(label)
            assert backend == "mlx", f"{label} resolved to unexpected backend: {backend}"

    def test_loaded_key_updates_on_each_switch(self, app):
        """Loading different models updates _loaded_key each time."""
        import model_manager
        mock_model = MagicMock()
        for label in model_manager.SIZES:
            with patch.object(app, "_begin_audio_capture"):
                app._on_model_loaded(label, "mlx", mock_model, None)
            assert app._loaded_key == label
            assert app.model is mock_model


class TestPromptField:
    def test_prompt_field_exists(self, app):
        assert hasattr(app, "prompt_var")
        assert hasattr(app, "prompt_entry")

    def test_prompt_initially_empty(self, app):
        assert app.prompt_var.get() == ""

    def test_prompt_can_be_set(self, app):
        app.prompt_var.set("cardiology, echocardiogram, myocardial infarction")
        assert "cardiology" in app.prompt_var.get()

    def test_prompt_entry_is_editable(self, app):
        app.prompt_entry.insert(0, "test medical terms")
        assert app.prompt_var.get() == "test medical terms"


class TestButtonStates:
    def test_initial_record_enabled(self, app):
        assert app.start_btn["state"] != "disabled"

    def test_initial_pause_disabled(self, app):
        assert str(app.pause_btn["state"]) == "disabled"

    def test_initial_stop_disabled(self, app):
        assert str(app.stop_btn["state"]) == "disabled"

    def test_recording_ui_state(self, app):
        app._set_recording_ui()
        assert str(app.start_btn["state"]) == "disabled"
        assert str(app.pause_btn["state"]) == "normal"
        assert str(app.stop_btn["state"]) == "normal"

    def test_paused_ui_state(self, app):
        app._set_paused_ui()
        assert str(app.start_btn["state"]) == "disabled"
        assert str(app.pause_btn["state"]) == "normal"
        assert app.pause_btn["text"] == "Resume"

    def test_stopped_ui_state(self, app):
        app._set_stopped_ui()
        assert str(app.start_btn["state"]) == "normal"
        assert str(app.pause_btn["state"]) == "disabled"
        assert str(app.stop_btn["state"]) == "disabled"

    def test_loading_ui_state(self, app):
        app._set_loading_ui()
        assert str(app.start_btn["state"]) == "disabled"
        assert str(app.pause_btn["state"]) == "disabled"
        assert str(app.stop_btn["state"]) == "disabled"


class TestTextDisplay:
    def test_append_final_adds_text(self, app):
        app._append_final("Hello world")
        content = app.text.get("1.0", tk.END).strip()
        assert content == "Hello world"

    def test_append_final_separates_with_space(self, app):
        app._append_final("Hello")
        app._append_final("world")
        content = app.text.get("1.0", tk.END).strip()
        assert content == "Hello world"

    def test_show_partial_adds_gray_text(self, app):
        app._show_partial("...")
        ranges = app.text.tag_ranges("partial")
        assert len(ranges) > 0

    def test_remove_partial_clears_partial(self, app):
        app._show_partial("...")
        app._remove_partial()
        ranges = app.text.tag_ranges("partial")
        assert len(ranges) == 0

    def test_append_final_removes_partial_first(self, app):
        app._show_partial("...")
        app._append_final("final text")
        ranges = app.text.tag_ranges("partial")
        assert len(ranges) == 0
        content = app.text.get("1.0", tk.END).strip()
        assert "final text" in content
        assert "..." not in content


class TestStatusLabel:
    def test_status_update(self, app):
        app._update_status("Testing status")
        assert app.status_label["text"] == "Testing status"

    def test_initial_status(self, app):
        assert "model" in app.status_label["text"].lower() or "select" in app.status_label["text"].lower()


class TestDeviceManagement:
    def test_mic_combo_exists(self, app):
        assert hasattr(app, "mic_combo")

    def test_selected_device_set(self, app):
        app.root.update()
        assert hasattr(app, "selected_device")


class TestRecordingsList:
    def test_recordings_listbox_exists(self, app):
        assert hasattr(app, "rec_listbox")

    def test_play_button_exists(self, app):
        assert hasattr(app, "play_btn")

    def test_delete_button_exists(self, app):
        assert hasattr(app, "delete_btn")

    def test_no_selection_returns_none(self, app):
        assert app._get_selected_recording_path() is None


class TestModelLoading:
    def test_on_model_loaded_success(self, app):
        mock_model = MagicMock()
        with patch.object(app, "_begin_audio_capture"):
            app._on_model_loaded("mlx-whisper small", "mlx", mock_model, None)
        assert app.model is mock_model
        assert app._loaded_key == "mlx-whisper small"

    def test_on_model_loaded_error(self, app):
        with patch("app.messagebox") as mock_mb:
            app._on_model_loaded("mlx-whisper small", None, None, RuntimeError("test error"))
        assert app.model is None
        assert str(app.status_label["text"]).startswith("Failed")


class TestLevelMeter:
    def test_level_canvas_exists(self, app):
        assert hasattr(app, "level_canvas_0")

    def test_draw_level_no_crash(self, app):
        app.mic_level = [0.0, 0.0]
        app._draw_level()
        app.mic_level = [0.5, 0.5]
        app._draw_level()
        app.mic_level = [1.0, 1.0]
        app._draw_level()

    def test_level_colors(self, app):
        app.root.update()

        app.mic_level = [0.3, 0.3]
        app._draw_level()
        color = app.level_canvas_0.itemcget(app.level_rect_0, "fill")
        assert color == "#4CAF50"  # green

        app.mic_level = [0.6, 0.6]
        app._draw_level()
        color = app.level_canvas_0.itemcget(app.level_rect_0, "fill")
        assert color == "#FFC107"  # yellow

        app.mic_level = [0.9, 0.9]
        app._draw_level()
        color = app.level_canvas_0.itemcget(app.level_rect_0, "fill")
        assert color == "#F44336"  # red
