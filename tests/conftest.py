"""Shared fixtures for all test suites."""

import sys
import os
import pytest

# Add project root to path so we can import app and model_manager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_recordings_dir(tmp_path):
    """Provide a temporary recordings directory."""
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir()
    return str(rec_dir)


@pytest.fixture
def sample_audio_bytes():
    """Generate 1 second of sine wave audio as int16 bytes (16kHz mono)."""
    import numpy as np
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 440 Hz sine wave at ~50% volume
    audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return audio.tobytes()


@pytest.fixture
def sample_audio_5s():
    """Generate 5 seconds of sine wave audio as int16 bytes (16kHz mono)."""
    import numpy as np
    duration = 5.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return audio.tobytes()
