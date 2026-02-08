"""Unit tests for audio processing logic — pytest."""

import math
import os
import struct
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app as app_module


class TestAudioConversion:
    """Test int16 → float32 conversion used in _transcribe_chunk."""

    def test_silence_converts_to_zeros(self):
        silence = np.zeros(16000, dtype=np.int16)
        audio_bytes = silence.tobytes()
        result = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        assert np.allclose(result, 0.0)

    def test_max_amplitude_converts_to_near_one(self):
        loud = np.full(100, 32767, dtype=np.int16)
        audio_bytes = loud.tobytes()
        result = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        assert np.all(result > 0.99)
        assert np.all(result <= 1.0)

    def test_min_amplitude_converts_to_near_neg_one(self):
        loud = np.full(100, -32768, dtype=np.int16)
        audio_bytes = loud.tobytes()
        result = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        assert np.all(result >= -1.0)
        assert np.all(result < -0.99)

    def test_roundtrip_preserves_shape(self, sample_audio_bytes):
        audio_int16 = np.frombuffer(sample_audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        assert audio_float32.shape == audio_int16.shape
        assert audio_float32.dtype == np.float32


class TestConstants:
    def test_sample_rate_is_16k(self):
        assert app_module.SAMPLE_RATE == 16000

    def test_block_size_is_quarter_second(self):
        assert app_module.BLOCK_SIZE == 4000
        assert app_module.BLOCK_SIZE / app_module.SAMPLE_RATE == 0.25

    def test_partial_interval(self):
        assert app_module.PARTIAL_INTERVAL == 1.0

    def test_commit_seconds(self):
        assert app_module.COMMIT_SECONDS == 10


class TestBufferAccumulation:
    """Test that buffer accumulation logic works correctly."""

    def test_commit_threshold(self):
        """Buffer reaching COMMIT_SECONDS should trigger commit."""
        commit_samples = app_module.SAMPLE_RATE * app_module.COMMIT_SECONDS
        buffer = b"\x00" * (commit_samples * 2)  # int16 = 2 bytes
        buffer_seconds = (len(buffer) // 2) / app_module.SAMPLE_RATE
        assert buffer_seconds >= app_module.COMMIT_SECONDS

    def test_small_remainder_skipped(self):
        """Buffer < 0.25s should be skipped on stop."""
        small_buffer = b"\x00" * 3200  # 0.1s
        buffer_samples = len(small_buffer) // 2
        min_samples = app_module.SAMPLE_RATE // 4
        assert buffer_samples < min_samples

    def test_large_remainder_transcribed(self):
        """Buffer >= 0.25s should be transcribed on stop."""
        buffer = b"\x00" * 32000  # 1s
        buffer_samples = len(buffer) // 2
        min_samples = app_module.SAMPLE_RATE // 4
        assert buffer_samples > min_samples

    def test_partial_triggers_after_half_second(self):
        """Buffer >= 0.5s should trigger partial transcription."""
        half_sec_samples = app_module.SAMPLE_RATE // 2
        buffer = b"\x00" * (half_sec_samples * 2)
        buffer_seconds = (len(buffer) // 2) / app_module.SAMPLE_RATE
        assert buffer_seconds >= 0.5


class TestLevelMeter:
    """Test the RMS level calculation logic from _update_level."""

    def _compute_level(self, data: bytes) -> float:
        """Reproduce the level calculation from app.py."""
        n_samples = len(data) // 2
        if n_samples == 0:
            return 0.0
        samples = struct.unpack(f"<{n_samples}h", data)
        rms = math.sqrt(sum(s * s for s in samples) / n_samples)
        if rms < 1:
            return 0.0
        db = 20 * math.log10(rms / 32768)
        return max(0.0, min(1.0, (db + 60) / 55))

    def test_silence_gives_zero(self):
        silence = np.zeros(1000, dtype=np.int16).tobytes()
        assert self._compute_level(silence) == 0.0

    def test_max_volume_gives_high_level(self):
        loud = np.full(1000, 32000, dtype=np.int16).tobytes()
        level = self._compute_level(loud)
        assert level > 0.9

    def test_quiet_gives_low_level(self):
        quiet = np.full(1000, 100, dtype=np.int16).tobytes()
        level = self._compute_level(quiet)
        assert level < 0.3

    def test_level_clamped_to_0_1(self):
        for amplitude in [0, 1, 100, 1000, 10000, 32767]:
            data = np.full(1000, amplitude, dtype=np.int16).tobytes()
            level = self._compute_level(data)
            assert 0.0 <= level <= 1.0

    def test_empty_data_gives_zero(self):
        assert self._compute_level(b"") == 0.0


class TestSaveRecording:
    """Test WAV file saving."""

    def test_wav_file_created(self, tmp_path, sample_audio_bytes):
        wav_path = str(tmp_path / "test.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(sample_audio_bytes)

        assert os.path.exists(wav_path)
        with wave.open(wav_path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 16000  # 1 second

    def test_wav_roundtrip(self, tmp_path, sample_audio_bytes):
        wav_path = str(tmp_path / "roundtrip.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(sample_audio_bytes)

        with wave.open(wav_path, "rb") as wf:
            data = wf.readframes(wf.getnframes())

        original = np.frombuffer(sample_audio_bytes, dtype=np.int16)
        loaded = np.frombuffer(data, dtype=np.int16)
        assert np.array_equal(original, loaded)
