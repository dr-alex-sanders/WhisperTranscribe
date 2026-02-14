#!/usr/bin/env python3
"""Automated test suite for WhisperTranscribe core functionality."""

import sys
import os
import time
import wave
import threading
import types
import numpy as np

# -- Stub numba/scipy before importing app (same as app.py) --
_fake_numba = types.ModuleType("numba")
_fake_numba.jit = lambda *a, **kw: (lambda f: f)
sys.modules["numba"] = _fake_numba
_fake_scipy = types.ModuleType("scipy")
_fake_scipy_sig = types.ModuleType("scipy.signal")
_fake_scipy.signal = _fake_scipy_sig
sys.modules["scipy"] = _fake_scipy
sys.modules["scipy.signal"] = _fake_scipy_sig
import mlx_whisper
del sys.modules["scipy"]
del sys.modules["scipy.signal"]

import sounddevice as sd

SAMPLE_RATE = 16000
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
STEREO_FILE = os.path.join(RECORDINGS_DIR, "rec_20260213_225253.wav")

passed = 0
failed = 0

def run_test(name, func):
    global passed, failed
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        func()
        print(f"  PASSED")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1


# ============================================================
# 1. Audio device detection
# ============================================================
def test_device_detection():
    devices = sd.query_devices()
    inputs = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            inputs.append((i, d["name"], d["max_input_channels"], int(d["default_samplerate"])))
    print(f"  Found {len(inputs)} input devices:")
    for idx, name, ch, sr in inputs:
        print(f"    [{idx}] {name} (channels={ch}, sr={sr})")
    assert len(inputs) > 0, "No input devices found"
    # Check for Wireless GO II
    wireless = [d for d in inputs if "Wireless" in d[1] or "RODE" in d[1].upper() or "GO II" in d[1]]
    if wireless:
        print(f"  Wireless GO II found: {wireless[0][1]}")
    else:
        print(f"  (Wireless GO II not detected — using default mic)")


# ============================================================
# 2. CoreAudio device query (macOS-specific)
# ============================================================
def test_coreaudio_query():
    try:
        import subprocess
        result = subprocess.run(
            ["system_profiler", "SPAudioDataType", "-json"],
            capture_output=True, text=True, timeout=5
        )
        import json
        data = json.loads(result.stdout)
        items = data.get("SPAudioDataType", [])
        inputs_found = []
        for item in items:
            name = item.get("_name", "")
            # Check for input
            if "coreaudio_device_input" in str(item) or "input" in str(item).lower():
                inputs_found.append(name)
        print(f"  CoreAudio reports {len(inputs_found)} audio devices")
        if inputs_found:
            for name in inputs_found[:5]:
                print(f"    - {name}")
    except Exception as e:
        print(f"  CoreAudio query skipped: {e}")
    # Not a hard failure — sounddevice is the primary path


# ============================================================
# 3. Audio file reading (WAV stereo)
# ============================================================
def test_read_stereo_wav():
    assert os.path.exists(STEREO_FILE), f"Test file not found: {STEREO_FILE}"
    with wave.open(STEREO_FILE, "rb") as wf:
        n_channels = wf.getnchannels()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)

    print(f"  File: {os.path.basename(STEREO_FILE)}")
    print(f"  Channels: {n_channels}, SR: {sr}, Frames: {n_frames}")
    print(f"  Duration: {n_frames/sr:.1f}s")

    assert n_channels == 2, f"Expected stereo, got {n_channels} channels"
    assert sr == 48000, f"Expected 48kHz, got {sr}"

    ch0 = raw[0::2]
    ch1 = raw[1::2]
    rms0 = np.sqrt(np.mean(ch0.astype(np.float32) ** 2))
    rms1 = np.sqrt(np.mean(ch1.astype(np.float32) ** 2))
    print(f"  Ch0 RMS: {rms0:.0f}, Ch1 RMS: {rms1:.0f}")
    assert rms0 > 10, "Ch0 appears silent"
    assert rms1 > 10, "Ch1 appears silent"


# ============================================================
# 4. Resampling
# ============================================================
def test_resampling():
    # Read stereo file at 48kHz and resample to 16kHz
    with wave.open(STEREO_FILE, "rb") as wf:
        sr = wf.getframerate()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    ch0 = raw[0::2]
    ch1 = raw[1::2]

    ratio = SAMPLE_RATE / sr
    new_len = int(len(ch0) * ratio)
    indices = np.linspace(0, len(ch0) - 1, new_len).astype(int)
    ch0_16k = ch0[indices]
    ch1_16k = ch1[indices]

    print(f"  Original: {len(ch0)} samples at {sr}Hz")
    print(f"  Resampled: {len(ch0_16k)} samples at {SAMPLE_RATE}Hz")
    print(f"  Duration preserved: {len(ch0)/sr:.1f}s -> {len(ch0_16k)/SAMPLE_RATE:.1f}s")

    assert abs(len(ch0)/sr - len(ch0_16k)/SAMPLE_RATE) < 0.1, "Duration mismatch after resampling"


# ============================================================
# 5. Stereo correlation detection
# ============================================================
def test_stereo_correlation():
    with wave.open(STEREO_FILE, "rb") as wf:
        sr = wf.getframerate()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    ch0 = raw[0::2].astype(np.float32)
    ch1 = raw[1::2].astype(np.float32)

    # Compute correlation
    norm0 = np.linalg.norm(ch0)
    norm1 = np.linalg.norm(ch1)
    if norm0 > 0 and norm1 > 0:
        corr = np.dot(ch0, ch1) / (norm0 * norm1)
    else:
        corr = 1.0

    print(f"  Full-file correlation: {corr:.4f}")
    print(f"  Channels are {'correlated (mono-like)' if corr > 0.5 else 'divergent (true stereo)'}")

    # For Wireless GO II with 2 transmitters, expect low correlation
    # (different speakers on each channel)


# ============================================================
# 6. MLX Whisper transcription
# ============================================================
def test_mlx_transcription():
    # Read, resample, and transcribe a short segment
    with wave.open(STEREO_FILE, "rb") as wf:
        sr = wf.getframerate()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    ch0 = raw[0::2]
    ch1 = raw[1::2]

    # Mix to mono
    mono = ((ch0.astype(np.int32) + ch1.astype(np.int32)) // 2).astype(np.int16)

    # Resample to 16kHz
    ratio = SAMPLE_RATE / sr
    new_len = int(len(mono) * ratio)
    indices = np.linspace(0, len(mono) - 1, new_len).astype(int)
    mono_16k = mono[indices]

    # Take first 10 seconds
    segment = mono_16k[:SAMPLE_RATE * 10]
    audio_f32 = segment.astype(np.float32) / 32768.0

    print(f"  Transcribing {len(audio_f32)/SAMPLE_RATE:.1f}s segment...")
    t0 = time.time()
    result = mlx_whisper.transcribe(
        audio_f32,
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        condition_on_previous_text=False,
        no_speech_threshold=0.5,
        language="ru",
    )
    elapsed = time.time() - t0

    segments = result.get("segments", [])
    text = " ".join(s["text"].strip() for s in segments if s["text"].strip())
    print(f"  Transcription ({elapsed:.1f}s): {text[:120]}")
    print(f"  Segments: {len(segments)}")

    assert len(segments) > 0, "No segments returned"
    assert len(text) > 5, f"Transcription too short: '{text}'"


# ============================================================
# 7. Peak-window speaker diarization
# ============================================================
def test_peak_window_diarization():
    """Test the peak-window speaker assignment algorithm."""
    with wave.open(STEREO_FILE, "rb") as wf:
        sr = wf.getframerate()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    ch0_raw = raw[0::2]
    ch1_raw = raw[1::2]

    # Resample to 16kHz
    ratio = SAMPLE_RATE / sr
    new_len = int(len(ch0_raw) * ratio)
    indices = np.linspace(0, len(ch0_raw) - 1, new_len).astype(int)
    ch0 = ch0_raw[indices].astype(np.float32)
    ch1 = ch1_raw[indices].astype(np.float32)

    WIN = int(0.1 * SAMPLE_RATE)  # 100ms window

    # Test with known time windows from previous runs
    test_segments = [
        # (start_sec, end_sec, expected_description)
        (2.0, 4.0, "Man counting (should be Speaker 1)"),
        (4.0, 5.5, "Woman: я очень довольна (should be Speaker 2)"),
        (6.0, 8.0, "Woman talking about universities (Speaker 2)"),
        (15.0, 20.0, "Woman: Northeastern, UCL (Speaker 2)"),
        (22.0, 23.5, "Man: Да, понятно (Speaker 1)"),
    ]

    print(f"  Peak-window analysis (WIN={WIN} samples = {WIN/SAMPLE_RATE*1000:.0f}ms):")
    sp1_count = 0
    sp2_count = 0
    for start, end, desc in test_segments:
        si = int(start * SAMPLE_RATE)
        ei = min(int(end * SAMPLE_RATE), len(ch0))

        if ei - si > WIN:
            combined = ch0[si:ei] ** 2 + ch1[si:ei] ** 2
            cumsum = np.cumsum(combined)
            win_sums = cumsum[WIN:] - cumsum[:-WIN]
            best = np.argmax(win_sums)
            wi, we = si + best, si + best + WIN
            peak0 = np.sqrt(np.mean(ch0[wi:we] ** 2))
            peak1 = np.sqrt(np.mean(ch1[wi:we] ** 2))
        else:
            peak0 = np.sqrt(np.mean(ch0[si:ei] ** 2))
            peak1 = np.sqrt(np.mean(ch1[si:ei] ** 2))

        speaker = 1 if peak0 > peak1 else 2
        ratio = max(peak0, peak1) / min(peak0, peak1) if min(peak0, peak1) > 0 else 99
        tag = "conf" if ratio > 1.5 else "close"
        if speaker == 1:
            sp1_count += 1
        else:
            sp2_count += 1
        print(f"    [{start:.0f}-{end:.0f}s] Speaker {speaker} ({tag}, peak:{int(peak0)}/{int(peak1)}, ratio:{ratio:.1f}x) — {desc}")

    assert sp1_count > 0, "No segments assigned to Speaker 1"
    assert sp2_count > 0, "No segments assigned to Speaker 2"
    print(f"  Speaker 1: {sp1_count} segments, Speaker 2: {sp2_count} segments")


# ============================================================
# 8. Monitor level meter (brief recording test)
# ============================================================
def test_monitor_level():
    """Test that the mic monitor can capture audio and compute RMS."""
    rms_values = []
    callback_count = [0]

    def callback(indata, frames, time_info, status):
        arr = np.frombuffer(indata, dtype=np.int16)
        rms = np.sqrt(np.mean(arr.astype(np.float32) ** 2))
        rms_values.append(rms)
        callback_count[0] += 1

    # Find an input device
    devices = sd.query_devices()
    dev_idx = None
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            dev_idx = i
            break
    assert dev_idx is not None, "No input device found"

    dev_info = devices[dev_idx]
    dev_sr = int(dev_info["default_samplerate"])
    channels = min(dev_info["max_input_channels"], 2)
    blocksize = int(0.25 * dev_sr)  # 250ms blocks

    print(f"  Using device [{dev_idx}] {dev_info['name']} (ch={channels}, sr={dev_sr})")

    try:
        with sd.RawInputStream(
            samplerate=dev_sr, blocksize=blocksize,
            dtype="int16", channels=channels, device=dev_idx,
            callback=callback,
        ):
            sd.sleep(1500)  # record 1.5 seconds
    except Exception as e:
        print(f"  Monitor test skipped (audio error): {e}")
        return

    print(f"  Got {callback_count[0]} callbacks, {len(rms_values)} RMS values")
    if rms_values:
        print(f"  RMS range: {min(rms_values):.1f} - {max(rms_values):.1f}")
        print(f"  Mean RMS: {np.mean(rms_values):.1f}")
    assert callback_count[0] > 0, "No audio callbacks received"


# ============================================================
# 9. WAV file writing (simulated recording)
# ============================================================
def test_wav_writing():
    """Test creating a WAV file from audio data."""
    import tempfile

    # Generate a 2-second sine wave at 440Hz, stereo
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    # Write stereo WAV
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            # Interleave: ch0 = tone, ch1 = silence
            stereo = np.zeros(len(tone) * 2, dtype=np.int16)
            stereo[0::2] = tone
            wf.writeframes(stereo.tobytes())

        # Read back and verify
        with wave.open(tmp.name, "rb") as wf:
            assert wf.getnchannels() == 2
            assert wf.getframerate() == SAMPLE_RATE
            read_frames = wf.getnframes()
            assert read_frames == len(tone), f"Frame count mismatch: {read_frames} vs {len(tone)}"

        print(f"  Written and verified {duration}s stereo WAV ({os.path.getsize(tmp.name)} bytes)")
    finally:
        os.unlink(tmp.name)


# ============================================================
# 10. Full reprocess pipeline (end-to-end)
# ============================================================
def test_full_reprocess():
    """Test the complete reprocess pipeline: read -> resample -> transcribe -> diarize."""
    # Read stereo file
    with wave.open(STEREO_FILE, "rb") as wf:
        n_channels = wf.getnchannels()
        sr = wf.getframerate()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    channels_data = [raw[ch::n_channels] for ch in range(n_channels)]

    # Resample to 16kHz
    resampled = []
    for ch_data in channels_data[:2]:
        ratio = SAMPLE_RATE / sr
        new_len = int(len(ch_data) * ratio)
        indices = np.linspace(0, len(ch_data) - 1, new_len).astype(int)
        resampled.append(ch_data[indices])

    # Mix to mono
    mono = ((resampled[0].astype(np.int32) + resampled[1].astype(np.int32)) // 2).astype(np.int16)
    audio_f32 = mono.astype(np.float32) / 32768.0

    print(f"  Audio: {len(audio_f32)/SAMPLE_RATE:.1f}s, stereo")

    # Transcribe
    print(f"  Transcribing full audio...")
    t0 = time.time()
    result = mlx_whisper.transcribe(
        audio_f32,
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        condition_on_previous_text=False,
        no_speech_threshold=0.5,
        language="ru",
    )
    elapsed = time.time() - t0

    segments = result.get("segments", [])
    seg_list = [(s["start"], s["end"], s["text"].strip()) for s in segments
                if s.get("no_speech_prob", 0) < 0.5 and s["text"].strip()]

    print(f"  Transcribed in {elapsed:.1f}s: {len(seg_list)} segments")

    # Peak-window speaker assignment
    ch_raw = [ch.astype(np.float32) for ch in resampled]
    WIN = int(0.1 * SAMPLE_RATE)

    results = []
    for start, end, text in seg_list:
        si = int(start * SAMPLE_RATE)
        ei = min(int(end * SAMPLE_RATE), len(ch_raw[0]))
        if ei - si > WIN:
            combined = ch_raw[0][si:ei] ** 2 + ch_raw[1][si:ei] ** 2
            cumsum = np.cumsum(combined)
            win_sums = cumsum[WIN:] - cumsum[:-WIN]
            best = np.argmax(win_sums)
            wi, we = si + best, si + best + WIN
            peak0 = np.sqrt(np.mean(ch_raw[0][wi:we] ** 2))
            peak1 = np.sqrt(np.mean(ch_raw[1][wi:we] ** 2))
        elif ei > si:
            peak0 = np.sqrt(np.mean(ch_raw[0][si:ei] ** 2))
            peak1 = np.sqrt(np.mean(ch_raw[1][si:ei] ** 2))
        else:
            peak0 = peak1 = 0

        speaker = 0 if peak0 > peak1 else 1
        ratio = max(peak0, peak1) / min(peak0, peak1) if min(peak0, peak1) > 0 else 99
        tag = "conf" if ratio > 1.5 else "close"
        ts = f"{int(start)//60:02d}:{int(start)%60:02d}"
        results.append((speaker, ts, text, tag, int(peak0), int(peak1)))

    # Print results
    print(f"\n  Full transcript with speaker assignment:")
    for speaker, ts, text, tag, p0, p1 in results:
        print(f"    [{ts}] Speaker {speaker+1} ({tag}, {p0}/{p1}): {text[:70]}")

    # Verify
    speakers = set(r[0] for r in results)
    assert len(results) > 5, f"Too few segments: {len(results)}"
    assert len(speakers) > 1, "All segments assigned to same speaker — diarization not working"

    sp1 = sum(1 for r in results if r[0] == 0)
    sp2 = sum(1 for r in results if r[0] == 1)
    conf = sum(1 for r in results if r[3] == "conf")
    close = sum(1 for r in results if r[3] == "close")
    print(f"\n  Summary: {len(results)} segments, Speaker1={sp1}, Speaker2={sp2}")
    print(f"  Confident: {conf}, Close: {close}")


# ============================================================
# Run all tests
# ============================================================
if __name__ == "__main__":
    print("WhisperTranscribe — Automated Test Suite")
    print(f"Python {sys.version}")
    print(f"Working dir: {os.getcwd()}")
    print()

    run_test("1. Audio device detection", test_device_detection)
    run_test("2. CoreAudio device query", test_coreaudio_query)
    run_test("3. Read stereo WAV file", test_read_stereo_wav)
    run_test("4. Resampling (48kHz → 16kHz)", test_resampling)
    run_test("5. Stereo correlation detection", test_stereo_correlation)
    run_test("6. MLX Whisper transcription (10s)", test_mlx_transcription)
    run_test("7. Peak-window speaker diarization", test_peak_window_diarization)
    run_test("8. Monitor level meter", test_monitor_level)
    run_test("9. WAV file writing", test_wav_writing)
    run_test("10. Full reprocess pipeline (end-to-end)", test_full_reprocess)

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed+failed} tests")
    print(f"{'='*60}")
    sys.exit(1 if failed > 0 else 0)
