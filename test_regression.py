#!/usr/bin/env python3
"""
WhisperTranscribe — Comprehensive Regression & Performance Test Suite
=====================================================================
Tests every core subsystem and collects product metrics:
  - Response times (model load, transcription, reprocess)
  - Audio pipeline (capture, resample, stereo detection)
  - Speaker diarization accuracy
  - File I/O (read/write WAV, MP3, transcripts)
  - UI-less model lifecycle
  - Device detection & monitoring
  - Memory usage
"""

import gc
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import wave

import numpy as np

# ── Stub numba/scipy before importing mlx_whisper (Python 3.14 deadlock fix) ──
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
import model_manager

# ── Constants ──
SAMPLE_RATE = 16000
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
STEREO_FILE = os.path.join(RECORDINGS_DIR, "rec_20260213_225253.wav")

# ── Test harness ──
results = []
metrics = {}

def test(name, category="general"):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            t0 = time.time()
            mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            try:
                result = func()
                elapsed = time.time() - t0
                mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                mem_delta = (mem_after - mem_before) / 1024  # KB on macOS (bytes) → MB approx
                results.append({"name": name, "category": category, "status": "PASSED",
                               "time_ms": round(elapsed * 1000, 1), "mem_delta_kb": mem_delta})
                print(f"  PASSED ({elapsed*1000:.0f}ms)")
                return result
            except Exception as e:
                elapsed = time.time() - t0
                results.append({"name": name, "category": category, "status": "FAILED",
                               "time_ms": round(elapsed * 1000, 1), "error": str(e)})
                print(f"  FAILED ({elapsed*1000:.0f}ms): {e}")
                import traceback; traceback.print_exc()
        wrapper._test_name = name
        wrapper._test_category = category
        wrapper._test_func = func
        return wrapper
    return decorator


def run_test(func):
    print(f"\n{'─'*60}")
    print(f"TEST: {func._test_name} [{func._test_category}]")
    print(f"{'─'*60}")
    return func()


# ════════════════════════════════════════════════════════════════
# CATEGORY: Audio Device & Hardware
# ════════════════════════════════════════════════════════════════

@test("Device enumeration", "device")
def test_device_enum():
    # Reinitialize PortAudio to pick up recently-plugged USB devices
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        pass
    devices = sd.query_devices()
    inputs = [(i, d["name"], d["max_input_channels"], int(d["default_samplerate"]))
              for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    print(f"  Input devices: {len(inputs)}")
    for idx, name, ch, sr in inputs:
        print(f"    [{idx}] {name} (ch={ch}, sr={sr})")
    assert len(inputs) > 0, "No input devices"
    metrics["input_device_count"] = len(inputs)
    wireless = [d for d in inputs if "Wireless" in d[1] or "GO II" in d[1]]
    metrics["wireless_go_ii_detected"] = len(wireless) > 0
    if wireless:
        print(f"  Wireless GO II: [{wireless[0][0]}] {wireless[0][1]}")
    else:
        print(f"  WARNING: Wireless GO II not detected")


@test("CoreAudio query", "device")
def test_coreaudio():
    t0 = time.time()
    try:
        result = subprocess.run(
            ["system_profiler", "SPAudioDataType", "-json"],
            capture_output=True, text=True, timeout=5
        )
        data = json.loads(result.stdout)
        elapsed = time.time() - t0
        metrics["coreaudio_query_ms"] = round(elapsed * 1000, 1)
        print(f"  CoreAudio query: {elapsed*1000:.0f}ms")
        items = data.get("SPAudioDataType", [])
        print(f"  Audio items: {len(items)}")
    except subprocess.TimeoutExpired:
        print(f"  CoreAudio query timed out (>5s)")
        metrics["coreaudio_query_ms"] = 5000


@test("Monitor level meter (2s capture)", "device")
def test_monitor():
    rms_values = []
    callback_count = [0]

    def cb(indata, frames, time_info, status):
        arr = np.frombuffer(indata, dtype=np.int16)
        rms_values.append(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
        callback_count[0] += 1

    # Find best input device
    devices = sd.query_devices()
    dev_idx = None
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            if "Wireless" in d["name"] or "GO II" in d["name"]:
                dev_idx = i
                break
    if dev_idx is None:
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                dev_idx = i
                break

    dev = devices[dev_idx]
    ch = min(dev["max_input_channels"], 2)
    sr = int(dev["default_samplerate"])
    blocksize = int(0.1 * sr)  # 100ms blocks (matches app fix)

    print(f"  Device: [{dev_idx}] {dev['name']} (ch={ch}, sr={sr})")
    t0 = time.time()
    with sd.RawInputStream(samplerate=sr, blocksize=blocksize, dtype="int16",
                           channels=ch, device=dev_idx, callback=cb):
        sd.sleep(2000)
    elapsed = time.time() - t0

    metrics["monitor_callbacks_2s"] = callback_count[0]
    metrics["monitor_rms_mean"] = round(np.mean(rms_values), 1) if rms_values else 0
    metrics["monitor_rms_max"] = round(max(rms_values), 1) if rms_values else 0
    metrics["monitor_latency_ms"] = round(elapsed * 1000 / max(callback_count[0], 1), 1)

    print(f"  Callbacks: {callback_count[0]} in {elapsed:.1f}s")
    print(f"  RMS: mean={metrics['monitor_rms_mean']}, max={metrics['monitor_rms_max']}")
    print(f"  Callback interval: ~{metrics['monitor_latency_ms']:.0f}ms")
    assert callback_count[0] >= 4, f"Too few callbacks: {callback_count[0]}"


# ════════════════════════════════════════════════════════════════
# CATEGORY: File I/O
# ════════════════════════════════════════════════════════════════

@test("Read stereo WAV", "file_io")
def test_read_stereo():
    assert os.path.exists(STEREO_FILE), f"Missing: {STEREO_FILE}"
    t0 = time.time()
    with wave.open(STEREO_FILE, "rb") as wf:
        n_ch = wf.getnchannels()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
    read_ms = (time.time() - t0) * 1000

    ch0 = raw[0::2]; ch1 = raw[1::2]
    metrics["wav_read_ms"] = round(read_ms, 1)
    metrics["wav_channels"] = n_ch
    metrics["wav_samplerate"] = sr
    metrics["wav_duration_s"] = round(n_frames / sr, 1)
    metrics["wav_filesize_kb"] = round(os.path.getsize(STEREO_FILE) / 1024, 1)

    print(f"  Read: {read_ms:.1f}ms, {n_ch}ch, {sr}Hz, {n_frames/sr:.1f}s")
    print(f"  Size: {metrics['wav_filesize_kb']:.0f}KB")
    print(f"  Ch0 RMS: {np.sqrt(np.mean(ch0.astype(np.float32)**2)):.0f}")
    print(f"  Ch1 RMS: {np.sqrt(np.mean(ch1.astype(np.float32)**2)):.0f}")
    assert n_ch == 2
    assert sr == 48000


@test("Write + read WAV roundtrip", "file_io")
def test_wav_roundtrip():
    # Generate test audio: 3s stereo, ch0=440Hz tone, ch1=880Hz tone
    dur = 3.0
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    ch0 = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    ch1 = (np.sin(2 * np.pi * 880 * t) * 16000).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        # Write
        t0 = time.time()
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            stereo = np.empty(len(ch0) * 2, dtype=np.int16)
            stereo[0::2] = ch0; stereo[1::2] = ch1
            wf.writeframes(stereo.tobytes())
        write_ms = (time.time() - t0) * 1000

        # Read back
        t0 = time.time()
        with wave.open(tmp.name, "rb") as wf:
            assert wf.getnchannels() == 2
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.getnframes() == len(ch0)
            read_raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        read_ms = (time.time() - t0) * 1000

        # Verify data integrity
        read_ch0 = read_raw[0::2]
        read_ch1 = read_raw[1::2]
        assert np.array_equal(ch0, read_ch0), "Ch0 data mismatch"
        assert np.array_equal(ch1, read_ch1), "Ch1 data mismatch"

        fsize = os.path.getsize(tmp.name)
        metrics["wav_write_ms"] = round(write_ms, 1)
        metrics["wav_read_roundtrip_ms"] = round(read_ms, 1)
        print(f"  Write: {write_ms:.1f}ms, Read: {read_ms:.1f}ms, Size: {fsize} bytes")
        print(f"  Data integrity: OK (bit-perfect roundtrip)")
    finally:
        os.unlink(tmp.name)


@test("Transcript save/load", "file_io")
def test_transcript_io():
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
    transcript = "[00:00] Speaker 1: Раз, два, три.\n[00:02] Speaker 2: Я рада.\n"
    t0 = time.time()
    tmp.write(transcript)
    tmp.close()
    write_ms = (time.time() - t0) * 1000

    t0 = time.time()
    with open(tmp.name, "r") as f:
        content = f.read()
    read_ms = (time.time() - t0) * 1000

    assert content == transcript, "Transcript content mismatch"
    print(f"  Write: {write_ms:.1f}ms, Read: {read_ms:.1f}ms")
    print(f"  UTF-8 Cyrillic: OK")
    os.unlink(tmp.name)


@test("Recordings directory listing", "file_io")
def test_recordings_list():
    t0 = time.time()
    files = sorted([f for f in os.listdir(RECORDINGS_DIR)
                   if f.endswith((".wav", ".mp3"))], reverse=True)
    elapsed = (time.time() - t0) * 1000
    metrics["recordings_count"] = len(files)
    metrics["recordings_list_ms"] = round(elapsed, 1)
    print(f"  Found {len(files)} recordings in {elapsed:.1f}ms")
    for f in files[:5]:
        size = os.path.getsize(os.path.join(RECORDINGS_DIR, f))
        print(f"    {f} ({size/1024:.0f}KB)")


# ════════════════════════════════════════════════════════════════
# CATEGORY: Audio Processing
# ════════════════════════════════════════════════════════════════

@test("Resampling 48kHz → 16kHz", "audio")
def test_resample():
    with wave.open(STEREO_FILE, "rb") as wf:
        sr = wf.getframerate()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    ch0 = raw[0::2]

    t0 = time.time()
    ratio = SAMPLE_RATE / sr
    new_len = int(len(ch0) * ratio)
    indices = np.linspace(0, len(ch0) - 1, new_len).astype(int)
    resampled = ch0[indices]
    elapsed = (time.time() - t0) * 1000

    metrics["resample_48_to_16_ms"] = round(elapsed, 1)
    metrics["resample_input_samples"] = len(ch0)
    metrics["resample_output_samples"] = len(resampled)
    dur_orig = len(ch0) / sr
    dur_res = len(resampled) / SAMPLE_RATE

    print(f"  {len(ch0)} → {len(resampled)} samples in {elapsed:.1f}ms")
    print(f"  Duration: {dur_orig:.3f}s → {dur_res:.3f}s (drift: {abs(dur_orig-dur_res)*1000:.1f}ms)")
    assert abs(dur_orig - dur_res) < 0.01, "Duration drift >10ms"


@test("Stereo correlation analysis", "audio")
def test_correlation():
    with wave.open(STEREO_FILE, "rb") as wf:
        sr = wf.getframerate()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    ch0 = raw[0::2].astype(np.float32)
    ch1 = raw[1::2].astype(np.float32)

    t0 = time.time()
    # Full-file correlation
    norm0 = np.linalg.norm(ch0); norm1 = np.linalg.norm(ch1)
    full_corr = np.dot(ch0, ch1) / (norm0 * norm1) if norm0 > 0 and norm1 > 0 else 1.0

    # Windowed correlation (1-second windows)
    win_size = sr
    window_corrs = []
    for i in range(0, len(ch0) - win_size, win_size):
        w0 = ch0[i:i+win_size]; w1 = ch1[i:i+win_size]
        n0 = np.linalg.norm(w0); n1 = np.linalg.norm(w1)
        if n0 > 0 and n1 > 0:
            window_corrs.append(np.dot(w0, w1) / (n0 * n1))
    elapsed = (time.time() - t0) * 1000

    metrics["stereo_full_correlation"] = round(full_corr, 4)
    metrics["stereo_window_corr_mean"] = round(np.mean(window_corrs), 4)
    metrics["stereo_window_corr_std"] = round(np.std(window_corrs), 4)
    metrics["stereo_divergent_windows"] = sum(1 for c in window_corrs if c < 0.5)
    metrics["stereo_correlated_windows"] = sum(1 for c in window_corrs if c >= 0.5)
    metrics["correlation_analysis_ms"] = round(elapsed, 1)

    print(f"  Full-file correlation: {full_corr:.4f}")
    print(f"  Window correlations ({len(window_corrs)} windows):")
    print(f"    Mean: {np.mean(window_corrs):.4f}, Std: {np.std(window_corrs):.4f}")
    print(f"    Divergent (<0.5): {metrics['stereo_divergent_windows']}")
    print(f"    Correlated (>=0.5): {metrics['stereo_correlated_windows']}")
    print(f"  Analysis time: {elapsed:.1f}ms")
    assert full_corr < 0.5, f"Channels too correlated ({full_corr:.3f}) — expected true stereo"


@test("Mono mixing", "audio")
def test_mono_mix():
    with wave.open(STEREO_FILE, "rb") as wf:
        sr = wf.getframerate()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    ch0 = raw[0::2]; ch1 = raw[1::2]

    t0 = time.time()
    mono = ((ch0.astype(np.int32) + ch1.astype(np.int32)) // 2).astype(np.int16)
    elapsed = (time.time() - t0) * 1000

    # Verify no clipping
    max_val = np.max(np.abs(mono))
    rms = np.sqrt(np.mean(mono.astype(np.float32) ** 2))
    metrics["mono_mix_ms"] = round(elapsed, 1)
    metrics["mono_peak"] = int(max_val)
    metrics["mono_rms"] = round(rms, 1)

    print(f"  Mix time: {elapsed:.1f}ms")
    print(f"  Mono peak: {max_val}, RMS: {rms:.0f}")
    print(f"  Clipping: {'YES' if max_val >= 32767 else 'No'}")
    assert len(mono) == len(ch0)


# ════════════════════════════════════════════════════════════════
# CATEGORY: Model & Transcription
# ════════════════════════════════════════════════════════════════

@test("Model manager — available models", "model")
def test_model_list():
    sizes = model_manager.SIZES
    default = model_manager.DEFAULT_SIZE
    metrics["model_count"] = len(sizes)
    metrics["default_model"] = default

    print(f"  Available models: {len(sizes)}")
    for s in sizes:
        backend, model_id = model_manager.get_model_info(s)
        cached = model_manager.is_model_cached(s)
        print(f"    {'[cached]' if cached else '[     ]'} {s} → {backend}:{model_id}")
    print(f"  Default: {default}")
    assert len(sizes) > 0
    assert default in sizes


@test("MLX model load (cold)", "model")
def test_mlx_load_cold():
    model_id = "mlx-community/whisper-large-v3-turbo"
    gc.collect()

    t0 = time.time()
    # Warmup call (same as app.py _load_model_sync)
    dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
    _ = mlx_whisper.transcribe(dummy, path_or_hf_repo=model_id,
                                condition_on_previous_text=False)
    warmup_ms = (time.time() - t0) * 1000

    metrics["mlx_warmup_ms"] = round(warmup_ms, 1)
    print(f"  MLX warmup (model compile + load): {warmup_ms:.0f}ms")


@test("MLX transcription — 5s segment", "transcription")
def test_transcribe_5s():
    audio = _load_test_audio(duration=5.0)
    t0 = time.time()
    result = mlx_whisper.transcribe(audio, path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                     condition_on_previous_text=False, no_speech_threshold=0.5,
                                     language="ru")
    elapsed = (time.time() - t0) * 1000
    segs = [s for s in result.get("segments", []) if s.get("no_speech_prob", 0) < 0.5 and s["text"].strip()]
    text = " ".join(s["text"].strip() for s in segs)
    metrics["transcribe_5s_ms"] = round(elapsed, 1)
    metrics["transcribe_5s_rtf"] = round(elapsed / 5000, 3)  # real-time factor
    metrics["transcribe_5s_segments"] = len(segs)
    metrics["transcribe_5s_chars"] = len(text)
    print(f"  Time: {elapsed:.0f}ms (RTF: {elapsed/5000:.3f}x)")
    print(f"  Segments: {len(segs)}, Chars: {len(text)}")
    print(f"  Text: {text[:100]}")
    assert len(segs) > 0


@test("MLX transcription — 10s segment", "transcription")
def test_transcribe_10s():
    audio = _load_test_audio(duration=10.0)
    t0 = time.time()
    result = mlx_whisper.transcribe(audio, path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                     condition_on_previous_text=False, no_speech_threshold=0.5,
                                     language="ru")
    elapsed = (time.time() - t0) * 1000
    segs = [s for s in result.get("segments", []) if s.get("no_speech_prob", 0) < 0.5 and s["text"].strip()]
    text = " ".join(s["text"].strip() for s in segs)
    metrics["transcribe_10s_ms"] = round(elapsed, 1)
    metrics["transcribe_10s_rtf"] = round(elapsed / 10000, 3)
    print(f"  Time: {elapsed:.0f}ms (RTF: {elapsed/10000:.3f}x)")
    print(f"  Text: {text[:120]}")
    assert len(segs) > 0


@test("MLX transcription — full 44.5s file", "transcription")
def test_transcribe_full():
    audio = _load_test_audio(duration=None)  # full file
    dur_s = len(audio) / SAMPLE_RATE
    t0 = time.time()
    result = mlx_whisper.transcribe(audio, path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                     condition_on_previous_text=False, no_speech_threshold=0.5,
                                     language="ru")
    elapsed = (time.time() - t0) * 1000
    segs = [s for s in result.get("segments", []) if s.get("no_speech_prob", 0) < 0.5 and s["text"].strip()]
    text = " ".join(s["text"].strip() for s in segs)
    metrics["transcribe_full_ms"] = round(elapsed, 1)
    metrics["transcribe_full_rtf"] = round(elapsed / (dur_s * 1000), 3)
    metrics["transcribe_full_segments"] = len(segs)
    metrics["transcribe_full_chars"] = len(text)
    metrics["transcribe_full_duration_s"] = round(dur_s, 1)
    print(f"  Audio: {dur_s:.1f}s")
    print(f"  Time: {elapsed:.0f}ms (RTF: {metrics['transcribe_full_rtf']:.3f}x)")
    print(f"  Segments: {len(segs)}, Chars: {len(text)}")
    assert len(segs) > 5, f"Too few segments: {len(segs)}"


@test("Transcription consistency (3 runs)", "transcription")
def test_transcribe_consistency():
    audio = _load_test_audio(duration=10.0)
    texts = []
    times = []
    for i in range(3):
        t0 = time.time()
        result = mlx_whisper.transcribe(audio, path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                         condition_on_previous_text=False, no_speech_threshold=0.5,
                                         language="ru")
        elapsed = (time.time() - t0) * 1000
        times.append(elapsed)
        segs = [s for s in result.get("segments", []) if s.get("no_speech_prob", 0) < 0.5 and s["text"].strip()]
        text = " ".join(s["text"].strip() for s in segs)
        texts.append(text)
        print(f"  Run {i+1}: {elapsed:.0f}ms — {text[:80]}")

    # Check consistency
    identical = all(t == texts[0] for t in texts)
    metrics["transcribe_consistency_identical"] = identical
    metrics["transcribe_time_mean_ms"] = round(np.mean(times), 1)
    metrics["transcribe_time_std_ms"] = round(np.std(times), 1)
    metrics["transcribe_time_min_ms"] = round(min(times), 1)
    metrics["transcribe_time_max_ms"] = round(max(times), 1)

    if not identical:
        # Measure word overlap
        words = [set(t.lower().split()) for t in texts]
        common = words[0]
        for w in words[1:]:
            common &= w
        union = words[0]
        for w in words[1:]:
            union |= w
        jaccard = len(common) / len(union) if union else 1.0
        metrics["transcribe_consistency_jaccard"] = round(jaccard, 3)
        print(f"  Consistency: NOT identical (Jaccard={jaccard:.3f})")
    else:
        metrics["transcribe_consistency_jaccard"] = 1.0
        print(f"  Consistency: IDENTICAL across 3 runs")

    print(f"  Timing: mean={np.mean(times):.0f}ms, std={np.std(times):.0f}ms")


# ════════════════════════════════════════════════════════════════
# CATEGORY: Speaker Diarization
# ════════════════════════════════════════════════════════════════

@test("Peak-window diarization — known segments", "diarization")
def test_diarization_known():
    ch0, ch1 = _load_stereo_channels()
    global_rms0 = np.sqrt(np.mean(ch0 ** 2))
    global_rms1 = np.sqrt(np.mean(ch1 ** 2))
    print(f"  Channel baselines: ch0={global_rms0:.0f}, ch1={global_rms1:.0f} (ratio: {global_rms1/global_rms0:.2f}x)")

    WIN = int(0.1 * SAMPLE_RATE)
    # Ground truth from the recording: man (ch0 louder) vs woman (ch1 louder)
    ground_truth = [
        (2.0, 4.0, 1, "Man: Раз, два, три"),
        (6.0, 8.0, 2, "Woman: Осталось только подождать"),
        (12.0, 15.0, 2, "Woman: Три университета"),
        (15.0, 20.0, 2, "Woman: Northeastern, UCL"),
        (22.0, 23.5, 1, "Man: Да, понятно"),
        (24.0, 30.0, 2, "Woman: ждём четыре университета"),
        (30.0, 35.0, 2, "Woman: Kings тоже хорошая опция"),
    ]

    correct = 0
    total = len(ground_truth)
    t0 = time.time()
    for start, end, expected_speaker, desc in ground_truth:
        speaker = _peak_window_speaker(ch0, ch1, start, end, WIN, global_rms0, global_rms1)
        ok = speaker == expected_speaker
        if ok:
            correct += 1
        print(f"  [{start:.0f}-{end:.0f}s] Speaker {speaker} (expected {expected_speaker}) {'✓' if ok else '✗'} — {desc}")
    elapsed = (time.time() - t0) * 1000

    accuracy = correct / total
    metrics["diarization_accuracy"] = round(accuracy, 3)
    metrics["diarization_correct"] = correct
    metrics["diarization_total"] = total
    metrics["diarization_analysis_ms"] = round(elapsed, 1)
    print(f"  Accuracy: {correct}/{total} = {accuracy:.0%}")
    print(f"  Analysis time: {elapsed:.1f}ms")
    assert accuracy >= 0.85, f"Diarization accuracy too low: {accuracy:.0%}"


@test("Peak-window diarization — full file", "diarization")
def test_diarization_full():
    audio = _load_test_audio(duration=None)
    ch0, ch1 = _load_stereo_channels()
    global_rms0 = np.sqrt(np.mean(ch0 ** 2))
    global_rms1 = np.sqrt(np.mean(ch1 ** 2))
    WIN = int(0.1 * SAMPLE_RATE)

    # Transcribe
    result = mlx_whisper.transcribe(audio, path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                     condition_on_previous_text=False, no_speech_threshold=0.5,
                                     language="ru")
    segs = [(s["start"], s["end"], s["text"].strip()) for s in result.get("segments", [])
            if s.get("no_speech_prob", 0) < 0.5 and s["text"].strip()]

    # Assign speakers
    t0 = time.time()
    assignments = []
    for start, end, text in segs:
        speaker = _peak_window_speaker(ch0, ch1, start, end, WIN, global_rms0, global_rms1)
        assignments.append((speaker, start, end, text))
    elapsed = (time.time() - t0) * 1000

    sp1 = sum(1 for a in assignments if a[0] == 1)
    sp2 = sum(1 for a in assignments if a[0] == 2)

    metrics["diarization_full_segments"] = len(assignments)
    metrics["diarization_full_speaker1"] = sp1
    metrics["diarization_full_speaker2"] = sp2
    metrics["diarization_full_ms"] = round(elapsed, 1)

    print(f"  Total segments: {len(assignments)}")
    print(f"  Speaker 1 (man): {sp1} segments")
    print(f"  Speaker 2 (woman): {sp2} segments")
    print(f"  Diarization time: {elapsed:.1f}ms for {len(assignments)} segments")
    for speaker, start, end, text in assignments:
        ts = f"{int(start)//60:02d}:{int(start)%60:02d}"
        print(f"    [{ts}] Speaker {speaker}: {text[:60]}")

    assert sp1 > 0, "No Speaker 1 segments"
    assert sp2 > 0, "No Speaker 2 segments"
    assert sp2 > sp1, "Woman should have more segments than man in this recording"


@test("Diarization determinism (3 runs)", "diarization")
def test_diarization_determinism():
    ch0, ch1 = _load_stereo_channels()
    global_rms0 = np.sqrt(np.mean(ch0 ** 2))
    global_rms1 = np.sqrt(np.mean(ch1 ** 2))
    WIN = int(0.1 * SAMPLE_RATE)

    # Fixed segments
    segments = [(2.0, 4.0), (6.0, 8.0), (15.0, 20.0), (22.0, 23.5), (30.0, 35.0)]

    run_results = []
    for run in range(3):
        speakers = []
        for start, end in segments:
            speakers.append(_peak_window_speaker(ch0, ch1, start, end, WIN, global_rms0, global_rms1))
        run_results.append(tuple(speakers))

    identical = all(r == run_results[0] for r in run_results)
    metrics["diarization_deterministic"] = identical
    print(f"  Run 1: {run_results[0]}")
    print(f"  Run 2: {run_results[1]}")
    print(f"  Run 3: {run_results[2]}")
    print(f"  Deterministic: {'YES' if identical else 'NO'}")
    assert identical, "Diarization not deterministic!"


# ════════════════════════════════════════════════════════════════
# CATEGORY: End-to-End Pipeline
# ════════════════════════════════════════════════════════════════

@test("Full reprocess pipeline (end-to-end)", "e2e")
def test_reprocess_e2e():
    t_total = time.time()

    # Step 1: Read audio
    t0 = time.time()
    with wave.open(STEREO_FILE, "rb") as wf:
        sr = wf.getframerate()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    ch0_raw = raw[0::2]; ch1_raw = raw[1::2]
    read_ms = (time.time() - t0) * 1000

    # Step 2: Resample
    t0 = time.time()
    ratio = SAMPLE_RATE / sr
    new_len = int(len(ch0_raw) * ratio)
    indices = np.linspace(0, len(ch0_raw) - 1, new_len).astype(int)
    ch0 = ch0_raw[indices]; ch1 = ch1_raw[indices]
    resample_ms = (time.time() - t0) * 1000

    # Step 3: Mix to mono
    t0 = time.time()
    mono = ((ch0.astype(np.int32) + ch1.astype(np.int32)) // 2).astype(np.int16)
    audio_f32 = mono.astype(np.float32) / 32768.0
    mix_ms = (time.time() - t0) * 1000

    # Step 4: Transcribe
    t0 = time.time()
    result = mlx_whisper.transcribe(audio_f32, path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                     condition_on_previous_text=False, no_speech_threshold=0.5,
                                     language="ru")
    transcribe_ms = (time.time() - t0) * 1000
    segs = [(s["start"], s["end"], s["text"].strip()) for s in result.get("segments", [])
            if s.get("no_speech_prob", 0) < 0.5 and s["text"].strip()]

    # Step 5: Speaker diarization (with global normalization)
    t0 = time.time()
    ch_f32 = [ch0.astype(np.float32), ch1.astype(np.float32)]
    WIN = int(0.1 * SAMPLE_RATE)
    global_rms0 = np.sqrt(np.mean(ch_f32[0] ** 2))
    global_rms1 = np.sqrt(np.mean(ch_f32[1] ** 2))
    if global_rms0 < 1: global_rms0 = 1
    if global_rms1 < 1: global_rms1 = 1
    final_results = []
    for start, end, text in segs:
        si = int(start * SAMPLE_RATE)
        ei = min(int(end * SAMPLE_RATE), len(ch_f32[0]))
        if ei - si > WIN:
            combined = ch_f32[0][si:ei] ** 2 + ch_f32[1][si:ei] ** 2
            cumsum = np.cumsum(combined)
            win_sums = cumsum[WIN:] - cumsum[:-WIN]
            best = np.argmax(win_sums)
            wi, we = si + best, si + best + WIN
            peak0 = np.sqrt(np.mean(ch_f32[0][wi:we] ** 2))
            peak1 = np.sqrt(np.mean(ch_f32[1][wi:we] ** 2))
        elif ei > si:
            peak0 = np.sqrt(np.mean(ch_f32[0][si:ei] ** 2))
            peak1 = np.sqrt(np.mean(ch_f32[1][si:ei] ** 2))
        else:
            peak0 = peak1 = 0
        norm0 = peak0 / global_rms0
        norm1 = peak1 / global_rms1
        speaker = 0 if norm0 > norm1 else 1
        ts = f"{int(start)//60:02d}:{int(start)%60:02d}"
        final_results.append((speaker, ts, text))
    diarize_ms = (time.time() - t0) * 1000

    # Step 6: Format output (simulate save)
    t0 = time.time()
    output_lines = []
    for speaker, ts, text in final_results:
        output_lines.append(f"[{ts}] Speaker {speaker+1}: {text}")
    output_text = "\n".join(output_lines)
    format_ms = (time.time() - t0) * 1000

    total_ms = (time.time() - t_total) * 1000

    metrics["e2e_read_ms"] = round(read_ms, 1)
    metrics["e2e_resample_ms"] = round(resample_ms, 1)
    metrics["e2e_mix_ms"] = round(mix_ms, 1)
    metrics["e2e_transcribe_ms"] = round(transcribe_ms, 1)
    metrics["e2e_diarize_ms"] = round(diarize_ms, 1)
    metrics["e2e_format_ms"] = round(format_ms, 1)
    metrics["e2e_total_ms"] = round(total_ms, 1)
    metrics["e2e_segments"] = len(final_results)

    print(f"  Pipeline breakdown:")
    print(f"    1. Read audio:     {read_ms:7.1f}ms")
    print(f"    2. Resample:       {resample_ms:7.1f}ms")
    print(f"    3. Mono mix:       {mix_ms:7.1f}ms")
    print(f"    4. Transcribe:     {transcribe_ms:7.1f}ms")
    print(f"    5. Diarize:        {diarize_ms:7.1f}ms")
    print(f"    6. Format:         {format_ms:7.1f}ms")
    print(f"    ─────────────────────────────")
    print(f"    TOTAL:             {total_ms:7.1f}ms")
    print(f"")
    print(f"  Segments: {len(final_results)}")
    print(f"  RTF: {total_ms / (len(audio_f32)/SAMPLE_RATE*1000):.3f}x")


@test("Simulated real-time chunked transcription", "e2e")
def test_realtime_chunked():
    """Simulate real-time recording: feed 1-second chunks, transcribe every 3s."""
    audio = _load_test_audio(duration=15.0)
    chunk_size = SAMPLE_RATE  # 1 second
    transcribe_interval = 3  # transcribe every 3 chunks
    buffer = np.array([], dtype=np.float32)

    transcriptions = []
    chunk_times = []
    transcribe_times = []

    for i in range(0, len(audio), chunk_size):
        t0 = time.time()
        chunk = audio[i:i+chunk_size]
        buffer = np.concatenate([buffer, chunk])
        chunk_ms = (time.time() - t0) * 1000
        chunk_times.append(chunk_ms)

        chunk_num = i // chunk_size + 1
        if chunk_num % transcribe_interval == 0 and len(buffer) > 0:
            t0 = time.time()
            result = mlx_whisper.transcribe(buffer, path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                             condition_on_previous_text=False, no_speech_threshold=0.5,
                                             language="ru")
            t_ms = (time.time() - t0) * 1000
            transcribe_times.append(t_ms)
            segs = [s["text"].strip() for s in result.get("segments", [])
                   if s.get("no_speech_prob", 0) < 0.5 and s["text"].strip()]
            text = " ".join(segs)
            transcriptions.append((chunk_num, t_ms, text))
            print(f"    Chunk {chunk_num}: transcribe {len(buffer)/SAMPLE_RATE:.1f}s → {t_ms:.0f}ms — {text[:60]}")

    metrics["realtime_chunk_buffer_ms"] = round(np.mean(chunk_times), 2)
    metrics["realtime_transcribe_mean_ms"] = round(np.mean(transcribe_times), 1)
    metrics["realtime_transcribe_max_ms"] = round(max(transcribe_times), 1)
    metrics["realtime_transcriptions"] = len(transcriptions)

    print(f"\n  Chunk buffer: mean {np.mean(chunk_times):.2f}ms")
    print(f"  Transcribe: mean {np.mean(transcribe_times):.0f}ms, max {max(transcribe_times):.0f}ms")
    print(f"  Transcriptions: {len(transcriptions)}")
    assert len(transcriptions) > 0


# ════════════════════════════════════════════════════════════════
# CATEGORY: System & Memory
# ════════════════════════════════════════════════════════════════

@test("Memory usage snapshot", "system")
def test_memory():
    mem = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_mb = mem.ru_maxrss / (1024 * 1024)  # macOS reports bytes
    metrics["peak_memory_mb"] = round(max_rss_mb, 1)
    metrics["python_version"] = platform.python_version()
    metrics["platform"] = platform.platform()
    metrics["cpu"] = platform.processor()

    print(f"  Peak RSS: {max_rss_mb:.0f}MB")
    print(f"  Python: {platform.python_version()}")
    print(f"  Platform: {platform.platform()}")
    print(f"  CPU: {platform.processor()}")


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def _load_test_audio(duration=None):
    """Load stereo recording, mix to mono, resample to 16kHz, return float32."""
    with wave.open(STEREO_FILE, "rb") as wf:
        sr = wf.getframerate()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    ch0 = raw[0::2]; ch1 = raw[1::2]
    mono = ((ch0.astype(np.int32) + ch1.astype(np.int32)) // 2).astype(np.int16)
    ratio = SAMPLE_RATE / sr
    new_len = int(len(mono) * ratio)
    indices = np.linspace(0, len(mono) - 1, new_len).astype(int)
    mono_16k = mono[indices]
    if duration is not None:
        mono_16k = mono_16k[:int(SAMPLE_RATE * duration)]
    return mono_16k.astype(np.float32) / 32768.0


def _load_stereo_channels():
    """Load stereo channels, resample to 16kHz, return (ch0_f32, ch1_f32)."""
    with wave.open(STEREO_FILE, "rb") as wf:
        sr = wf.getframerate()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    ch0 = raw[0::2]; ch1 = raw[1::2]
    ratio = SAMPLE_RATE / sr
    new_len = int(len(ch0) * ratio)
    indices = np.linspace(0, len(ch0) - 1, new_len).astype(int)
    return ch0[indices].astype(np.float32), ch1[indices].astype(np.float32)


def _peak_window_speaker(ch0, ch1, start, end, win, global_rms0=None, global_rms1=None):
    """Return 1 (ch0 louder) or 2 (ch1 louder) using normalized peak-window analysis."""
    si = int(start * SAMPLE_RATE)
    ei = min(int(end * SAMPLE_RATE), len(ch0))
    if ei - si > win:
        combined = ch0[si:ei] ** 2 + ch1[si:ei] ** 2
        cumsum = np.cumsum(combined)
        win_sums = cumsum[win:] - cumsum[:-win]
        best = np.argmax(win_sums)
        wi, we = si + best, si + best + win
        peak0 = np.sqrt(np.mean(ch0[wi:we] ** 2))
        peak1 = np.sqrt(np.mean(ch1[wi:we] ** 2))
    elif ei > si:
        peak0 = np.sqrt(np.mean(ch0[si:ei] ** 2))
        peak1 = np.sqrt(np.mean(ch1[si:ei] ** 2))
    else:
        return 1
    # Normalize by global channel energy to compensate for gain imbalance
    if global_rms0 and global_rms1:
        norm0 = peak0 / global_rms0
        norm1 = peak1 / global_rms1
        return 1 if norm0 > norm1 else 2
    return 1 if peak0 > peak1 else 2


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  WhisperTranscribe — Regression & Performance Suite     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Python {sys.version.split()[0]} | {platform.platform()}")
    print(f"  Working dir: {os.getcwd()}")
    print()

    all_tests = [
        # Device & Hardware
        test_device_enum,
        test_coreaudio,
        test_monitor,
        # File I/O
        test_read_stereo,
        test_wav_roundtrip,
        test_transcript_io,
        test_recordings_list,
        # Audio Processing
        test_resample,
        test_correlation,
        test_mono_mix,
        # Model & Transcription
        test_model_list,
        test_mlx_load_cold,
        test_transcribe_5s,
        test_transcribe_10s,
        test_transcribe_full,
        test_transcribe_consistency,
        # Speaker Diarization
        test_diarization_known,
        test_diarization_full,
        test_diarization_determinism,
        # End-to-End
        test_reprocess_e2e,
        test_realtime_chunked,
        # System
        test_memory,
    ]

    for t in all_tests:
        run_test(t)

    # ── Summary ──
    passed = sum(1 for r in results if r["status"] == "PASSED")
    failed = sum(1 for r in results if r["status"] == "FAILED")

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  RESULTS                                                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # By category
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"passed": 0, "failed": 0, "total_ms": 0}
        categories[cat]["passed" if r["status"] == "PASSED" else "failed"] += 1
        categories[cat]["total_ms"] += r["time_ms"]

    for cat, stats in categories.items():
        total = stats["passed"] + stats["failed"]
        status = "✓" if stats["failed"] == 0 else "✗"
        print(f"  {status} {cat:20s} {stats['passed']}/{total} passed  ({stats['total_ms']:.0f}ms)")

    print(f"\n  TOTAL: {passed}/{passed+failed} passed, {failed} failed")
    print(f"  Total time: {sum(r['time_ms'] for r in results)/1000:.1f}s")

    # ── Metrics ──
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  PRODUCT METRICS                                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    metric_groups = {
        "Performance (Response Times)": [
            ("MLX warmup (cold start)", "mlx_warmup_ms", "ms"),
            ("Transcribe 5s audio", "transcribe_5s_ms", "ms"),
            ("Transcribe 10s audio", "transcribe_10s_ms", "ms"),
            ("Transcribe 44.5s (full)", "transcribe_full_ms", "ms"),
            ("RTF (5s)", "transcribe_5s_rtf", "x"),
            ("RTF (full file)", "transcribe_full_rtf", "x"),
            ("Real-time chunk transcribe (mean)", "realtime_transcribe_mean_ms", "ms"),
            ("Real-time chunk transcribe (max)", "realtime_transcribe_max_ms", "ms"),
            ("E2E reprocess total", "e2e_total_ms", "ms"),
        ],
        "Pipeline Breakdown (44.5s file)": [
            ("Read audio", "e2e_read_ms", "ms"),
            ("Resample 48→16kHz", "e2e_resample_ms", "ms"),
            ("Mono mix", "e2e_mix_ms", "ms"),
            ("Transcription", "e2e_transcribe_ms", "ms"),
            ("Speaker diarization", "e2e_diarize_ms", "ms"),
        ],
        "Quality": [
            ("Transcription consistency (Jaccard)", "transcribe_consistency_jaccard", ""),
            ("Diarization accuracy", "diarization_accuracy", ""),
            ("Diarization deterministic", "diarization_deterministic", ""),
            ("Full-file segments", "transcribe_full_segments", ""),
        ],
        "Audio": [
            ("Stereo correlation", "stereo_full_correlation", ""),
            ("Divergent windows", "stereo_divergent_windows", ""),
            ("Monitor callback interval", "monitor_latency_ms", "ms"),
        ],
        "System": [
            ("Peak memory", "peak_memory_mb", "MB"),
            ("Python version", "python_version", ""),
            ("Available models", "model_count", ""),
            ("Recordings on disk", "recordings_count", ""),
        ],
    }

    for group_name, items in metric_groups.items():
        print(f"  {group_name}:")
        for label, key, unit in items:
            val = metrics.get(key, "N/A")
            if isinstance(val, float):
                print(f"    {label:45s} {val:>10.3f} {unit}")
            else:
                print(f"    {label:45s} {str(val):>10s} {unit}")
        print()

    # Save metrics to JSON
    # Convert numpy types for JSON serialization
    def jsonify(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj
    clean_metrics = {k: jsonify(v) for k, v in metrics.items()}
    clean_results = [{k: jsonify(v) for k, v in r.items()} for r in results]

    metrics_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump({"metrics": clean_metrics, "results": clean_results, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"  Metrics saved to: {metrics_file}")

    sys.exit(1 if failed > 0 else 0)
