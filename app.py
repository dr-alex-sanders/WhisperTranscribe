"""WhisperTranscribe — Lightweight macOS Speech-to-Text App."""

import math
import os
import sys
import queue
import threading
import time
import wave
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime

import numpy as np
import sounddevice as sd

# Ensure print output is unbuffered (so logs appear in real-time)
if not sys.stdout.line_buffering:
    sys.stdout.reconfigure(line_buffering=True)

import model_manager

# Stub out numba & scipy before importing mlx_whisper — they deadlock on
# Python 3.14 and are only used for optional word-level timestamps we don't need.
import types as _types
_fake_numba = _types.ModuleType("numba")
_fake_numba.jit = lambda *a, **kw: (lambda f: f)  # no-op decorator
sys.modules["numba"] = _fake_numba
_fake_scipy = _types.ModuleType("scipy")
_fake_scipy_sig = _types.ModuleType("scipy.signal")
_fake_scipy.signal = _fake_scipy_sig
sys.modules["scipy"] = _fake_scipy
sys.modules["scipy.signal"] = _fake_scipy_sig

try:
    import mlx_whisper as _mlx_whisper
    print("[startup] mlx_whisper pre-imported OK")
except ImportError:
    _mlx_whisper = None
    print("[startup] mlx_whisper not available")

# Remove fake scipy so other code can load the real one
del sys.modules["scipy"]
del sys.modules["scipy.signal"]

# resemblyzer requires PyTorch which deadlocks on Python 3.14 (get_data hang).
# Speaker diarization is disabled until torch is fixed for 3.14.
_resemblyzer_encoder = None
_resemblyzer_ready = threading.Event()
_resemblyzer_ready.set()  # mark as done (no-op)

SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # ~250ms chunks at 16kHz

PROCESSING_MODES = [
    "Mono (no diarization)",
    "RMS Energy",
    "Peak-Window",
    "Norm. Peak-Window",
    "Dominance-Flip",
    "Per-Channel",
]
PARTIAL_INTERVAL = 1.0  # re-transcribe working buffer every N seconds
COMMIT_SECONDS = 10  # commit working buffer as final text after N seconds (max)
SILENCE_COMMIT = 1.0  # commit after N seconds of silence following speech
SPEECH_RMS_THRESHOLD = 0.005  # RMS below this = silence

SIZES = model_manager.SIZES

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")


def get_input_devices():
    devices = sd.query_devices()
    inputs = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            max_ch = min(d["max_input_channels"], 2)
            inputs.append((i, d["name"], max_ch, int(d["default_samplerate"])))
    return inputs


class SpeakerTracker:
    """Track speakers using neural voice embeddings (resemblyzer GE2E)."""

    _shared_encoder = None  # class-level to avoid reloading the neural net

    def __init__(self, max_speakers=2, threshold=0.75):
        self.max_speakers = max_speakers
        self.threshold = threshold  # cosine similarity threshold (>= means same speaker)
        self.centroids = []  # list of embedding vectors per speaker
        self.counts = []  # number of segments per speaker
        # Use pre-loaded encoder from background thread (non-blocking)
        if _resemblyzer_encoder is None:
            raise RuntimeError("resemblyzer not ready yet")
        self.encoder = _resemblyzer_encoder

    def identify(self, audio_f32, sr=16000):
        """Return speaker index (0-based) for an audio segment."""
        if len(audio_f32) < sr * 0.3:
            return 0

        from resemblyzer import preprocess_wav

        # Resample to 16kHz if needed
        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(len(audio_f32) * ratio)
            indices = np.linspace(0, len(audio_f32) - 1, new_len).astype(int)
            audio_f32 = audio_f32[indices]

        wav = preprocess_wav(audio_f32, source_sr=16000)
        if len(wav) < 1600:  # too short after VAD
            return 0

        embed = self.encoder.embed_utterance(wav)

        if not self.centroids:
            self.centroids.append(embed)
            self.counts.append(1)
            print(f"[speaker] new Speaker 1 (first)")
            return 0

        # Cosine similarity
        sims = []
        for c in self.centroids:
            sim = np.dot(embed, c) / (np.linalg.norm(embed) * np.linalg.norm(c) + 1e-9)
            sims.append(sim)
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        print(f"[speaker] sims={[f'{s:.3f}' for s in sims]}, best={best_idx} ({best_sim:.3f})")

        if best_sim >= self.threshold:
            # Same speaker — update centroid with running average
            n = self.counts[best_idx]
            self.centroids[best_idx] = (self.centroids[best_idx] * n + embed) / (n + 1)
            self.counts[best_idx] = n + 1
            return best_idx

        # New speaker
        if len(self.centroids) < self.max_speakers:
            self.centroids.append(embed)
            self.counts.append(1)
            idx = len(self.centroids) - 1
            print(f"[speaker] new Speaker {idx + 1} (sim={best_sim:.3f} < {self.threshold})")
            return idx

        # Max speakers reached — assign to closest
        n = self.counts[best_idx]
        self.centroids[best_idx] = (self.centroids[best_idx] * n + embed) / (n + 1)
        self.counts[best_idx] = n + 1
        return best_idx


class ChannelState:
    """Per-channel state for the audio processing loop."""
    __slots__ = (
        "working_buffer", "had_speech", "last_speech_time",
        "last_transcribe_time", "speech_samples_in_buffer",
        "energy_sum", "energy_count",
    )

    def __init__(self):
        self.working_buffer = bytearray()
        self.had_speech = False
        self.last_speech_time = 0.0
        self.last_transcribe_time = 0.0
        self.speech_samples_in_buffer = 0
        self.energy_sum = 0.0
        self.energy_count = 0


class WhisperTranscribe:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("WhisperTranscribe")
        self.root.geometry("850x600")
        self.root.minsize(700, 450)

        self.model = None
        self._backend = None  # "mlx"
        self._transcribe_lock = threading.Lock()  # serialize GPU transcription
        self._speaker_tracker = None  # initialized when recording starts
        self.recording = False
        self.paused = False
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        self.selected_device = None
        self._loaded_key = None
        self.mic_level = [0.0, 0.0]

        # Recording to file
        self.recorded_frames = []
        self.current_wav_path = None

        # Playback
        self.playing = False
        self.play_thread = None
        self._reprocessing = False

        # Stereo / channel tracking
        self._device_channels = 1  # 1=mono, 2=stereo
        self._device_samplerate = SAMPLE_RATE  # native device sample rate

        # Channel bleeding detection (rolling correlation)
        self._bleed_history = []  # recent boolean values
        self._bleed_alarm_on = False
        self._bleed_calibration = []  # energy ratios during calibration phase
        self._bleed_baseline = None   # calibrated baseline energy ratio

        # Background mic monitor (for level meter when not recording)
        self.monitoring = False
        self.monitor_thread = None

        os.makedirs(RECORDINGS_DIR, exist_ok=True)

        self._build_ui()
        self._refresh_devices()
        self._refresh_recordings_list()
        self._update_status("Select model size, then click Record")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_paused = False  # pause polling during model loading (Metal conflict)
        # Seed known devices from CoreAudio so first poll doesn't trigger a false refresh
        try:
            self._known_devices = self._query_coreaudio_inputs() or set()
        except Exception:
            self._known_devices = set()
        self._poll_devices()
        # Auto-preload the default model in background so Record starts instantly
        self.root.after(500, self._auto_preload_model)

    def _build_ui(self):
        # --- Main paned area: transcript left, recordings right ---
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=6)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        # Left: text area
        text_frame = tk.Frame(paned)
        self.text = tk.Text(
            text_frame, wrap=tk.WORD, font=("Helvetica", 16),
            relief=tk.FLAT, bd=0, highlightthickness=1,
            highlightcolor="#007AFF", highlightbackground="#E0E0E0", insertwidth=2,
        )
        text_scroll = ttk.Scrollbar(text_frame, command=self.text.yview)
        self.text.configure(yscrollcommand=text_scroll.set)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(fill=tk.BOTH, expand=True)
        self.text.tag_configure("partial", foreground="gray")
        self.text.tag_configure("partial_0", foreground="gray")
        self.text.tag_configure("partial_1", foreground="gray")
        self.text.tag_configure("speaker1", foreground="#0066CC")
        self.text.tag_configure("speaker2", foreground="#228B22")
        self.text.tag_configure("speaker_unknown", foreground="#CC6600")
        self.text.tag_configure("timestamp", foreground="#999999", font=("Helvetica", 13))
        paned.add(text_frame, stretch="always", minsize=300)

        # Right: recordings panel
        rec_frame = tk.Frame(paned)
        tk.Label(rec_frame, text="Recordings", font=("Helvetica", 13, "bold")).pack(anchor=tk.W, padx=4, pady=(4, 2))
        list_frame = tk.Frame(rec_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.rec_listbox = tk.Listbox(
            list_frame, font=("Helvetica", 12), activestyle="none",
            selectbackground="#007AFF", selectforeground="white",
            exportselection=False,
        )
        rec_scroll = ttk.Scrollbar(list_frame, command=self.rec_listbox.yview)
        self.rec_listbox.configure(yscrollcommand=rec_scroll.set)
        rec_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.rec_listbox.pack(fill=tk.BOTH, expand=True)
        self.rec_listbox.bind("<Double-1>", lambda e: self._play_selected())
        self.rec_listbox.bind("<<ListboxSelect>>", lambda e: self._on_recording_selected())

        # Play/Stop play buttons for recordings
        play_bar = tk.Frame(rec_frame)
        play_bar.pack(fill=tk.X, pady=(4, 4), padx=4)
        self.play_btn = tk.Button(
            play_bar, text="Play", font=("Helvetica", 12),
            command=self._play_selected, fg="#007AFF",
        )
        self.play_btn.pack(side=tk.LEFT, padx=(0, 4), fill=tk.X, expand=True)
        self.stop_play_btn = tk.Button(
            play_bar, text="Stop", font=("Helvetica", 12),
            command=self._stop_playback, state=tk.DISABLED, fg="#CC0000",
        )
        self.stop_play_btn.pack(side=tk.LEFT, padx=(0, 4), fill=tk.X, expand=True)
        self.delete_btn = tk.Button(
            play_bar, text="Delete", font=("Helvetica", 12),
            command=self._delete_selected,
        )
        self.delete_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        paned.add(rec_frame, stretch="never", minsize=180, width=220)

        # --- Mic selector row ---
        mic_bar = tk.Frame(self.root)
        mic_bar.pack(fill=tk.X, padx=8, pady=(6, 0))

        tk.Label(mic_bar, text="Mic:", font=("Helvetica", 12)).pack(side=tk.LEFT)
        self.mic_var = tk.StringVar()
        self.mic_combo = ttk.Combobox(
            mic_bar, textvariable=self.mic_var, state="readonly", font=("Helvetica", 11),
        )
        self.mic_combo.pack(side=tk.LEFT, padx=(4, 8), fill=tk.X, expand=True)
        self.mic_combo.bind("<<ComboboxSelected>>", lambda _: self._on_mic_changed())

        tk.Button(mic_bar, text="Refresh", font=("Helvetica", 11), command=self._refresh_devices).pack(side=tk.LEFT)

        ttk.Separator(mic_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        tk.Label(mic_bar, text="Processing:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.processing_var = tk.StringVar(value="Norm. Peak-Window")
        self.processing_combo = ttk.Combobox(
            mic_bar, textvariable=self.processing_var,
            values=PROCESSING_MODES,
            state="readonly", width=20, font=("Helvetica", 11),
        )
        self.processing_combo.pack(side=tk.LEFT, padx=(4, 0))

        # --- Mic level indicator (dual channel) ---
        level_bar = tk.Frame(self.root)
        level_bar.pack(fill=tk.X, padx=8, pady=(4, 0))

        tk.Label(level_bar, text="Ch1:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.level_canvas_0 = tk.Canvas(level_bar, height=12, bg="#E0E0E0", highlightthickness=0)
        self.level_canvas_0.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
        self.level_rect_0 = self.level_canvas_0.create_rectangle(0, 0, 0, 12, fill="#4CAF50", width=0)
        self.level_canvas_0.bind("<Configure>", lambda e: self._draw_level())

        tk.Label(level_bar, text="Ch2:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.level_canvas_1 = tk.Canvas(level_bar, height=12, bg="#E0E0E0", highlightthickness=0)
        self.level_canvas_1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self.level_rect_1 = self.level_canvas_1.create_rectangle(0, 0, 0, 12, fill="#4CAF50", width=0)
        self.level_canvas_1.bind("<Configure>", lambda e: self._draw_level())

        # --- Channel bleeding alarm (hidden by default) ---
        self.bleed_frame = tk.Frame(self.root, bg="#FFE0E0")
        # Not packed initially — use pack()/pack_forget() to show/hide
        self.bleed_label = tk.Label(
            self.bleed_frame, text="",
            font=("Helvetica", 12, "bold"), fg="#CC0000", bg="#FFE0E0",
            anchor=tk.W, padx=8, pady=2,
        )
        self.bleed_label.pack(fill=tk.X)
        self._bleed_visible = False
        # Remember the widget that comes after, so we can re-pack in correct order
        self._bleed_pack_after = self.level_canvas_0.master  # level_bar

        # --- Progress bar (always packed, content hidden by default) ---
        self.progress_frame = tk.Frame(self.root, height=0)
        self.progress_frame.pack(fill=tk.X, padx=8)
        self.progress_frame.pack_propagate(False)
        self.progress_label = tk.Label(self.progress_frame, text="", font=("Helvetica", 11), anchor=tk.W)
        self.progress_label.pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode="determinate", length=200)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # --- Bottom toolbar: buttons ---
        toolbar = tk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=8, pady=(8, 4))

        self.start_btn = tk.Button(
            toolbar, text="Record", width=7, command=self._on_start,
            font=("Helvetica", 13), fg="#CC0000",
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.pause_btn = tk.Button(
            toolbar, text="Pause", width=7, command=self._on_pause,
            font=("Helvetica", 13), state=tk.DISABLED,
        )
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.stop_btn = tk.Button(
            toolbar, text="Stop", width=7, command=self._on_stop,
            font=("Helvetica", 13), state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.reprocess_btn = tk.Button(
            toolbar, text="Upload…", command=self._on_reprocess_file,
            font=("Helvetica", 13), fg="#6A0DAD",
        )
        self.reprocess_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.rerun_btn = tk.Button(
            toolbar, text="Reprocess", command=self._on_rerun_selected,
            font=("Helvetica", 13), fg="#6A0DAD", state=tk.DISABLED,
        )
        self.rerun_btn.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(toolbar, text="Model:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.size_var = tk.StringVar(value=model_manager.DEFAULT_SIZE)
        model_combo = ttk.Combobox(
            toolbar, textvariable=self.size_var, values=SIZES,
            state="readonly", width=18, font=("Helvetica", 11),
            height=len(SIZES),
        )
        model_combo.pack(side=tk.LEFT, padx=(4, 6))

        tk.Label(toolbar, text="Lang:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value="ru")
        ttk.Combobox(
            toolbar, textvariable=self.lang_var,
            values=["ru", "en", "es", "fr", "de", "zh", "ja", "ko", "pt", "it", "auto"],
            state="readonly", width=5, font=("Helvetica", 11),
        ).pack(side=tk.LEFT, padx=(4, 0))

        self.status_label = tk.Label(
            toolbar, text="", font=("Helvetica", 11), fg="gray", anchor=tk.E,
        )
        self.status_label.pack(side=tk.RIGHT)

        # --- Prompt row (context hints for domain-specific vocabulary) ---
        prompt_bar = tk.Frame(self.root)
        prompt_bar.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(prompt_bar, text="Prompt:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.prompt_var = tk.StringVar()
        self.prompt_entry = tk.Entry(
            prompt_bar, textvariable=self.prompt_var, font=("Helvetica", 11),
        )
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

    # --- Recordings list ---

    def _refresh_recordings_list(self):
        self.rec_listbox.delete(0, tk.END)
        if not os.path.isdir(RECORDINGS_DIR):
            return
        files = sorted(
            [f for f in os.listdir(RECORDINGS_DIR) if f.endswith((".wav", ".mp3"))],
            reverse=True,
        )
        for f in files:
            self.rec_listbox.insert(tk.END, f)

    def _on_recording_selected(self, _event=None):
        """Show the transcript for the selected recording in the text area."""
        if self.recording:
            return
        path = self._get_selected_recording_path()
        if not path:
            self.rerun_btn.config(state=tk.DISABLED)
            return
        self.rerun_btn.config(state=tk.NORMAL)
        txt_path = os.path.splitext(path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                transcript = f.read()
            self.text.delete("1.0", tk.END)
            self.text.insert("1.0", transcript)
            self._update_status(f"Transcript: {os.path.basename(path)}")
        else:
            self.text.delete("1.0", tk.END)
            self._update_status(f"No transcript for {os.path.basename(path)}")

    def _get_selected_recording_path(self):
        sel = self.rec_listbox.curselection()
        if not sel:
            return None
        filename = self.rec_listbox.get(sel[0])
        return os.path.join(RECORDINGS_DIR, filename)

    def _play_selected(self):
        path = self._get_selected_recording_path()
        if not path:
            self._update_status("Select a recording to play")
            return
        if self.playing:
            self._stop_playback()
        self.playing = True
        self.play_btn.config(state=tk.DISABLED)
        self.stop_play_btn.config(state=tk.NORMAL)
        self._update_status(f"Playing: {os.path.basename(path)}")
        self.play_thread = threading.Thread(target=self._play_audio, args=(path,), daemon=True)
        self.play_thread.start()

    def _play_audio(self, path):
        try:
            if path.lower().endswith(".wav"):
                with wave.open(path, "rb") as wf:
                    sr = wf.getframerate()
                    channels = wf.getnchannels()
                    data = wf.readframes(wf.getnframes())
                audio = np.frombuffer(data, dtype=np.int16)
            else:
                import av
                with av.open(path) as container:
                    stream = container.streams.audio[0]
                    sr = stream.rate
                    channels = stream.channels
                    frames = []
                    for frame in container.decode(audio=0):
                        frames.append(frame.to_ndarray())
                audio_f = np.concatenate(frames, axis=1)
                audio_f = np.clip(audio_f, -1.0, 1.0)
                audio = (np.column_stack([audio_f[ch] for ch in range(audio_f.shape[0])]).flatten() * 32767).astype(np.int16)
            if channels > 1:
                audio = audio.reshape(-1, channels)
            sd.play(audio, samplerate=sr)
            sd.wait()
        except Exception as e:
            self.root.after(0, self._update_status, f"Play error: {e}")
        finally:
            self.playing = False
            self.root.after(0, self._on_playback_finished)

    def _stop_playback(self):
        sd.stop()
        self.playing = False
        self._on_playback_finished()

    def _on_playback_finished(self):
        self.play_btn.config(state=tk.NORMAL)
        self.stop_play_btn.config(state=tk.DISABLED)
        if not self.recording:
            self._update_status("Stopped")

    def _delete_selected(self):
        path = self._get_selected_recording_path()
        if not path:
            return
        if self.playing:
            self._stop_playback()
        try:
            os.remove(path)
            # Also remove matching .txt if exists
            base, _ = os.path.splitext(path)
            txt_path = base + ".txt"
            if os.path.exists(txt_path):
                os.remove(txt_path)
        except OSError:
            pass
        self._refresh_recordings_list()

    def _on_rerun_selected(self):
        """Reprocess the currently selected recording with the current model."""
        path = self._get_selected_recording_path()
        if not path:
            self._update_status("Select a recording to reprocess")
            return
        if self.recording:
            self._update_status("Stop recording before reprocessing")
            return
        self._reprocess_saved_path = path
        self._reprocess_path(path)

    def _on_reprocess_file(self):
        if self.recording:
            self._update_status("Stop recording before reprocessing")
            return
        path = filedialog.askopenfilename(
            title="Select audio file to reprocess",
            filetypes=[("Audio files", "*.wav *.mp3"), ("WAV files", "*.wav"), ("MP3 files", "*.mp3"), ("All files", "*.*")],
            initialdir=RECORDINGS_DIR,
        )
        if not path:
            return
        self._reprocess_path(path)

    def _reprocess_path(self, path):
        if self.recording:
            self._update_status("Stop recording before reprocessing")
            return
        if self._reprocessing:
            self._update_status("Reprocessing already in progress...")
            return
        # Load model if needed
        size = self.size_var.get()
        if self._loaded_key != size:
            self._set_loading_ui()
            self.rerun_btn.config(state=tk.DISABLED)
            self._show_progress(f"Loading {size} model...", 10, 100)
            self._reprocess_pending_path = path
            self._load_model_for_reprocess(size)
            return
        self._do_reprocess(path)

    def _load_model_for_reprocess(self, size):
        self._poll_paused = True
        self._stop_monitor(wait=True)
        cached = model_manager.is_model_cached(size)
        if cached:
            self._update_status(f"Loading {size} model...")
        else:
            self._update_status(f"Downloading & loading {size} model (first time)...")
        threading.Thread(target=self._load_model_sync_reprocess, args=(size,), daemon=True).start()

    def _load_model_sync_reprocess(self, size):
        try:
            backend, model_id = model_manager.get_model_info(size)
            import mlx_whisper
            dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
            try:
                mlx_whisper.transcribe(dummy, path_or_hf_repo=model_id)
            except Exception as e:
                print(f"[model] mlx warmup failed ({e}), retrying...")
                import gc; gc.collect()
                time.sleep(1)
                mlx_whisper.transcribe(dummy, path_or_hf_repo=model_id)
            model = model_id
            self.root.after(0, self._on_model_loaded_reprocess, size, backend, model, None)
        except Exception as e:
            self.root.after(0, self._on_model_loaded_reprocess, size, None, None, e)

    def _on_model_loaded_reprocess(self, size, backend, model, error):
        self._poll_paused = False
        if error:
            self._set_stopped_ui()
            self._update_status(f"Failed to load model: {error}")
            return
        self.model = model
        self._backend = backend
        self._loaded_key = size
        path = getattr(self, "_reprocess_pending_path", None)
        if path:
            self._reprocess_pending_path = None
            self._do_reprocess(path)

    def _show_progress(self, text="", value=0, maximum=100):
        self.progress_label.config(text=text)
        self.progress_bar.config(maximum=maximum, value=value)
        self.progress_frame.config(height=24)
        self.progress_frame.pack_configure(pady=(4, 0))

    def _hide_progress(self):
        self.progress_frame.config(height=0)
        self.progress_frame.pack_configure(pady=0)

    def _do_reprocess(self, path):
        if self._reprocessing:
            return
        self._reprocessing = True
        self.reprocess_btn.config(state=tk.DISABLED)
        self.rerun_btn.config(state=tk.DISABLED)
        self._update_status(f"Reprocessing: {os.path.basename(path)}...")
        self.text.delete("1.0", tk.END)
        self._show_progress("Reading file...", 0, 100)
        threading.Thread(target=self._reprocess_file, args=(path,), daemon=True).start()

    @staticmethod
    def _read_audio_file(path):
        """Read WAV or MP3 file, return (channels_data, sample_rate).

        channels_data is a list of int16 numpy arrays, one per channel.
        """
        if path.lower().endswith(".wav"):
            with wave.open(path, "rb") as wf:
                n_channels = wf.getnchannels()
                sr = wf.getframerate()
                raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        else:
            import av
            with av.open(path) as container:
                stream = container.streams.audio[0]
                sr = stream.rate
                n_channels = stream.channels
                frames = []
                for frame in container.decode(audio=0):
                    arr = frame.to_ndarray()  # shape: (channels, samples) float
                    frames.append(arr)
            audio_f = np.concatenate(frames, axis=1)  # (channels, total_samples)
            # Convert to interleaved int16
            audio_f = np.clip(audio_f, -1.0, 1.0)
            interleaved = np.column_stack([audio_f[ch] for ch in range(audio_f.shape[0])])
            raw = (interleaved.flatten() * 32767).astype(np.int16)

        if n_channels >= 2:
            channels_data = [raw[ch::n_channels] for ch in range(min(n_channels, 2))]
        else:
            channels_data = [raw]
        return channels_data, sr

    def _transcribe_audio(self, audio_f32, lang, prompt):
        """Transcribe float32 audio, return list of (start, end, text) segments."""
        seg_list = []
        import mlx_whisper
        kwargs = {"path_or_hf_repo": self.model, "condition_on_previous_text": False, "no_speech_threshold": 0.5}
        if lang and lang != "auto":
            kwargs["language"] = lang
        if prompt:
            kwargs["initial_prompt"] = prompt
        result = mlx_whisper.transcribe(audio_f32, **kwargs)
        for s in result.get("segments", []):
            if s.get("no_speech_prob", 0) < 0.5 and s["text"].strip():
                seg_list.append((s["start"], s["end"], s["text"].strip()))
        return seg_list

    def _reprocess_file(self, path):
        import traceback, shutil
        try:
            print(f"[reprocess] reading: {path}")
            self.root.after(0, self._show_progress, "Reading file...", 10, 100)
            channels_data, sr = self._read_audio_file(path)
            print(f"[reprocess] channels={len(channels_data)}, sr={sr}, samples={len(channels_data[0])}")

            # Copy file to recordings dir if it's not already there
            abs_rec = os.path.abspath(RECORDINGS_DIR)
            abs_path = os.path.abspath(path)
            if not abs_path.startswith(abs_rec + os.sep):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ext = os.path.splitext(path)[1]
                dest = os.path.join(RECORDINGS_DIR, f"rec_{timestamp}{ext}")
                shutil.copy2(path, dest)
                self._reprocess_saved_path = dest
                print(f"[reprocess] copied to: {dest}")
            else:
                self._reprocess_saved_path = path

            lang = self.lang_var.get()
            prompt = self.prompt_var.get().strip()
            is_stereo = len(channels_data) >= 2

            # Prepare per-channel audio at SAMPLE_RATE
            resampled_channels = []
            for ch_data in channels_data[:2]:
                if sr != SAMPLE_RATE:
                    ratio = SAMPLE_RATE / sr
                    new_len = int(len(ch_data) * ratio)
                    indices = np.linspace(0, len(ch_data) - 1, new_len).astype(int)
                    ch_data = ch_data[indices]
                resampled_channels.append(ch_data)

            # Transcribe mono (best quality — both channels mixed)
            if len(resampled_channels) >= 2:
                mono_data = ((resampled_channels[0].astype(np.int32) + resampled_channels[1].astype(np.int32)) // 2).astype(np.int16)
            else:
                mono_data = resampled_channels[0]
            audio_f32 = mono_data.astype(np.float32) / 32768.0
            self.root.after(0, self._show_progress, "Transcribing...", 30, 100)
            print(f"[reprocess] transcribing {len(audio_f32)/SAMPLE_RATE:.1f}s audio...")
            seg_list = self._transcribe_audio(audio_f32, lang, prompt)

            # Speaker identification
            self.root.after(0, self._show_progress, "Identifying speakers...", 80, 100)
            results = []
            mode = self.processing_var.get()

            if not is_stereo or mode == "Mono (no diarization)":
                # All segments → Speaker 1 (no diarization)
                for start, end, text in seg_list:
                    ts = f"{int(start)//60:02d}:{int(start)%60:02d}"
                    results.append((0, ts, text))
                    print(f"[reprocess] [{ts}] Speaker 1 (mono): {text[:60]}")

            elif mode == "Per-Channel":
                # Transcribe each channel separately, then merge + dedup
                print("[reprocess] Per-Channel: transcribing channels separately...")
                ch0_f32 = resampled_channels[0].astype(np.float32) / 32768.0
                ch1_f32 = resampled_channels[1].astype(np.float32) / 32768.0
                self.root.after(0, self._show_progress, "Transcribing ch0...", 40, 100)
                seg_list_0 = self._transcribe_audio(ch0_f32, lang, prompt)
                self.root.after(0, self._show_progress, "Transcribing ch1...", 60, 100)
                seg_list_1 = self._transcribe_audio(ch1_f32, lang, prompt)
                self.root.after(0, self._show_progress, "Merging...", 80, 100)
                # Merge: ch0 → Speaker 1, ch1 → Speaker 2
                merged = []
                for start, end, text in seg_list_0:
                    merged.append((0, start, end, text))
                for start, end, text in seg_list_1:
                    merged.append((1, start, end, text))
                # Sort by start time
                merged.sort(key=lambda x: x[1])
                # Dedup: remove segments with identical text within overlapping time windows
                seen = []
                for speaker, start, end, text in merged:
                    is_dup = False
                    for s_spk, s_start, s_end, s_text in seen:
                        if s_text == text and abs(s_start - start) < 2.0:
                            is_dup = True
                            break
                    if not is_dup:
                        ts = f"{int(start)//60:02d}:{int(start)%60:02d}"
                        results.append((speaker, ts, text))
                        seen.append((speaker, start, end, text))
                        print(f"[reprocess] [{ts}] Speaker {speaker+1} (per-ch): {text[:60]}")

            else:
                # Energy-based speaker assignment: RMS Energy, Peak-Window, Norm. Peak-Window, Dominance-Flip
                ch_raw = [ch.astype(np.float32) for ch in resampled_channels]

                use_normalization = mode in ("Norm. Peak-Window", "Dominance-Flip")
                use_peak_window = mode in ("Peak-Window", "Norm. Peak-Window", "Dominance-Flip")

                global_rms0 = np.sqrt(np.mean(ch_raw[0] ** 2)) if use_normalization else 1.0
                global_rms1 = np.sqrt(np.mean(ch_raw[1] ** 2)) if use_normalization else 1.0
                if global_rms0 < 1: global_rms0 = 1
                if global_rms1 < 1: global_rms1 = 1
                print(f"[reprocess] mode={mode}, norm={use_normalization}, peak_win={use_peak_window}, baseline: ch0={int(global_rms0)}, ch1={int(global_rms1)}")

                WIN = int(0.1 * SAMPLE_RATE)  # 100ms window

                for start, end, text in seg_list:
                    si = int(start * SAMPLE_RATE)
                    ei = min(int(end * SAMPLE_RATE), len(ch_raw[0]))

                    if use_peak_window and ei - si > WIN:
                        # Find loudest 100ms window
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

                    norm0 = peak0 / global_rms0
                    norm1 = peak1 / global_rms1
                    speaker = 0 if norm0 > norm1 else 1
                    ts = f"{int(start)//60:02d}:{int(start)%60:02d}"
                    results.append((speaker, ts, text))
                    ratio = max(norm0, norm1) / min(norm0, norm1) if min(norm0, norm1) > 0 else 99
                    tag = "conf" if ratio > 1.5 else "close"
                    print(f"[reprocess] [{ts}] Speaker {speaker+1} ({tag}, norm:{norm0:.2f}/{norm1:.2f}, peak:{int(peak0)}/{int(peak1)}): {text[:60]}")

            self.root.after(0, self._show_progress, "Done", 100, 100)
            self.root.after(0, self._on_reprocess_done, path, results, False)
        except Exception as e:
            print(f"[reprocess] ERROR: {e}")
            traceback.print_exc()
            self.root.after(0, self._on_reprocess_error, e)

    def _on_reprocess_done(self, path, results, is_stereo):
        self._reprocessing = False
        self.text.delete("1.0", tk.END)
        has_multiple_speakers = len(set(r[0] for r in results)) > 1
        for speaker, ts, text in results:
            if not text:
                continue
            content = self.text.get("1.0", tk.END).strip()
            if has_multiple_speakers:
                speaker_tag = f"speaker{speaker + 1}"
                if content:
                    self.text.insert(tk.END, "\n")
                self.text.insert(tk.END, f"Speaker {speaker + 1}", speaker_tag)
                self.text.insert(tk.END, f" [{ts}]", "timestamp")
                self.text.insert(tk.END, f": {text}")
            else:
                if content:
                    self.text.insert(tk.END, "\n")
                self.text.insert(tk.END, f"[{ts}]", "timestamp")
                self.text.insert(tk.END, f" {text}")

        # Save transcript next to the file in recordings
        saved_path = getattr(self, "_reprocess_saved_path", path)
        transcript = self.text.get("1.0", tk.END).strip()
        if transcript:
            txt_path = os.path.splitext(saved_path)[0] + ".txt"
            with open(txt_path, "w") as f:
                f.write(transcript)

        self._refresh_recordings_list()
        # Re-select the file that was just reprocessed
        fname = os.path.basename(saved_path)
        for i in range(self.rec_listbox.size()):
            if self.rec_listbox.get(i) == fname:
                self.rec_listbox.selection_set(i)
                self.rec_listbox.see(i)
                self.rerun_btn.config(state=tk.NORMAL)
                break
        self._hide_progress()
        self._set_stopped_ui()
        self._update_status(f"Reprocessed: {os.path.basename(saved_path)}")

    def _on_reprocess_error(self, error):
        self._reprocessing = False
        self._hide_progress()
        self._set_stopped_ui()
        self._update_status(f"Reprocess error: {error}")
        messagebox.showerror("Reprocess Error", str(error))

    # --- Save recording ---

    def _save_recording(self):
        if not self.recorded_frames:
            return
        # Determine hw channels from recorded data
        idx = self.mic_combo.current()
        hw_ch = self._input_devices[idx][2] if 0 <= idx < len(self._input_devices) else 1
        save_stereo = self._device_channels == 2  # auto-detected stereo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(RECORDINGS_DIR, f"rec_{timestamp}.wav")
        raw_data = b"".join(self.recorded_frames)
        if hw_ch >= 2 and not save_stereo:
            # Mix stereo hardware capture to mono for saving
            stereo = np.frombuffer(raw_data, dtype=np.int16)
            mono = ((stereo[0::2].astype(np.int32) + stereo[1::2].astype(np.int32)) // 2).astype(np.int16)
            raw_data = mono.tobytes()
            hw_ch = 1
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(hw_ch)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self._device_samplerate)
            wf.writeframes(raw_data)
        # Save transcript alongside
        transcript = self.text.get("1.0", tk.END).strip()
        if transcript:
            base, _ = os.path.splitext(wav_path)
            txt_path = base + ".txt"
            with open(txt_path, "w") as f:
                f.write(transcript)
        self.current_wav_path = wav_path
        self.recorded_frames = []
        self._refresh_recordings_list()
        # Auto-select the newly saved file
        fname = os.path.basename(wav_path)
        for i in range(self.rec_listbox.size()):
            if self.rec_listbox.get(i) == fname:
                self.rec_listbox.selection_set(i)
                self.rec_listbox.see(i)
                self.rerun_btn.config(state=tk.NORMAL)
                break
        return wav_path

    # --- Mic level meter ---

    def _draw_level(self):
        for i, (canvas, rect) in enumerate([
            (self.level_canvas_0, self.level_rect_0),
            (self.level_canvas_1, self.level_rect_1),
        ]):
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            lvl = self.mic_level[i]
            bar_w = int(w * lvl)
            if lvl < 0.5:
                color = "#4CAF50"
            elif lvl < 0.8:
                color = "#FFC107"
            else:
                color = "#F44336"
            canvas.coords(rect, 0, 0, bar_w, h)
            canvas.itemconfig(rect, fill=color)

    def _update_level(self, data: bytes):
        samples = np.frombuffer(data, dtype=np.int16)
        if len(samples) == 0:
            return
        if self._device_channels == 2 and len(samples) >= 2:
            channels = [samples[0::2].astype(np.float32), samples[1::2].astype(np.float32)]
        else:
            mono = samples.astype(np.float32)
            channels = [mono, mono]  # show same level on both bars
        for i, ch in enumerate(channels):
            rms = np.sqrt(np.mean(ch * ch))
            if rms < 1:
                self.mic_level[i] = 0.0
            else:
                db = 20 * math.log10(rms / 32768)
                self.mic_level[i] = max(0.0, min(1.0, (db + 60) / 55))
        self.root.after(0, self._draw_level)

    # --- Channel bleeding alarm ---

    BLEED_WINDOW = 6         # rolling window size (chunks, ~1.5s at 250ms/chunk)
    BLEED_CORR_THRESH = 0.80 # correlation above this = waveform bleeding
    BLEED_ALARM_RATIO = 0.5  # alarm if ≥50% of recent chunks bleed
    BLEED_RMS_FLOOR = 150    # ignore silence (int16 RMS below this)
    BLEED_CALIBRATION_CHUNKS = 12  # ~3s calibration period
    BLEED_BASELINE_MARGIN = 1.8    # alarm if energy ratio > baseline * margin

    def _check_bleeding(self, left_16k, right_16k):
        """Detect channel bleeding via correlation OR adaptive energy ratio."""
        lf = left_16k.astype(np.float32)
        rf = right_16k.astype(np.float32)
        l_rms = np.sqrt(np.mean(lf * lf))
        r_rms = np.sqrt(np.mean(rf * rf))

        # Skip silence
        peak_rms = max(l_rms, r_rms)
        if peak_rms < self.BLEED_RMS_FLOOR:
            return

        # Method 1: Pearson correlation (catches symmetric bleeding)
        lf_c = lf - lf.mean()
        rf_c = rf - rf.mean()
        denom = np.sqrt(np.sum(lf_c * lf_c) * np.sum(rf_c * rf_c))
        corr = (np.sum(lf_c * rf_c) / denom) if denom > 1e-9 else 0.0
        corr_bleed = corr > self.BLEED_CORR_THRESH

        # Method 2: Adaptive energy ratio
        min_rms = min(l_rms, r_rms)
        energy_ratio = min_rms / peak_rms

        # Calibration phase: collect baseline energy ratios
        if self._bleed_baseline is None:
            self._bleed_calibration.append(energy_ratio)
            if len(self._bleed_calibration) >= self.BLEED_CALIBRATION_CHUNKS:
                # Use median as baseline (robust to outliers)
                sorted_vals = sorted(self._bleed_calibration)
                mid = len(sorted_vals) // 2
                self._bleed_baseline = sorted_vals[mid]
                # Clamp baseline to reasonable range [0.03, 0.60]
                self._bleed_baseline = max(0.03, min(0.60, self._bleed_baseline))
                threshold = self._bleed_baseline * self.BLEED_BASELINE_MARGIN
                print(f"[bleed] calibrated: baseline={self._bleed_baseline:.2f}, threshold={threshold:.2f}")
            else:
                print(f"[bleed] calibrating ({len(self._bleed_calibration)}/{self.BLEED_CALIBRATION_CHUNKS}) ratio={energy_ratio:.2f} corr={corr:.2f}")
                return

        energy_threshold = self._bleed_baseline * self.BLEED_BASELINE_MARGIN
        energy_bleed = energy_ratio > energy_threshold

        is_bleeding = corr_bleed or energy_bleed

        self._bleed_history.append(is_bleeding)
        if len(self._bleed_history) > self.BLEED_WINDOW:
            self._bleed_history = self._bleed_history[-self.BLEED_WINDOW:]

        # Log
        hi = sum(self._bleed_history)
        tag = "CORR" if corr_bleed else ("ENERGY" if energy_bleed else "ok")
        print(f"[bleed] corr={corr:.2f} ratio={energy_ratio:.2f} thr={energy_threshold:.2f} [{tag}] hi={hi}/{len(self._bleed_history)}")

        if len(self._bleed_history) < 2:
            return

        bleed_ratio = hi / len(self._bleed_history)
        should_alarm = bleed_ratio >= self.BLEED_ALARM_RATIO

        if should_alarm != self._bleed_alarm_on:
            self._bleed_alarm_on = should_alarm
            print(f"[bleed] alarm={'ON' if should_alarm else 'OFF'} ({bleed_ratio:.0%})")
            self.root.after(0, self._update_bleed_alarm, should_alarm)

    def _update_bleed_alarm(self, alarm_on):
        if alarm_on:
            if not self._bleed_visible:
                self._bleed_visible = True
                self.bleed_label.config(
                    text="Channel bleeding — separate mics or stop interrupting!",
                )
                self.bleed_frame.pack(fill=tk.X, padx=8, pady=(2, 0),
                                      after=self._bleed_pack_after)
                self._blink_bleed()
        else:
            self._bleed_visible = False
            self.bleed_frame.pack_forget()

    def _blink_bleed(self):
        """Blink the bleeding label background to draw attention."""
        if not self._bleed_visible:
            self.bleed_label.config(bg="#FFE0E0")
            return
        current = self.bleed_label.cget("bg")
        next_bg = "#FFCCCC" if current == "#FFE0E0" else "#FFE0E0"
        self.bleed_label.config(bg=next_bg)
        self.bleed_frame.config(bg=next_bg)
        self.root.after(500, self._blink_bleed)

    def _reset_bleed_state(self):
        """Clear bleeding state (on stop/start)."""
        self._bleed_history.clear()
        self._bleed_alarm_on = False
        self._bleed_visible = False
        self._bleed_calibration.clear()
        self._bleed_baseline = None
        self.bleed_frame.pack_forget()

    # --- Background mic monitor (level meter always active) ---

    def _start_monitor(self):
        """Start a background audio stream just for the level meter."""
        if self.monitoring or self.recording:
            return
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _stop_monitor(self, wait=False):
        self.monitoring = False
        if wait and self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        self.monitor_thread = None

    def _monitor_loop(self):
        dev_sr = self._device_samplerate
        # Use smaller blocksize for monitor (~100ms) for responsive level meter
        blocksize = int(0.1 * dev_sr)
        dev = self.selected_device
        ch = self._device_channels
        print(f"[monitor] starting: device={dev}, channels={ch}, sr={dev_sr}")
        try:
            with sd.RawInputStream(
                samplerate=dev_sr, blocksize=blocksize,
                dtype="int16", channels=ch,
                device=dev,
                callback=self._monitor_callback,
            ):
                print("[monitor] stream opened")
                while self.monitoring and not self.recording:
                    sd.sleep(100)
        except Exception as e:
            print(f"[monitor] ERROR: {e}")
        self.monitoring = False
        print("[monitor] stopped")

    _monitor_log_count = 0
    def _monitor_callback(self, indata, frames, time_info, status):
        if self.monitoring and not self.recording:
            data = bytes(indata)
            self._monitor_log_count += 1
            if self._monitor_log_count <= 3:
                samples = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
                print(f"[monitor] callback #{self._monitor_log_count}: {len(data)} bytes, rms={rms:.1f}")
            self._update_level(data)

    # --- Device management ---

    @staticmethod
    def _query_coreaudio_inputs():
        """Query input device names via macOS system_profiler (no PortAudio reinit needed)."""
        import subprocess
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPAudioDataType", "-detailLevel", "mini"],
                timeout=3, text=True, stderr=subprocess.DEVNULL,
            )
            names = set()
            lines = out.splitlines()
            for i, line in enumerate(lines):
                if "Input Channels:" in line:
                    # Device name is a few lines before — it's a line ending with ":"
                    # but NOT a key-value property line (those have ": " with a value)
                    for j in range(i - 1, max(i - 6, -1), -1):
                        stripped = lines[j].strip()
                        # Device name lines look like "MacBook Pro Microphone:" (no value after colon)
                        if stripped.endswith(":") and ": " not in stripped:
                            names.add(stripped.rstrip(":"))
                            break
            return names
        except Exception:
            return None

    def _poll_devices(self):
        """Periodically check for added/removed audio devices."""
        if self._poll_paused:
            self.root.after(2000, self._poll_devices)
            return
        try:
            # Use CoreAudio query (doesn't interfere with active PortAudio streams)
            current = self._query_coreaudio_inputs()
            if current is None:
                # Fallback: query PortAudio directly (stale but safe)
                current = set(d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0)
            if hasattr(self, "_known_devices") and current != self._known_devices:
                print(f"[devices] change detected: {self._known_devices} → {current}")
                self._known_devices = current
                self._refresh_devices()
            else:
                self._known_devices = current
        except Exception:
            pass
        self.root.after(2000, self._poll_devices)

    def _refresh_devices(self):
        """Refresh device list. Use Refresh button after plugging in new hardware."""
        self._stop_monitor()
        threading.Thread(target=self._do_refresh_devices_bg, daemon=True).start()

    def _do_refresh_devices_bg(self):
        # Wait for monitor stream to fully close, then reinit PortAudio
        time.sleep(0.3)
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass
        devices = get_input_devices()
        self.root.after(0, self._apply_device_list, devices)

    def _apply_device_list(self, devices):
        old_selection = self.mic_var.get()
        self._input_devices = devices
        for dev_idx, name, ch, sr in devices:
            print(f"[devices] #{dev_idx}: {name} (ch={ch}, sr={sr})")
        names = [name for _, name, _, _ in self._input_devices]
        self.mic_combo["values"] = names
        if not names:
            return
        if old_selection in names:
            idx = names.index(old_selection)
            self.mic_combo.current(idx)
            self._select_device(idx)
        else:
            default_idx = sd.default.device[0]
            selected = 0
            for i, (dev_idx, _, _, _) in enumerate(self._input_devices):
                if dev_idx == default_idx:
                    selected = i
                    break
            self.mic_combo.current(selected)
            self._select_device(selected)
        if not self.recording:
            self._start_monitor()

    def _select_device(self, idx):
        """Update device settings from the device list entry."""
        dev_idx, _, max_ch, sr = self._input_devices[idx]
        self.selected_device = dev_idx
        self._device_samplerate = sr
        self._device_channels = max_ch  # auto-detect will decide mono/stereo

    def _on_mic_changed(self):
        idx = self.mic_combo.current()
        if 0 <= idx < len(self._input_devices):
            self._select_device(idx)
        if self.recording:
            self._on_stop()
            self._update_status("Microphone changed — click Record to resume")
        else:
            # Restart monitor with new device
            self._stop_monitor()
            self.root.after(300, self._start_monitor)

    # --- UI helpers ---

    def _update_status(self, text: str):
        self.status_label.config(text=text)

    def _set_recording_ui(self):
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL, text="Pause")
        self.stop_btn.config(state=tk.NORMAL)
        self.reprocess_btn.config(state=tk.DISABLED)

    def _set_paused_ui(self):
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL, text="Resume")
        self.stop_btn.config(state=tk.NORMAL)
        self.reprocess_btn.config(state=tk.DISABLED)

    def _set_stopped_ui(self):
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="Pause")
        self.stop_btn.config(state=tk.DISABLED)
        self.reprocess_btn.config(state=tk.NORMAL)

    def _set_loading_ui(self):
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.reprocess_btn.config(state=tk.DISABLED)

    # --- Auto-preload ---

    def _auto_preload_model(self):
        """Pre-warm the default MLX model in background so Record starts instantly."""
        size = self.size_var.get()
        backend, model_id = model_manager.get_model_info(size)
        if self._loaded_key == size:
            return  # already loaded
        if not model_manager.is_model_cached(size):
            return  # don't auto-download
        print(f"[preload] warming up {size} in background...")
        self._update_status(f"Pre-loading {size}...")
        threading.Thread(target=self._preload_model_sync, args=(size,), daemon=True).start()

    def _preload_model_sync(self, size):
        try:
            backend, model_id = model_manager.get_model_info(size)
            import mlx_whisper
            dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
            mlx_whisper.transcribe(dummy, path_or_hf_repo=model_id)
            self.root.after(0, self._on_preload_done, size, backend, model_id)
        except Exception as e:
            print(f"[preload] failed: {e}")
            self.root.after(0, self._update_status, "Ready — select model and click Record")

    def _on_preload_done(self, size, backend, model_id):
        self.model = model_id
        self._backend = backend
        self._loaded_key = size
        self._update_status(f"Ready — {size} loaded")
        print(f"[preload] {size} ready")

    # --- Button actions ---

    def _on_start(self):
        print(f"[ui] Record pressed, loaded_key={self._loaded_key!r}")
        if self.playing:
            self._stop_playback()
        size = self.size_var.get()
        print(f"[ui] size={size!r}, need_load={self._loaded_key != size}")
        if self._loaded_key != size:
            self._load_model(size)
            return
        self._begin_audio_capture()

    def _on_pause(self):
        if self.paused:
            self.paused = False
            self._set_recording_ui()
            self._update_status("Listening...")
        else:
            self.paused = True
            self._set_paused_ui()
            self._remove_partial()
            self._update_status("Paused")
            self.mic_level = [0.0, 0.0]
            self._draw_level()

    def _on_stop(self):
        was_recording = self.recording
        self.recording = False
        self.paused = False
        # Restore hardware channel count so monitor and next recording get the right value
        idx = self.mic_combo.current()
        if 0 <= idx < len(self._input_devices):
            self._device_channels = self._input_devices[idx][2]
        self._set_stopped_ui()
        self._reset_bleed_state()
        self._commit_partial()
        if was_recording and self.recorded_frames:
            path = self._save_recording()
            self._update_status(f"Saved: {os.path.basename(path)}")
        else:
            self._update_status("Stopped")
        # Restart background mic monitor for level meter
        self.root.after(300, self._start_monitor)

    # --- Model loading ---

    def _load_model(self, size: str):
        print(f"[model] _load_model called: {size!r}")
        # Stop monitor stream AND device polling BEFORE loading model —
        # active PortAudio streams and system_profiler subprocess both
        # block MLX Metal GPU initialization (import mlx_whisper hangs)
        self._poll_paused = True
        self._stop_monitor(wait=True)
        self._set_loading_ui()
        cached = model_manager.is_model_cached(size)
        if cached:
            msg = f"Loading {size} model..."
        else:
            msg = f"Downloading & loading {size} model (first time)..."
        self._update_status(msg)
        self._show_progress(msg, 20, 100)
        threading.Thread(target=self._load_model_sync, args=(size,), daemon=True).start()

    def _load_model_sync(self, size: str):
        print(f"[model] _load_model_sync thread started: {size!r}")
        try:
            print(f"[model] getting model info...")
            backend, model_id = model_manager.get_model_info(size)
            print(f"[model] got info: backend={backend!r}, model_id={model_id!r}")
            print(f"[model] importing mlx_whisper...")
            import mlx_whisper
            print(f"[model] mlx_whisper imported, warming up {model_id!r}...")
            dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
            try:
                mlx_whisper.transcribe(dummy, path_or_hf_repo=model_id)
            except Exception as e:
                print(f"[model] mlx warmup failed ({e}), retrying once...")
                import gc; gc.collect()
                time.sleep(1)
                mlx_whisper.transcribe(dummy, path_or_hf_repo=model_id)
            model = model_id
            print(f"[model] mlx-whisper ready: {model_id}")
            self.root.after(0, self._on_model_loaded, size, backend, model, None)
        except Exception as e:
            print(f"[model] ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, self._on_model_loaded, size, None, None, e)

    def _on_model_loaded(self, size: str, backend, model, error):
        print(f"[model] _on_model_loaded called: size={size!r}, error={error}")
        self._poll_paused = False
        self._hide_progress()
        if error:
            self._set_stopped_ui()
            self._update_status(f"Failed to load model: {error}")
            messagebox.showerror("Model Error", str(error))
            return
        self.model = model
        self._backend = backend
        self._loaded_key = size
        self._update_status(f"Model ready: {size}")
        try:
            self._begin_audio_capture()
        except Exception as e:
            print(f"[audio] ERROR starting capture: {e}")
            import traceback; traceback.print_exc()
            self._set_stopped_ui()
            self._update_status(f"Audio error: {e}")
            messagebox.showerror("Audio Error", str(e))

    # --- Audio capture ---

    def _begin_audio_capture(self):
        # Save monitor thread ref before clearing it
        old_monitor = self.monitor_thread
        self._stop_monitor()
        self.recording = True
        self.paused = False
        self.recorded_frames = []
        self._reset_bleed_state()
        if _resemblyzer_ready.is_set():
            try:
                self._speaker_tracker = SpeakerTracker()
                print("[speaker] SpeakerTracker ready")
            except Exception as e:
                print(f"[speaker] WARNING: SpeakerTracker init failed ({e}), continuing without diarization")
                self._speaker_tracker = None
        else:
            print("[speaker] resemblyzer still loading, diarization will be skipped this session")
            self._speaker_tracker = None
        self._recording_start_time = time.time()
        self._set_recording_ui()

        # Create new recording entry and clear text area
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._pending_rec_name = f"rec_{timestamp}"
        self.text.delete("1.0", tk.END)
        self.rec_listbox.selection_clear(0, tk.END)
        self.rerun_btn.config(state=tk.DISABLED)
        self._update_status(f"Recording: {self._pending_rec_name}")

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        self.audio_thread = threading.Thread(
            target=self._audio_loop, args=(old_monitor,), daemon=True,
        )
        self.audio_thread.start()

    def _resample_chunk(self, data_int16, from_sr, to_sr):
        """Resample an int16 numpy array from from_sr to to_sr."""
        if from_sr == to_sr:
            return data_int16
        ratio = to_sr / from_sr
        new_len = int(len(data_int16) * ratio)
        indices = np.linspace(0, len(data_int16) - 1, new_len).astype(int)
        return data_int16[indices]

    @staticmethod
    def _channels_correlated(left, right):
        """Return True if two int16 arrays carry essentially the same audio."""
        if len(left) < 100 or len(right) < 100:
            return True
        lf = left.astype(np.float32)
        rf = right.astype(np.float32)
        lf -= lf.mean()
        rf -= rf.mean()
        denom = np.sqrt(np.sum(lf * lf) * np.sum(rf * rf))
        if denom < 1e-9:
            return True  # both silent
        corr = np.sum(lf * rf) / denom
        return corr > 0.5

    @staticmethod
    def _channels_dominance_flip(left, right):
        """Return which channel is louder: 0=left, 1=right."""
        lf = left.astype(np.float32)
        rf = right.astype(np.float32)
        l_rms = np.sqrt(np.mean(lf ** 2))
        r_rms = np.sqrt(np.mean(rf ** 2))
        return 0 if l_rms > r_rms else 1

    def _audio_loop(self, old_monitor=None):
        # Wait for monitor stream to fully close before opening recording stream
        if old_monitor and old_monitor.is_alive():
            old_monitor.join(timeout=2.0)
        # Extra delay to let PortAudio fully release the device
        time.sleep(0.3)
        # Read hardware channel count from the device list (not self._device_channels
        # which gets modified to 1/2 during recording for display purposes)
        idx = self.mic_combo.current()
        hw_channels = self._input_devices[idx][2] if 0 <= idx < len(self._input_devices) else self._device_channels
        dev_sr = self._device_samplerate
        needs_resample = dev_sr != SAMPLE_RATE
        blocksize = int(BLOCK_SIZE * dev_sr / SAMPLE_RATE)
        # Start in mono mode for display; auto-detect will switch to stereo if needed
        self._device_channels = 1
        dev_name = self._input_devices[idx][1] if 0 <= idx < len(self._input_devices) else "unknown"
        print(f"[audio_loop] starting, device={self.selected_device} ({dev_name}), hw_channels={hw_channels}, sr={dev_sr}")
        self._transcribing = set()  # set of channel indices currently transcribing

        # Auto-detect mono/stereo continuously per-chunk
        use_stereo = False
        is_hw_stereo = hw_channels >= 2
        diverge_count = 0  # consecutive non-correlated chunks
        DIVERGE_THRESHOLD = 8  # require N consecutive divergent chunks to switch to stereo
        converge_count = 0
        CONVERGE_THRESHOLD = 8  # require N consecutive correlated chunks to switch back to mono

        # Processing mode affects stereo detection strategy
        from collections import deque
        processing_mode = self.processing_var.get()
        dominance_history = deque(maxlen=16)  # for Dominance-Flip mode

        # Override stereo behavior based on processing mode
        if processing_mode == "Mono (no diarization)":
            is_hw_stereo = False  # force mono regardless of hardware
        elif processing_mode == "Per-Channel" and hw_channels >= 2:
            # Force stereo immediately
            use_stereo = True

        try:
            # Start with mono transcription; upgrade to stereo if channels diverge
            n_channels = 1
            if use_stereo:
                n_channels = 2
                self._device_channels = 2
            channels = [ChannelState() for _ in range(n_channels)]

            with sd.RawInputStream(
                samplerate=dev_sr, blocksize=blocksize,
                dtype="int16", channels=hw_channels, device=self.selected_device,
                callback=self._audio_callback,
            ):
                print(f"[audio_loop] stream opened (processing={processing_mode})")
                while self.recording:
                    try:
                        data = self.audio_queue.get(timeout=0.05)
                    except queue.Empty:
                        data = None

                    if self.paused:
                        continue

                    now = time.time()

                    if data:
                        self.recorded_frames.append(data)
                        self._update_level(data)

                        raw = np.frombuffer(data, dtype=np.int16)

                        if is_hw_stereo:
                            left_raw = raw[0::2]
                            right_raw = raw[1::2]

                            # Resample each channel
                            if needs_resample:
                                left_16k = self._resample_chunk(left_raw, dev_sr, SAMPLE_RATE)
                                right_16k = self._resample_chunk(right_raw, dev_sr, SAMPLE_RATE)
                            else:
                                left_16k = left_raw
                                right_16k = right_raw

                            # Check for channel bleeding
                            self._check_bleeding(left_16k, right_16k)

                            if processing_mode == "Dominance-Flip":
                                # Track which channel is dominant per chunk
                                dominant = self._channels_dominance_flip(left_16k, right_16k)
                                dominance_history.append(dominant)
                                if len(dominance_history) >= 2:
                                    flips = sum(1 for i in range(1, len(dominance_history))
                                                if dominance_history[i] != dominance_history[i - 1])
                                    flip_rate = flips / (len(dominance_history) - 1)
                                else:
                                    flip_rate = 0.0

                                if not use_stereo:
                                    if flip_rate > 0.25:
                                        diverge_count += 1
                                    else:
                                        diverge_count = 0
                                    if diverge_count >= DIVERGE_THRESHOLD:
                                        use_stereo = True
                                        n_channels = 2
                                        self._device_channels = 2
                                        channels.append(ChannelState())
                                        converge_count = 0
                                        print("[audio_loop] switched to stereo (dominance flips detected)")
                                        self.root.after(0, self._update_status, "Recording — stereo (2 speakers)")
                                else:
                                    if flip_rate < 0.1:
                                        converge_count += 1
                                    else:
                                        converge_count = 0
                                    if converge_count >= CONVERGE_THRESHOLD:
                                        use_stereo = False
                                        n_channels = 1
                                        self._device_channels = 1
                                        channels = [channels[0]]
                                        diverge_count = 0
                                        print("[audio_loop] switched back to mono (no dominance flips)")
                                        self.root.after(0, self._update_status, "Recording — mono")

                            elif processing_mode == "Per-Channel":
                                pass  # stereo forced on, no detection needed

                            else:
                                # Correlation-based detection (RMS Energy, Peak-Window, Norm. Peak-Window)
                                correlated = self._channels_correlated(left_16k, right_16k)

                                if not use_stereo:
                                    if not correlated:
                                        diverge_count += 1
                                    else:
                                        diverge_count = 0
                                    if diverge_count >= DIVERGE_THRESHOLD:
                                        use_stereo = True
                                        n_channels = 2
                                        self._device_channels = 2
                                        channels.append(ChannelState())
                                        converge_count = 0
                                        print("[audio_loop] switched to stereo (2 speakers detected)")
                                        self.root.after(0, self._update_status, "Recording — stereo (2 speakers)")
                                else:
                                    if correlated:
                                        converge_count += 1
                                    else:
                                        converge_count = 0
                                    if converge_count >= CONVERGE_THRESHOLD:
                                        use_stereo = False
                                        n_channels = 1
                                        self._device_channels = 1
                                        channels = [channels[0]]
                                        diverge_count = 0
                                        print("[audio_loop] switched back to mono (channels correlated)")
                                        self.root.after(0, self._update_status, "Recording — mono")

                            if use_stereo:
                                chan_data = [left_16k.tobytes(), right_16k.tobytes()]
                            else:
                                # Mix to mono
                                mono = ((left_16k.astype(np.int32) + right_16k.astype(np.int32)) // 2).astype(np.int16)
                                chan_data = [mono.tobytes()]
                        else:
                            # Single-channel device
                            if needs_resample:
                                resampled = self._resample_chunk(raw, dev_sr, SAMPLE_RATE)
                                chan_data = [resampled.tobytes()]
                            else:
                                chan_data = [data]

                        for ch in range(n_channels):
                            cs = channels[ch]
                            cs.working_buffer.extend(chan_data[ch])

                            # Check if this chunk has speech
                            chunk_int16 = np.frombuffer(chan_data[ch], dtype=np.int16)
                            chunk_rms = np.sqrt(np.mean((chunk_int16.astype(np.float32) / 32768.0) ** 2))
                            cs.energy_sum += chunk_rms
                            cs.energy_count += 1
                            if chunk_rms >= SPEECH_RMS_THRESHOLD:
                                cs.had_speech = True
                                cs.last_speech_time = now
                                cs.speech_samples_in_buffer += len(chan_data[ch]) // 2

                    # Per-channel transcription decisions
                    # Snapshot energy averages BEFORE processing so ch0 reset doesn't affect ch1 check
                    if self._bleed_baseline is not None:
                        stereo_suppress_ratio = self._bleed_baseline * 2.0
                    else:
                        stereo_suppress_ratio = 0.15
                    ch_energies = [
                        channels[ch].energy_sum / max(1, channels[ch].energy_count)
                        for ch in range(n_channels)
                    ]
                    for ch in range(n_channels):
                        cs = channels[ch]
                        buffer_seconds = (len(cs.working_buffer) // 2) / SAMPLE_RATE

                        should_transcribe = False
                        is_commit = False

                        if buffer_seconds < 0.3 or ch in self._transcribing:
                            pass
                        elif buffer_seconds >= COMMIT_SECONDS:
                            should_transcribe = True
                            is_commit = True
                        elif cs.had_speech and now - cs.last_speech_time >= SILENCE_COMMIT:
                            should_transcribe = True
                            is_commit = True
                        elif (cs.had_speech
                              and cs.speech_samples_in_buffer > 0
                              and now - cs.last_transcribe_time >= PARTIAL_INTERVAL):
                            should_transcribe = True

                        # Stereo energy gate: suppress the quiet channel (skip when bleeding — both go to Unrecognized)
                        if should_transcribe and n_channels == 2 and not self._bleed_alarm_on:
                            my_avg = ch_energies[ch]
                            other_avg = ch_energies[1 - ch]
                            if other_avg > 0 and my_avg / other_avg < stereo_suppress_ratio:
                                if is_commit:
                                    print(f"[transcribe] suppressed ch{ch} (energy {my_avg:.4f} vs {other_avg:.4f}, ratio={my_avg/other_avg:.2f})")
                                should_transcribe = False
                                # Still clear buffer on commit to prevent accumulation
                                if is_commit:
                                    cs.working_buffer.clear()
                                    cs.had_speech = False
                                    cs.energy_sum = 0.0
                                    cs.energy_count = 0

                        if should_transcribe:
                            cs.last_transcribe_time = now
                            buf_copy = bytes(cs.working_buffer)
                            self._transcribing.add(ch)
                            threading.Thread(
                                target=self._transcribe_working,
                                args=(buf_copy, is_commit, ch, n_channels > 1),
                                daemon=True,
                            ).start()
                            cs.speech_samples_in_buffer = 0
                            if is_commit:
                                cs.working_buffer.clear()
                                cs.had_speech = False
                                cs.energy_sum = 0.0
                                cs.energy_count = 0

                # Flush remaining working buffers on stop
                for ch in range(n_channels):
                    cs = channels[ch]
                    if len(cs.working_buffer) > 0 and cs.had_speech:
                        # Stereo energy gate on flush too (skip when bleeding)
                        suppress = False
                        if n_channels == 2 and not self._bleed_alarm_on:
                            my_avg = cs.energy_sum / max(1, cs.energy_count)
                            other_avg = channels[1 - ch].energy_sum / max(1, channels[1 - ch].energy_count)
                            flush_suppress = self._bleed_baseline * 2.0 if self._bleed_baseline is not None else 0.15
                            if other_avg > 0 and my_avg / other_avg < flush_suppress:
                                suppress = True
                                print(f"[transcribe] suppressed flush ch{ch} (energy ratio={my_avg/other_avg:.2f})")
                        if not suppress:
                            self._transcribe_working(bytes(cs.working_buffer), True, ch, n_channels > 1)
                    cs.working_buffer.clear()

        except Exception as e:
            self.root.after(0, self._on_audio_error, e)

    def _transcribe_working(self, audio_bytes: bytes, commit: bool, channel: int = 0, is_stereo: bool = False):
        """Transcribe the working buffer. If commit=True, append as final text."""
        import time as _time
        t0 = _time.time()
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        duration = len(audio_float32) / SAMPLE_RATE

        ch_label = f"ch{channel}" if is_stereo else ""
        print(f"[transcribe] {'COMMIT' if commit else 'partial'}{ch_label}: {duration:.1f}s audio ({self._backend})")

        try:
            lang = self.lang_var.get()
            prompt = self.prompt_var.get().strip()

            with self._transcribe_lock:
                import mlx_whisper
                kwargs = {
                    "path_or_hf_repo": self.model,
                    "condition_on_previous_text": False,
                    "no_speech_threshold": 0.5,
                    "compression_ratio_threshold": 1.8,
                }
                if lang and lang != "auto":
                    kwargs["language"] = lang
                if prompt:
                    kwargs["initial_prompt"] = prompt
                result = mlx_whisper.transcribe(audio_float32, **kwargs)
                segments = result.get("segments", [])
                text_parts = []
                for seg in segments:
                    if seg.get("no_speech_prob", 0) < 0.5:
                        text_parts.append(seg["text"].strip())
                text = " ".join(text_parts).strip()

            elapsed = _time.time() - t0
            print(f"[transcribe] {text!r} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"[transcribe] ERROR: {e}")
            self._transcribing.discard(channel)
            return

        self._transcribing.discard(channel)
        if commit:
            if text:
                # Identify speaker from voice characteristics
                speaker = 0
                if self._speaker_tracker and not is_stereo:
                    speaker = self._speaker_tracker.identify(audio_float32, SAMPLE_RATE)
                    print(f"[speaker] → Speaker {speaker + 1}")
                # Compute timestamp relative to recording start
                elapsed = _time.time() - getattr(self, '_recording_start_time', _time.time())
                ts = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
                self.root.after(0, self._append_final, text, channel, speaker, ts, is_stereo)
            else:
                self.root.after(0, self._remove_partial, channel)
        else:
            # Show as updating partial (replaces previous partial)
            self.root.after(0, self._show_partial, text if text else "", channel, is_stereo)

    def _audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            self.audio_queue.put(bytes(indata))

    # --- Text display ---

    def _append_final(self, text: str, channel: int = 0, speaker: int = -1, timestamp: str = "", is_stereo: bool = False):
        self._remove_partial(channel)
        content = self.text.get("1.0", tk.END).rstrip("\n")

        # Determine speaker index: from hardware stereo channel or software diarization
        # When bleeding alarm is on, speaker is ambiguous — label as Unrecognized
        if is_stereo and self._bleed_alarm_on:
            spk = -2  # unrecognized
        elif is_stereo:
            spk = channel
        elif speaker >= 0:
            spk = speaker
        else:
            spk = -1

        if spk == -2:
            if content:
                self.text.insert(tk.END, "\n")
            self.text.insert(tk.END, "Unrecognized", "speaker_unknown")
            if timestamp:
                self.text.insert(tk.END, f" [{timestamp}]", "timestamp")
            self.text.insert(tk.END, f": {text}")
        elif spk >= 0:
            label = f"Speaker {spk + 1}"
            speaker_tag = f"speaker{spk + 1}"
            if content:
                self.text.insert(tk.END, "\n")
            self.text.insert(tk.END, label, speaker_tag)
            if timestamp:
                self.text.insert(tk.END, f" [{timestamp}]", "timestamp")
            self.text.insert(tk.END, f": {text}")
        else:
            if content:
                self.text.insert(tk.END, " ")
            self.text.insert(tk.END, text)
        self.text.see(tk.END)

    def _show_partial(self, text: str, channel: int = 0, is_stereo: bool = False):
        self._remove_partial(channel)
        if text:
            partial_tag = f"partial_{channel}" if is_stereo else "partial"
            if is_stereo and self._bleed_alarm_on:
                self.text.insert(tk.END, "\nUnrecognized: ", (partial_tag, "speaker_unknown"))
                self.text.insert(tk.END, text, partial_tag)
            elif is_stereo:
                label = f"Speaker {channel + 1}: "
                speaker_tag = f"speaker{channel + 1}"
                self.text.insert(tk.END, "\n" + label, (partial_tag, speaker_tag))
                self.text.insert(tk.END, text, partial_tag)
            else:
                self.text.insert(tk.END, " " + text, partial_tag)
            self.text.see(tk.END)

    def _commit_partial(self, channel=None):
        """Promote partial (gray) text to final — keep text, remove tag."""
        if channel is not None:
            tags = [f"partial_{channel}"] if self._device_channels > 1 else ["partial"]
        else:
            tags = ["partial", "partial_0", "partial_1"]
        for tag in tags:
            ranges = self.text.tag_ranges(tag)
            if ranges:
                for i in range(0, len(ranges), 2):
                    self.text.tag_remove(tag, ranges[i], ranges[i + 1])

    def _remove_partial(self, channel=None):
        if channel is not None:
            tags = [f"partial_{channel}"] if self._device_channels > 1 else ["partial"]
        else:
            tags = ["partial", "partial_0", "partial_1"]
        for tag in tags:
            ranges = self.text.tag_ranges(tag)
            if ranges:
                # Delete in reverse to preserve earlier indices
                for i in range(len(ranges) - 2, -1, -2):
                    self.text.delete(ranges[i], ranges[i + 1])

    def _on_audio_error(self, error):
        self.recording = False
        self.paused = False
        try:
            self.root.after(0, self._set_stopped_ui)
            self.root.after(0, self._update_status, f"Audio error: {error}")
            self.root.after(0, lambda: messagebox.showerror("Audio Error", str(error)))
        except tk.TclError:
            pass  # Window already destroyed

    def _on_close(self):
        """Clean shutdown — stop all audio streams before destroying the window."""
        self.recording = False
        self.paused = False
        self._stop_monitor()
        if self.playing:
            sd.stop()
            self.playing = False
        # Give audio threads a moment to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=0.5)
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk::mac::useThemedToplevel", True)
    except tk.TclError:
        pass
    WhisperTranscribe(root)
    root.mainloop()


if __name__ == "__main__":
    main()
