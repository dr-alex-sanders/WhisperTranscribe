"""WhisperTranscribe — Lightweight macOS Speech-to-Text App."""

import math
import os
import queue
import struct
import threading
import time
import wave
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

import model_manager

SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # ~250ms chunks at 16kHz
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
            inputs.append((i, d["name"]))
    return inputs


class WhisperTranscribe:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("WhisperTranscribe")
        self.root.geometry("850x600")
        self.root.minsize(700, 450)

        self.model = None
        self.recording = False
        self.paused = False
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        self.selected_device = None
        self._loaded_key = None
        self.mic_level = 0.0

        # Recording to file
        self.recorded_frames = []
        self.current_wav_path = None

        # Playback
        self.playing = False
        self.play_thread = None

        # Background mic monitor (for level meter when not recording)
        self.monitoring = False
        self.monitor_thread = None

        os.makedirs(RECORDINGS_DIR, exist_ok=True)

        self._build_ui()
        self._refresh_devices()
        self._refresh_recordings_list()
        self._update_status("Select model size, then click Record")

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
        paned.add(text_frame, stretch="always", minsize=300)

        # Right: recordings panel
        rec_frame = tk.Frame(paned)
        tk.Label(rec_frame, text="Recordings", font=("Helvetica", 13, "bold")).pack(anchor=tk.W, padx=4, pady=(4, 2))
        list_frame = tk.Frame(rec_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.rec_listbox = tk.Listbox(
            list_frame, font=("Helvetica", 12), activestyle="none",
            selectbackground="#007AFF", selectforeground="white",
        )
        rec_scroll = ttk.Scrollbar(list_frame, command=self.rec_listbox.yview)
        self.rec_listbox.configure(yscrollcommand=rec_scroll.set)
        rec_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.rec_listbox.pack(fill=tk.BOTH, expand=True)
        self.rec_listbox.bind("<Double-1>", lambda e: self._play_selected())

        # Play/Stop play buttons for recordings
        play_bar = tk.Frame(rec_frame)
        play_bar.pack(fill=tk.X, pady=(4, 4), padx=4)
        self.play_btn = tk.Button(
            play_bar, text="Play", width=8, font=("Helvetica", 12),
            command=self._play_selected, fg="#007AFF",
        )
        self.play_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.stop_play_btn = tk.Button(
            play_bar, text="Stop", width=8, font=("Helvetica", 12),
            command=self._stop_playback, state=tk.DISABLED, fg="#CC0000",
        )
        self.stop_play_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.delete_btn = tk.Button(
            play_bar, text="Delete", width=8, font=("Helvetica", 12),
            command=self._delete_selected,
        )
        self.delete_btn.pack(side=tk.LEFT)

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

        # --- Mic level indicator ---
        level_bar = tk.Frame(self.root)
        level_bar.pack(fill=tk.X, padx=8, pady=(4, 0))

        tk.Label(level_bar, text="Level:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.level_canvas = tk.Canvas(level_bar, height=16, bg="#E0E0E0", highlightthickness=0)
        self.level_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self.level_rect = self.level_canvas.create_rectangle(0, 0, 0, 16, fill="#4CAF50", width=0)
        self.level_canvas.bind("<Configure>", lambda e: self._draw_level())

        # --- Bottom toolbar ---
        toolbar = tk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=8, pady=8)

        self.start_btn = tk.Button(
            toolbar, text="Record", width=8, command=self._on_start,
            font=("Helvetica", 13), fg="#CC0000",
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.pause_btn = tk.Button(
            toolbar, text="Pause", width=8, command=self._on_pause,
            font=("Helvetica", 13), state=tk.DISABLED,
        )
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.stop_btn = tk.Button(
            toolbar, text="Stop", width=8, command=self._on_stop,
            font=("Helvetica", 13), state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(toolbar, text="Model:", font=("Helvetica", 12)).pack(side=tk.LEFT)
        self.size_var = tk.StringVar(value=model_manager.DEFAULT_SIZE)
        ttk.Combobox(
            toolbar, textvariable=self.size_var, values=SIZES,
            state="readonly", width=16, font=("Helvetica", 12),
        ).pack(side=tk.LEFT, padx=(4, 8))

        tk.Label(toolbar, text="Lang:", font=("Helvetica", 12)).pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value="ru")
        ttk.Combobox(
            toolbar, textvariable=self.lang_var,
            values=["ru", "en", "es", "fr", "de", "zh", "ja", "ko", "pt", "it", "auto"],
            state="readonly", width=5, font=("Helvetica", 12),
        ).pack(side=tk.LEFT, padx=(4, 12))

        self.status_label = tk.Label(
            toolbar, text="", font=("Helvetica", 11), fg="gray", anchor=tk.W,
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

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
            [f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".wav")],
            reverse=True,
        )
        for f in files:
            self.rec_listbox.insert(tk.END, f)

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
            with wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                data = wf.readframes(wf.getnframes())
            audio = np.frombuffer(data, dtype=np.int16)
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
            txt_path = path.replace(".wav", ".txt")
            if os.path.exists(txt_path):
                os.remove(txt_path)
        except OSError:
            pass
        self._refresh_recordings_list()

    # --- Save recording ---

    def _save_recording(self):
        if not self.recorded_frames:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(RECORDINGS_DIR, f"rec_{timestamp}.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(self.recorded_frames))
        # Save transcript alongside
        transcript = self.text.get("1.0", tk.END).strip()
        if transcript:
            txt_path = wav_path.replace(".wav", ".txt")
            with open(txt_path, "w") as f:
                f.write(transcript)
        self.current_wav_path = wav_path
        self.recorded_frames = []
        self._refresh_recordings_list()
        return wav_path

    # --- Mic level meter ---

    def _draw_level(self):
        w = self.level_canvas.winfo_width()
        h = self.level_canvas.winfo_height()
        bar_w = int(w * self.mic_level)
        if self.mic_level < 0.5:
            color = "#4CAF50"
        elif self.mic_level < 0.8:
            color = "#FFC107"
        else:
            color = "#F44336"
        self.level_canvas.coords(self.level_rect, 0, 0, bar_w, h)
        self.level_canvas.itemconfig(self.level_rect, fill=color)

    def _update_level(self, data: bytes):
        n_samples = len(data) // 2
        if n_samples == 0:
            return
        samples = struct.unpack(f"<{n_samples}h", data)
        rms = math.sqrt(sum(s * s for s in samples) / n_samples)
        if rms < 1:
            level = 0.0
        else:
            db = 20 * math.log10(rms / 32768)
            level = max(0.0, min(1.0, (db + 60) / 55))
        self.mic_level = level
        self.root.after(0, self._draw_level)

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
        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                dtype="int16", channels=1, device=self.selected_device,
                callback=self._monitor_callback,
            ):
                while self.monitoring and not self.recording:
                    sd.sleep(100)
        except Exception:
            pass
        self.monitoring = False

    def _monitor_callback(self, indata, frames, time_info, status):
        if self.monitoring and not self.recording:
            self._update_level(bytes(indata))

    # --- Device management ---

    def _refresh_devices(self):
        """Refresh device list. Use Refresh button after plugging in new hardware."""
        self._stop_monitor()
        threading.Thread(target=self._do_refresh_devices_bg, daemon=True).start()

    def _do_refresh_devices_bg(self):
        # Wait for monitor stream to fully close
        time.sleep(0.3)
        # Force PortAudio to see new hardware
        try:
            sd._lib.Pa_Terminate()
            sd._lib.Pa_Initialize()
        except Exception:
            pass
        devices = get_input_devices()
        self.root.after(0, self._apply_device_list, devices)

    def _apply_device_list(self, devices):
        old_selection = self.mic_var.get()
        self._input_devices = devices
        names = [name for _, name in self._input_devices]
        self.mic_combo["values"] = names
        if not names:
            return
        if old_selection in names:
            idx = names.index(old_selection)
            self.mic_combo.current(idx)
            self.selected_device = self._input_devices[idx][0]
        else:
            default_idx = sd.default.device[0]
            selected = 0
            for i, (dev_idx, _) in enumerate(self._input_devices):
                if dev_idx == default_idx:
                    selected = i
                    break
            self.mic_combo.current(selected)
            self.selected_device = self._input_devices[selected][0]
        if not self.recording:
            self._start_monitor()

    def _on_mic_changed(self):
        idx = self.mic_combo.current()
        if 0 <= idx < len(self._input_devices):
            self.selected_device = self._input_devices[idx][0]
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

    def _set_paused_ui(self):
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL, text="Resume")
        self.stop_btn.config(state=tk.NORMAL)

    def _set_stopped_ui(self):
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="Pause")
        self.stop_btn.config(state=tk.DISABLED)

    def _set_loading_ui(self):
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)

    # --- Button actions ---

    def _on_start(self):
        if self.playing:
            self._stop_playback()
        size = self.size_var.get()
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
            self.mic_level = 0.0
            self._draw_level()

    def _on_stop(self):
        was_recording = self.recording
        self.recording = False
        self.paused = False
        self._set_stopped_ui()
        self._remove_partial()
        if was_recording and self.recorded_frames:
            path = self._save_recording()
            self._update_status(f"Saved: {os.path.basename(path)}")
        else:
            self._update_status("Stopped")
        # Restart background mic monitor for level meter
        self.root.after(300, self._start_monitor)

    # --- Model loading ---

    def _load_model(self, size: str):
        self._set_loading_ui()
        cached = model_manager.is_model_cached(size)
        if cached:
            self._update_status(f"Loading {size} model...")
        else:
            self._update_status(f"Downloading & loading {size} model (first time)...")
        threading.Thread(target=self._load_model_sync, args=(size,), daemon=True).start()

    def _load_model_sync(self, size: str):
        try:
            model_size = model_manager.get_model_path(size)
            print(f"[model] loading WhisperModel({model_size!r}, device='auto', compute_type='int8')...")
            model = WhisperModel(model_size, device="auto", compute_type="int8")
            print(f"[model] loaded successfully")
            self.root.after(0, self._on_model_loaded, size, model, None)
        except Exception as e:
            print(f"[model] ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, self._on_model_loaded, size, None, e)

    def _on_model_loaded(self, size: str, model, error):
        if error:
            self._set_stopped_ui()
            self._update_status(f"Failed to load model: {error}")
            messagebox.showerror("Model Error", str(error))
            return
        self.model = model
        self._loaded_key = size
        self._update_status(f"Model ready: {size}")
        self._begin_audio_capture()

    # --- Audio capture ---

    def _begin_audio_capture(self):
        # Save monitor thread ref before clearing it
        old_monitor = self.monitor_thread
        self._stop_monitor()
        self.recording = True
        self.paused = False
        self.recorded_frames = []
        self._set_recording_ui()
        self._update_status("Recording & transcribing...")

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        self.audio_thread = threading.Thread(
            target=self._audio_loop, args=(old_monitor,), daemon=True,
        )
        self.audio_thread.start()

    def _audio_loop(self, old_monitor=None):
        # Wait for monitor stream to fully close before opening recording stream
        if old_monitor and old_monitor.is_alive():
            old_monitor.join(timeout=1.0)
        print(f"[audio_loop] starting, device={self.selected_device}")
        self._transcribing = False
        try:
            working_buffer = bytearray()
            last_transcribe_time = 0.0
            had_speech = False          # did we see speech in current buffer?
            last_speech_time = 0.0      # when we last heard speech
            speech_samples_in_buffer = 0  # how many speech samples since last transcribe

            with sd.RawInputStream(
                samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                dtype="int16", channels=1, device=self.selected_device,
                callback=self._audio_callback,
            ):
                print("[audio_loop] stream opened")
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
                        working_buffer.extend(data)

                        # Check if this chunk has speech
                        chunk_int16 = np.frombuffer(data, dtype=np.int16)
                        chunk_rms = np.sqrt(np.mean((chunk_int16.astype(np.float32) / 32768.0) ** 2))
                        if chunk_rms >= SPEECH_RMS_THRESHOLD:
                            had_speech = True
                            last_speech_time = now
                            speech_samples_in_buffer += len(data) // 2

                    buffer_seconds = (len(working_buffer) // 2) / SAMPLE_RATE

                    # Decide whether to transcribe
                    should_transcribe = False
                    is_commit = False

                    if buffer_seconds < 0.3 or self._transcribing:
                        pass
                    elif buffer_seconds >= COMMIT_SECONDS:
                        # Hard limit — commit to keep transcription fast
                        should_transcribe = True
                        is_commit = True
                    elif had_speech and now - last_speech_time >= SILENCE_COMMIT:
                        # Speaker paused — commit immediately
                        should_transcribe = True
                        is_commit = True
                    elif (had_speech
                          and speech_samples_in_buffer > 0
                          and now - last_transcribe_time >= PARTIAL_INTERVAL):
                        # New speech since last transcription — show partial
                        should_transcribe = True

                    if should_transcribe:
                        last_transcribe_time = now
                        buf_copy = bytes(working_buffer)
                        self._transcribing = True
                        threading.Thread(
                            target=self._transcribe_working,
                            args=(buf_copy, is_commit),
                            daemon=True,
                        ).start()
                        speech_samples_in_buffer = 0
                        if is_commit:
                            working_buffer.clear()
                            had_speech = False

                # Flush remaining working buffer on stop
                if len(working_buffer) > 0 and had_speech:
                    self._transcribe_working(bytes(working_buffer), True)
                working_buffer.clear()

        except Exception as e:
            self.root.after(0, self._on_audio_error, e)

    def _transcribe_working(self, audio_bytes: bytes, commit: bool):
        """Transcribe the working buffer. If commit=True, append as final text."""
        import time as _time
        t0 = _time.time()
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        duration = len(audio_float32) / SAMPLE_RATE

        print(f"[transcribe] {'COMMIT' if commit else 'partial'}: {duration:.1f}s audio")

        try:
            kwargs = {"beam_size": 1}  # greedy for speed
            lang = self.lang_var.get()
            if lang and lang != "auto":
                kwargs["language"] = lang
            prompt = self.prompt_var.get().strip()
            if prompt:
                kwargs["initial_prompt"] = prompt
            segments, info = self.model.transcribe(audio_float32, **kwargs)
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            text = " ".join(text_parts).strip()
            elapsed = _time.time() - t0
            print(f"[transcribe] {text!r} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"[transcribe] ERROR: {e}")
            self._transcribing = False
            return

        self._transcribing = False
        if commit:
            if text:
                self.root.after(0, self._append_final, text)
            else:
                self.root.after(0, self._remove_partial)
        else:
            # Show as updating partial (replaces previous partial)
            self.root.after(0, self._show_partial, text if text else "")

    def _audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            self.audio_queue.put(bytes(indata))

    # --- Text display ---

    def _append_final(self, text: str):
        self._remove_partial()
        content = self.text.get("1.0", tk.END).rstrip("\n")
        if content:
            self.text.insert(tk.END, " ")
        self.text.insert(tk.END, text)
        self.text.see(tk.END)

    def _show_partial(self, text: str):
        self._remove_partial()
        if text:
            self.text.insert(tk.END, " " + text, "partial")
            self.text.see(tk.END)

    def _remove_partial(self):
        ranges = self.text.tag_ranges("partial")
        if ranges:
            for i in range(0, len(ranges), 2):
                self.text.delete(ranges[i], ranges[i + 1])

    def _on_audio_error(self, error):
        self.recording = False
        self.paused = False
        self.root.after(0, self._set_stopped_ui)
        self.root.after(0, self._update_status, f"Audio error: {error}")
        self.root.after(0, lambda: messagebox.showerror("Audio Error", str(error)))


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
