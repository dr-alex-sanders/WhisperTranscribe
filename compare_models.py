"""Compare all available local models on a single recording."""

import time
import wave
import sys
import numpy as np

SAMPLE_RATE = 16000


def read_wav(path):
    """Read a WAV file and return (int16_array, sample_rate)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    # Mono mixdown if stereo
    n_ch = 1
    with wave.open(path, "rb") as wf:
        n_ch = wf.getnchannels()
    if n_ch >= 2:
        raw = raw[0::n_ch]  # take first channel
    # Resample if needed
    if sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sr
        new_len = int(len(raw) * ratio)
        indices = np.linspace(0, len(raw) - 1, new_len).astype(int)
        raw = raw[indices]
    return raw


def run_mlx(model_id, audio_f32, lang):
    import mlx_whisper
    result = mlx_whisper.transcribe(
        audio_f32, path_or_hf_repo=model_id,
        language=lang, condition_on_previous_text=False,
        no_speech_threshold=0.5,
    )
    segments = result.get("segments", [])
    return " ".join(
        s["text"].strip() for s in segments if s.get("no_speech_prob", 0) < 0.5
    ).strip()


def run_faster(model_id, audio_f32, lang):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_id, device="auto", compute_type="int8")
    segments, _ = model.transcribe(audio_f32, beam_size=5, vad_filter=True, language=lang)
    return " ".join(s.text.strip() for s in segments).strip()


def run_medasr(model_id, audio_f32):
    import medasr_client
    model = medasr_client.MedASRModel(model_id)
    return model.transcribe(audio_f32, sample_rate=SAMPLE_RATE)


def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "recordings/rec_20260210_231917.wav"
    lang = sys.argv[2] if len(sys.argv) > 2 else "ru"

    print(f"Audio: {wav_path}")
    print(f"Language: {lang}")
    print("=" * 90)

    audio_int16 = read_wav(wav_path)
    audio_f32 = audio_int16.astype(np.float32) / 32768.0
    duration = len(audio_f32) / SAMPLE_RATE
    print(f"Duration: {duration:.1f}s\n")

    import model_manager

    results = []

    for label in model_manager.SIZES:
        backend, model_id = model_manager.get_model_info(label)

        # Skip cloud APIs (need keys)
        if backend in ("deepgram", "nvidia", "aws-transcribe", "openai", "speechmatics"):
            print(f"  [{label}] SKIP (cloud API — needs API key)")
            continue

        print(f"  [{label}] loading...", end="", flush=True)
        try:
            t0 = time.time()
            if backend == "mlx":
                text = run_mlx(model_id, audio_f32, lang)
            elif backend == "faster":
                text = run_faster(model_id, audio_f32, lang)
            elif backend == "medasr":
                text = run_medasr(model_id, audio_f32)
            else:
                print(f" unknown backend {backend}")
                continue
            elapsed = time.time() - t0
            results.append((label, elapsed, text))
            print(f" {elapsed:.1f}s")
            print(f"    => {text}")
        except Exception as e:
            print(f" ERROR: {e}")
            results.append((label, 0, f"ERROR: {e}"))

        print()

    # Summary table
    print("=" * 90)
    print(f"{'Model':<32} {'Time':>6}  Transcript")
    print("-" * 90)
    for label, elapsed, text in results:
        time_str = f"{elapsed:.1f}s" if elapsed > 0 else "ERR"
        # Truncate long text for table
        display = text[:55] + "..." if len(text) > 55 else text
        print(f"{label:<32} {time_str:>6}  {display}")


if __name__ == "__main__":
    main()
