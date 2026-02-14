"""NVIDIA NIM Canary ASR REST API client."""

import io
import os
import wave


class NvidiaError(Exception):
    """Raised on NVIDIA NIM API errors (auth, network, quota)."""


_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
_API_KEY_VAR = "NVIDIA_API_KEY"


def get_api_key() -> str | None:
    """Return the NVIDIA API key from environment or .env file."""
    key = os.environ.get(_API_KEY_VAR)
    if key:
        return key
    if os.path.exists(_ENV_FILE):
        with open(_ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{_API_KEY_VAR}="):
                    val = line.split("=", 1)[1].strip().strip("'\"")
                    if val:
                        return val
    return None


def set_api_key(key: str) -> None:
    """Save the NVIDIA API key to the .env file."""
    lines = []
    replaced = False
    if os.path.exists(_ENV_FILE):
        with open(_ENV_FILE) as f:
            for line in f:
                if line.strip().startswith(f"{_API_KEY_VAR}="):
                    lines.append(f"{_API_KEY_VAR}={key}\n")
                    replaced = True
                else:
                    lines.append(line)
    if not replaced:
        lines.append(f"{_API_KEY_VAR}={key}\n")
    fd = os.open(_ENV_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.writelines(lines)
    os.environ[_API_KEY_VAR] = key


def _wrap_wav(pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw int16 PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def transcribe(
    audio_int16_bytes: bytes,
    sample_rate: int = 16000,
    language: str = "en",
    channels: int = 1,
    model: str = "nvidia/canary-1b",
) -> str:
    """Send audio to NVIDIA NIM and return the transcript text.

    Args:
        audio_int16_bytes: Raw int16 PCM audio bytes.
        sample_rate: Audio sample rate (default 16000).
        language: Language code (default "en").
        channels: Number of audio channels.
        model: NVIDIA NIM model name.

    Returns:
        Transcript string.

    Raises:
        NvidiaError: On missing key, HTTP errors, or empty response.
    """
    import httpx

    api_key = get_api_key()
    if not api_key:
        raise NvidiaError("No NVIDIA API key configured")

    wav_data = _wrap_wav(audio_int16_bytes, sample_rate, channels)

    try:
        resp = httpx.post(
            "https://integrate.api.nvidia.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("audio.wav", wav_data, "audio/wav")},
            data={"model": model, "language": language},
            timeout=120.0,
        )
    except httpx.HTTPError as e:
        raise NvidiaError(f"Network error: {e}") from e

    if resp.status_code == 401:
        raise NvidiaError("Invalid NVIDIA API key (401 Unauthorized)")
    if resp.status_code == 402:
        raise NvidiaError("NVIDIA quota exceeded (402 Payment Required)")
    if resp.status_code != 200:
        raise NvidiaError(f"NVIDIA API error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    return data.get("text", "").strip()
