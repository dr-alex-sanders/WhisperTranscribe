"""Deepgram Nova-3 Medical REST API client."""

import io
import os
import wave


class DeepgramError(Exception):
    """Raised on Deepgram API errors (auth, network, quota)."""


_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
_API_KEY_VAR = "DEEPGRAM_API_KEY"


def get_api_key() -> str | None:
    """Return the Deepgram API key from environment or .env file."""
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
    """Save the Deepgram API key to the .env file."""
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
    model: str = "nova-3-medical",
) -> str:
    """Send audio to Deepgram and return the transcript text.

    Args:
        audio_int16_bytes: Raw int16 PCM audio bytes.
        sample_rate: Audio sample rate (default 16000).
        language: Language code (default "en").
        channels: Number of audio channels.
        model: Deepgram model name.

    Returns:
        Transcript string.

    Raises:
        DeepgramError: On missing key, HTTP errors, or empty response.
    """
    import httpx

    api_key = get_api_key()
    if not api_key:
        raise DeepgramError("No Deepgram API key configured")

    wav_data = _wrap_wav(audio_int16_bytes, sample_rate, channels)

    params = {
        "model": model,
        "language": language,
        "smart_format": "true",
        "punctuate": "true",
    }
    if channels > 1:
        params["multichannel"] = "true"

    try:
        resp = httpx.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "audio/wav",
            },
            content=wav_data,
            timeout=120.0,
        )
    except httpx.HTTPError as e:
        raise DeepgramError(f"Network error: {e}") from e

    if resp.status_code == 401:
        raise DeepgramError("Invalid Deepgram API key (401 Unauthorized)")
    if resp.status_code == 402:
        raise DeepgramError("Deepgram quota exceeded (402 Payment Required)")
    if resp.status_code != 200:
        raise DeepgramError(f"Deepgram API error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    try:
        channels_result = data["results"]["channels"]
        parts = []
        for ch in channels_result:
            alt = ch["alternatives"]
            if alt:
                parts.append(alt[0].get("transcript", ""))
        return " ".join(p for p in parts if p).strip()
    except (KeyError, IndexError):
        return ""
