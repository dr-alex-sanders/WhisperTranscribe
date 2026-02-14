"""Speechmatics Medical cloud API client (batch jobs)."""

import io
import json
import os
import time
import wave


class SpeechmaticsError(Exception):
    """Raised on Speechmatics API errors."""


_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
_API_KEY_VAR = "SPEECHMATICS_API_KEY"


def get_api_key() -> str | None:
    """Return the Speechmatics API key from environment or .env file."""
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
    """Save the Speechmatics API key to the .env file."""
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


_BASE_URL = "https://asr.api.speechmatics.com/v2"


def transcribe(
    audio_int16_bytes: bytes,
    sample_rate: int = 16000,
    language: str = "en",
    channels: int = 1,
) -> str:
    """Send audio to Speechmatics Medical and return the transcript.

    Creates a batch job, polls for completion, fetches and parses results.

    Raises:
        SpeechmaticsError: On missing key, HTTP errors, or job failure.
    """
    import httpx

    api_key = get_api_key()
    if not api_key:
        raise SpeechmaticsError("No Speechmatics API key configured")

    wav_data = _wrap_wav(audio_int16_bytes, sample_rate, channels)
    headers = {"Authorization": f"Bearer {api_key}"}

    config = {
        "type": "transcription",
        "transcription_config": {
            "language": language,
            "operating_point": "enhanced",
            "domain": "medical",
        },
    }

    # 1. Create job
    try:
        resp = httpx.post(
            f"{_BASE_URL}/jobs/",
            headers=headers,
            files={"data_file": ("audio.wav", wav_data, "audio/wav")},
            data={"config": json.dumps(config)},
            timeout=60.0,
        )
    except httpx.HTTPError as e:
        raise SpeechmaticsError(f"Network error: {e}") from e

    if resp.status_code == 401:
        raise SpeechmaticsError("Invalid Speechmatics API key (401 Unauthorized)")
    if resp.status_code not in (200, 201):
        raise SpeechmaticsError(
            f"Speechmatics job creation failed {resp.status_code}: {resp.text[:200]}"
        )

    job_id = resp.json().get("id")
    if not job_id:
        raise SpeechmaticsError("No job ID in Speechmatics response")

    # 2. Poll for completion
    client = httpx.Client(headers=headers, timeout=30.0)
    try:
        for _ in range(300):  # up to 5 minutes
            poll = client.get(f"{_BASE_URL}/jobs/{job_id}")
            job = poll.json().get("job", {})
            status = job.get("status", "")
            if status == "done":
                break
            elif status in ("rejected", "deleted"):
                raise SpeechmaticsError(f"Speechmatics job {status}: {job}")
            time.sleep(1)
        else:
            raise SpeechmaticsError("Speechmatics job timed out after 5 minutes")

        # 3. Fetch transcript
        tr_resp = client.get(
            f"{_BASE_URL}/jobs/{job_id}/transcript",
            headers={**headers, "Accept": "application/json"},
        )
        if tr_resp.status_code != 200:
            raise SpeechmaticsError(
                f"Failed to fetch transcript: {tr_resp.status_code}"
            )
    finally:
        client.close()

    # 4. Parse results — concatenate word content
    data = tr_resp.json()
    results = data.get("results", [])
    parts = []
    for item in results:
        alts = item.get("alternatives", [])
        if alts:
            content = alts[0].get("content", "")
            if item.get("type") == "punctuation":
                # Punctuation attaches to the previous word (no space)
                if parts:
                    parts[-1] += content
                else:
                    parts.append(content)
            else:
                parts.append(content)
    return " ".join(parts).strip()
