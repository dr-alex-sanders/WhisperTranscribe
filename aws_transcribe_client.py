"""Amazon Transcribe Medical client using boto3."""

import io
import json
import os
import time
import uuid
import wave


class AWSTranscribeError(Exception):
    """Raised on AWS Transcribe Medical errors."""


_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

_CONFIG_KEYS = {
    "AWS_ACCESS_KEY_ID": None,
    "AWS_SECRET_ACCESS_KEY": None,
    "AWS_REGION": "us-east-1",
    "AWS_TRANSCRIBE_BUCKET": None,
}


def _read_env(var: str) -> str | None:
    """Read a variable from environment or .env file."""
    val = os.environ.get(var)
    if val:
        return val
    if os.path.exists(_ENV_FILE):
        with open(_ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{var}="):
                    v = line.split("=", 1)[1].strip().strip("'\"")
                    if v:
                        return v
    return None


def _write_env(var: str, value: str) -> None:
    """Write a variable to the .env file."""
    lines = []
    replaced = False
    if os.path.exists(_ENV_FILE):
        with open(_ENV_FILE) as f:
            for line in f:
                if line.strip().startswith(f"{var}="):
                    lines.append(f"{var}={value}\n")
                    replaced = True
                else:
                    lines.append(line)
    if not replaced:
        lines.append(f"{var}={value}\n")
    fd = os.open(_ENV_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.writelines(lines)
    os.environ[var] = value


def get_config() -> dict:
    """Return current AWS config as a dict. Missing values are None."""
    return {
        "access_key": _read_env("AWS_ACCESS_KEY_ID"),
        "secret_key": _read_env("AWS_SECRET_ACCESS_KEY"),
        "region": _read_env("AWS_REGION") or "us-east-1",
        "bucket": _read_env("AWS_TRANSCRIBE_BUCKET"),
    }


def is_configured() -> bool:
    """Check if all required AWS config is present."""
    cfg = get_config()
    return all([cfg["access_key"], cfg["secret_key"], cfg["bucket"]])


def set_config(access_key: str, secret_key: str, region: str, bucket: str) -> None:
    """Save AWS config to .env file."""
    _write_env("AWS_ACCESS_KEY_ID", access_key)
    _write_env("AWS_SECRET_ACCESS_KEY", secret_key)
    _write_env("AWS_REGION", region)
    _write_env("AWS_TRANSCRIBE_BUCKET", bucket)


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
    language: str = "en-US",
    channels: int = 1,
) -> str:
    """Send audio to AWS Transcribe Medical and return the transcript.

    Uploads WAV to S3, runs a medical transcription job, polls for
    completion, fetches results, and cleans up.

    Note: AWS Transcribe Medical only supports en-US.

    Raises:
        AWSTranscribeError: On missing config, AWS errors, or job failure.
    """
    try:
        import boto3
    except ImportError:
        raise AWSTranscribeError(
            "boto3 is required for AWS Transcribe Medical.\n"
            "Install it with: pip install boto3"
        )

    cfg = get_config()
    if not cfg["access_key"] or not cfg["secret_key"] or not cfg["bucket"]:
        raise AWSTranscribeError("AWS credentials or S3 bucket not configured")

    session = boto3.Session(
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"],
        region_name=cfg["region"],
    )

    wav_data = _wrap_wav(audio_int16_bytes, sample_rate, channels)
    job_name = f"wt-{uuid.uuid4().hex[:12]}"
    s3_key = f"transcribe-temp/{job_name}.wav"

    # 1. Upload WAV to S3
    try:
        s3 = session.client("s3")
        s3.put_object(Bucket=cfg["bucket"], Key=s3_key, Body=wav_data)
    except Exception as e:
        raise AWSTranscribeError(f"S3 upload failed: {e}") from e

    # 2. Start medical transcription job
    try:
        tc = session.client("transcribe")
        tc.start_medical_transcription_job(
            MedicalTranscriptionJobName=job_name,
            LanguageCode="en-US",
            MediaFormat="wav",
            Media={"MediaFileUri": f"s3://{cfg['bucket']}/{s3_key}"},
            OutputBucketName=cfg["bucket"],
            OutputKey=f"transcribe-temp/{job_name}-output.json",
            Specialty="PRIMARYCARE",
            Type="DICTATION",
        )
    except Exception as e:
        _cleanup_s3(s3, cfg["bucket"], s3_key)
        raise AWSTranscribeError(f"Failed to start transcription job: {e}") from e

    # 3. Poll for completion
    output_key = f"transcribe-temp/{job_name}-output.json"
    try:
        for _ in range(300):  # up to 5 minutes
            result = tc.get_medical_transcription_job(
                MedicalTranscriptionJobName=job_name,
            )
            status = result["MedicalTranscriptionJob"]["TranscriptionJobStatus"]
            if status == "COMPLETED":
                break
            elif status == "FAILED":
                reason = result["MedicalTranscriptionJob"].get("FailureReason", "unknown")
                raise AWSTranscribeError(f"Transcription job failed: {reason}")
            time.sleep(1)
        else:
            raise AWSTranscribeError("Transcription job timed out after 5 minutes")
    except AWSTranscribeError:
        raise
    except Exception as e:
        raise AWSTranscribeError(f"Error polling transcription job: {e}") from e
    finally:
        _cleanup_s3(s3, cfg["bucket"], s3_key)

    # 4. Fetch results from S3
    try:
        obj = s3.get_object(Bucket=cfg["bucket"], Key=output_key)
        result_data = json.loads(obj["Body"].read())
        transcripts = result_data.get("results", {}).get("transcripts", [])
        text = " ".join(t.get("transcript", "") for t in transcripts).strip()
    except Exception as e:
        raise AWSTranscribeError(f"Failed to fetch results: {e}") from e
    finally:
        _cleanup_s3(s3, cfg["bucket"], output_key)

    return text


def _cleanup_s3(s3_client, bucket: str, key: str) -> None:
    """Silently delete an S3 object."""
    try:
        s3_client.delete_object(Bucket=bucket, Key=key)
    except Exception:
        pass
