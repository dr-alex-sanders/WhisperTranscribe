"""Whisper model manager — supports faster-whisper and mlx-whisper backends."""

import os

# Map UI display labels → (backend, model_id) tuples
MODELS = {
    # mlx-whisper models (Apple Silicon GPU)
    "mlx-whisper tiny": ("mlx", "mlx-community/whisper-tiny"),
    "mlx-whisper small": ("mlx", "mlx-community/whisper-small-mlx"),
    "mlx-whisper medium": ("mlx", "mlx-community/whisper-medium-mlx"),
    "mlx-whisper large-v3": ("mlx", "mlx-community/whisper-large-v3-mlx"),
    "mlx-whisper large-v3-turbo": ("mlx", "mlx-community/whisper-large-v3-turbo"),
    "mlx-whisper distil-large-v3": ("mlx", "mlx-community/distil-whisper-large-v3"),
    "mlx-whisper medical-v1": ("mlx", "Crystalcareai/Whisper-Medicalv1"),
    # faster-whisper models (CPU)
    "faster-whisper tiny": ("faster", "tiny"),
    "faster-whisper small": ("faster", "small"),
    "faster-whisper medium": ("faster", "medium"),
    "faster-whisper large-v3": ("faster", "large-v3"),
    "faster-whisper distil-large-v3": ("faster", "Systran/faster-distil-whisper-large-v3"),
    # Local medical model
    "google medasr": ("medasr", "google/medasr"),
    # Cloud API models
    "deepgram nova-3-medical": ("deepgram", "nova-3-medical"),
    "nvidia nemo canary": ("nvidia", "nvidia/canary-1b"),
    "aws transcribe medical": ("aws-transcribe", "medical"),
    "openai gpt-4o-transcribe": ("openai", "gpt-4o-transcribe"),
    "speechmatics medical": ("speechmatics", "medical"),
}

SIZES = list(MODELS.keys())
DEFAULT_SIZE = "mlx-whisper small"

# HuggingFace cache where models are stored
HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def get_model_info(size: str) -> tuple[str, str]:
    """Return (backend, model_id) for the given display label."""
    return MODELS.get(size, ("faster", size))


def is_model_cached(size: str) -> bool:
    """Check if a model is already cached in HuggingFace hub."""
    backend, model_id = get_model_info(size)
    if backend in ("deepgram", "nvidia", "aws-transcribe", "openai", "speechmatics"):
        return True  # cloud models, always available
    if not os.path.isdir(HF_CACHE):
        return False
    if backend == "faster":
        if "/" in model_id:
            # Full repo path (e.g. Systran/faster-distil-whisper-large-v3)
            expected_dir = f"models--{model_id.replace('/', '--')}"
        else:
            # Short name (e.g. "tiny") → Systran/faster-whisper-{size}
            expected_dir = f"models--Systran--faster-whisper-{model_id}"
    else:
        # mlx-whisper stores under models--{org}--{repo} with / replaced by --
        expected_dir = f"models--{model_id.replace('/', '--')}"
    model_dir = os.path.join(HF_CACHE, expected_dir)
    if not os.path.isdir(model_dir):
        return False
    # Check that snapshots directory has content (model fully downloaded)
    snapshots = os.path.join(model_dir, "snapshots")
    if os.path.isdir(snapshots) and os.listdir(snapshots):
        return True
    return False
