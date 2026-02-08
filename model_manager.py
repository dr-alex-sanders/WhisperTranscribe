"""Faster-whisper model manager."""

import os

# Map UI display labels → faster-whisper model identifiers
MODELS = {
    "tiny (75 MB)": "tiny",
    "base (150 MB)": "base",
    "small (500 MB)": "small",
    "medium (1.5 GB)": "medium",
    "large-v3 (3 GB)": "large-v3",
}

SIZES = list(MODELS.keys())
DEFAULT_SIZE = "small (500 MB)"

# HuggingFace cache where faster-whisper stores downloaded models
HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def get_model_path(size: str) -> str:
    """Return the Whisper model size string. faster-whisper uses this directly."""
    return MODELS.get(size, size)


def is_model_cached(size: str) -> bool:
    """Check if a faster-whisper model is already cached in HuggingFace hub."""
    if not os.path.isdir(HF_CACHE):
        return False
    model_name = MODELS.get(size, size)
    # faster-whisper stores models under models--Systran--faster-whisper-{size}
    expected_dir = f"models--Systran--faster-whisper-{model_name}"
    model_dir = os.path.join(HF_CACHE, expected_dir)
    if not os.path.isdir(model_dir):
        return False
    # Check that snapshots directory has content (model fully downloaded)
    snapshots = os.path.join(model_dir, "snapshots")
    if os.path.isdir(snapshots) and os.listdir(snapshots):
        return True
    return False
