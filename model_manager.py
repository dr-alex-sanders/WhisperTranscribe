"""Whisper model manager — MLX-whisper backend for Apple Silicon GPU."""

import os

# Map UI display labels → (backend, model_id) tuples
MODELS = {
    "mlx-whisper tiny": ("mlx", "mlx-community/whisper-tiny"),
    "mlx-whisper small": ("mlx", "mlx-community/whisper-small-mlx"),
    "mlx-whisper medium": ("mlx", "mlx-community/whisper-medium-mlx"),
    "mlx-whisper large-v3": ("mlx", "mlx-community/whisper-large-v3-mlx"),
    "mlx-whisper large-v3-turbo": ("mlx", "mlx-community/whisper-large-v3-turbo"),
    "mlx-whisper distil-large-v3": ("mlx", "mlx-community/distil-whisper-large-v3"),
    "mlx-whisper medical-v1": ("mlx", "Crystalcareai/Whisper-Medicalv1"),
}

SIZES = list(MODELS.keys())
DEFAULT_SIZE = "mlx-whisper small"

# HuggingFace cache where models are stored
HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def get_model_info(size: str) -> tuple[str, str]:
    """Return (backend, model_id) for the given display label."""
    return MODELS.get(size, ("mlx", size))


def is_model_cached(size: str) -> bool:
    """Check if a model is already cached in HuggingFace hub."""
    _, model_id = get_model_info(size)
    if not os.path.isdir(HF_CACHE):
        return False
    expected_dir = f"models--{model_id.replace('/', '--')}"
    model_dir = os.path.join(HF_CACHE, expected_dir)
    if not os.path.isdir(model_dir):
        return False
    # Check that snapshots directory has content (model fully downloaded)
    snapshots = os.path.join(model_dir, "snapshots")
    if os.path.isdir(snapshots) and os.listdir(snapshots):
        return True
    return False
