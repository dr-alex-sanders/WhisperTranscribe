"""Google MedASR local model wrapper.

Requires: pip install transformers>=5.0.0 torch
The model is gated — accept the license at https://huggingface.co/google/medasr
and run `huggingface-cli login` before first use.
"""

import os


class MedASRError(Exception):
    """Raised on MedASR loading/inference errors."""


class MedASRModel:
    """Wraps Google MedASR (Conformer-CTC) for transcription."""

    def __init__(self, model_id: str = "google/medasr"):
        try:
            from transformers import AutoModelForCTC, AutoProcessor
            import torch
        except ImportError:
            raise MedASRError(
                "MedASR requires transformers and torch.\n"
                "Install with: pip install 'transformers>=5.0.0' torch"
            )

        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModelForCTC.from_pretrained(model_id)
        except OSError as e:
            if "gated" in str(e).lower() or "401" in str(e):
                raise MedASRError(
                    "google/medasr is a gated model.\n"
                    "1. Accept the license at https://huggingface.co/google/medasr\n"
                    "2. Run: huggingface-cli login"
                ) from e
            raise MedASRError(f"Failed to load MedASR: {e}") from e

        self._torch = torch

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self._model = self._model.to(self.device)
        self._model.eval()

    def transcribe(self, audio_float32, sample_rate: int = 16000) -> str:
        """Transcribe float32 audio array to text.

        Args:
            audio_float32: numpy float32 array of audio samples.
            sample_rate: Sample rate (must be 16000).

        Returns:
            Transcript string.
        """
        inputs = self.processor(
            audio_float32, sampling_rate=sample_rate,
            return_tensors="pt", padding=True,
        )
        inputs = inputs.to(self.device)

        with self._torch.no_grad():
            logits = self._model(**inputs).logits

        predicted_ids = self._torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(predicted_ids)[0]
        return text.strip()
