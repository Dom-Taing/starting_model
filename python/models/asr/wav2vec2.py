import numpy as np
from transformers import pipeline as hf_pipeline

from .base import ASRModel


class Wav2Vec2ASR(ASRModel):
    def __init__(self, model_id: str = "facebook/wav2vec2-base-960h", device: int = -1):
        self.model_id = model_id
        self.device = device
        self._pipeline = None

    def load(self) -> None:
        self._pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=self.device,
            chunk_length_s=20,
            stride_length_s=4,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if self._pipeline is None:
            self.load()
        result = self._pipeline(audio, padding=True)
        return result["text"]
