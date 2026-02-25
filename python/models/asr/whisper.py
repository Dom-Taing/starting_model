import warnings

import numpy as np
import transformers
from transformers import pipeline as hf_pipeline

from .base import ASRModel

# Whisper via HuggingFace transformers does not require the openai-whisper
# package. The transformers library ships its own Whisper implementation and
# downloads the same model weights from the Hub, keeping the interface
# consistent with other ASR models and avoiding an extra dependency.


class WhisperASR(ASRModel):
    def __init__(
        self,
        model_id: str = "openai/whisper-tiny",
        device: int = -1,
        silence_threshold: float = 0.01,
    ):
        self.model_id = model_id
        self.device = device
        self._pipeline = None
        # Whisper hallucinates text on silent audio because it is a
        # sequence-to-sequence model trained to always produce output.
        # Chunks whose RMS energy is below this threshold are skipped.
        # Float32 audio is in [-1, 1]; typical speech RMS is 0.02-0.3+.
        self.silence_threshold = silence_threshold

    def load(self) -> None:
        # Suppress noisy but harmless HuggingFace warnings (attention mask,
        # forced_decoder_ids deprecation, generation config mismatches).
        transformers.logging.set_verbosity_error()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=self.device,
                chunk_length_s=30,
                stride_length_s=5,
                generate_kwargs={"language": "english", "task": "transcribe"},
            )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if self._pipeline is None:
            self.load()
        if np.sqrt(np.mean(audio ** 2)) < self.silence_threshold:
            return ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._pipeline({"array": audio, "sampling_rate": sample_rate})
        return result["text"]
