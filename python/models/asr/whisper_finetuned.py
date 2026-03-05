import warnings

import numpy as np
import torch
import transformers
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .base import ASRModel


class FinetunedWhisperASR(ASRModel):
    """ASR using a locally fine-tuned Whisper checkpoint.

    Loads the model directly with WhisperForConditionalGeneration and
    WhisperProcessor (the same way the model was trained in Colab), bypassing
    the HuggingFace pipeline which is incompatible with some fine-tuned
    checkpoints.

    The processor is loaded from the base model (openai/whisper-tiny.en by
    default) because the tokenizer/feature extractor are unchanged by
    fine-tuning and the saved copy may have format issues with newer
    transformers versions.
    """

    def __init__(
        self,
        model_path: str,
        base_model_id: str = "openai/whisper-tiny.en",
        device: int = -1,
        silence_threshold: float = 0.01,
    ):
        self.model_path = model_path
        self.base_model_id = base_model_id
        self.device = torch.device("cuda" if device >= 0 else "cpu")
        self._model = None
        self._processor = None
        self.silence_threshold = silence_threshold

    def load(self) -> None:
        transformers.logging.set_verbosity_error()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._processor = WhisperProcessor.from_pretrained(self.base_model_id)
            self._model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
            self._model.to(self.device)
            self._model.eval()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if self._model is None:
            self.load()
        if np.sqrt(np.mean(audio ** 2)) < self.silence_threshold:
            return ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inputs = self._processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)
            with torch.no_grad():
                predicted_ids = self._model.generate(input_features, use_cache=False)
            text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return text[0].strip()
