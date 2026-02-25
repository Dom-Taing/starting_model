import wave

from .base import TTSModel


class PiperTTS(TTSModel):
    def __init__(self, model_path: str = "voice_model/en_US-lessac-medium.onnx"):
        self.model_path = model_path
        self._voice = None

    def load(self) -> None:
        from piper import PiperVoice
        self._voice = PiperVoice.load(self.model_path)

    def synthesize(self, text: str, output_path: str) -> str:
        if self._voice is None:
            self.load()
        with wave.open(output_path, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)
        return output_path
