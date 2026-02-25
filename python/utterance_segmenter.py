from typing import Optional

import numpy as np


class UtteranceSegmenter:
    """
    State machine that detects utterance boundaries from a stream of audio chunks.

    States: IDLE → SPEAKING → TRAILING_SILENCE → emit → IDLE
    """

    _IDLE = "idle"
    _SPEAKING = "speaking"
    _TRAILING = "trailing"

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold: float = 0.01,
        min_speech_duration_s: float = 0.3,
        trailing_silence_s: float = 0.8,
    ):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.min_speech_frames = int(min_speech_duration_s * sample_rate)
        self.required_trailing_frames = int(trailing_silence_s * sample_rate)

        self._state = self._IDLE
        self._buffer: list[np.ndarray] = []
        self._speech_frames = 0
        self._trailing_frames = 0
        self._pending: Optional[np.ndarray] = None

    def push(self, chunk: np.ndarray) -> None:
        samples = chunk.flatten().astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        n = len(samples)

        if self._state == self._IDLE:
            if rms >= self.silence_threshold:
                self._state = self._SPEAKING
                self._buffer = [samples]
                self._speech_frames = n
                self._trailing_frames = 0

        elif self._state == self._SPEAKING:
            self._buffer.append(samples)
            if rms >= self.silence_threshold:
                self._speech_frames += n
            else:
                self._state = self._TRAILING
                self._trailing_frames = n

        elif self._state == self._TRAILING:
            self._buffer.append(samples)
            if rms >= self.silence_threshold:
                # Voice resumed — back to speaking
                self._state = self._SPEAKING
                self._speech_frames += n
                self._trailing_frames = 0
            else:
                self._trailing_frames += n
                if self._trailing_frames >= self.required_trailing_frames:
                    self._maybe_emit()
                    self._state = self._IDLE
                    self._buffer = []
                    self._speech_frames = 0
                    self._trailing_frames = 0

    def emit(self) -> Optional[np.ndarray]:
        """Returns the completed utterance once, then None."""
        result = self._pending
        self._pending = None
        return result

    def reset(self) -> None:
        self._state = self._IDLE
        self._buffer = []
        self._speech_frames = 0
        self._trailing_frames = 0
        self._pending = None

    def _maybe_emit(self) -> None:
        if self._speech_frames >= self.min_speech_frames and self._buffer:
            self._pending = np.concatenate(self._buffer)
