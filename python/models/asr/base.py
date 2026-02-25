from abc import ABC, abstractmethod
import numpy as np


class ASRModel(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...
