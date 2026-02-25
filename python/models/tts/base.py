from abc import ABC, abstractmethod


class TTSModel(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def synthesize(self, text: str, output_path: str) -> str: ...
