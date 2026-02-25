from .base import TTSModel


class VoxCPMTTS(TTSModel):
    # TODO: Docker server needs HTTP endpoint
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url

    def load(self) -> None:
        # Server handles model loading; nothing to do here
        pass

    def synthesize(self, text: str, output_path: str) -> str:
        import requests
        response = requests.post(
            f"{self.server_url}/synthesize",
            json={"text": text},
            timeout=30,
        )
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path
