# Speech Pipeline

Real-time and file-based speech processing using a modular ASR + TTS architecture.

---

## Overview

This project provides a speech pipeline that records audio, transcribes it with an ASR model (wav2vec2 by default), optionally post-processes the text, and synthesises speech from it using a TTS model (Piper by default). Each component is swappable via a clean OOP interface.

---

## Architecture

```
Microphone / Audio File
        │
        ▼
  AudioRecorder / load_audio()
        │
        ▼
    ASRModel.transcribe()          (Wav2Vec2ASR, ...)
        │
        ▼
  TextProcessor.process()          (optional: normalize + autocorrect)
        │
        ▼
    TTSModel.synthesize()          (PiperTTS, VoxCPMTTS, ...)
        │
        ▼
   output.wav / realtime playback
```

---

## Requirements

- Python 3.10+
- System packages: `ffmpeg`, `libsndfile`

Install system dependencies (macOS):
```bash
brew install ffmpeg libsndfile
```

Install system dependencies (Ubuntu/Debian):
```bash
sudo apt install ffmpeg libsndfile1
```

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd starting_model

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download Piper ONNX voice model
mkdir -p voice_model
# Download en_US-lessac-medium.onnx and its .json config into voice_model/
# Source: https://github.com/rhasspy/piper/releases

# 5. (Optional) Download SymSpell dictionary for --autocorrect
# Download frequency_dictionary_en_82_765.txt and place in the project root
# Source: https://github.com/wolfgarbe/SymSpell
```

---

## Usage

### Realtime transcription (default)

```bash
python python/main.py
```

Press **ENTER** to stop recording. The pipeline transcribes in 3-second chunks, compiles the full text, and generates a TTS audio file.

### File transcription

```bash
python python/main.py --mode file --input Audio/test_1.wav
```

### Use GPU for faster ASR

```bash
python python/main.py --gpu
```

### Use VoxCPM TTS (requires Docker server)

```bash
python python/main.py --tts voxcpm --voxcpm-url http://localhost:8080
```

### Disable TTS output

```bash
python python/main.py --tts none
```

### Enable autocorrect (requires SymSpell dictionary)

```bash
python python/main.py --mode file --input Audio/test_1.wav --autocorrect
```

### All options

```
usage: main.py [-h] [--asr {wav2vec2}] [--tts {piper,voxcpm,none}]
               [--mode {realtime,file}] [--input PATH] [--output PATH]
               [--sample-rate INT] [--chunk-duration FLOAT] [--gpu]
               [--piper-model PATH] [--voxcpm-url URL] [--autocorrect]
```

---

## Adding a New Model

1. **Implement the interface** — subclass `ASRModel` or `TTSModel` from `python/models/asr/base.py` or `python/models/tts/base.py`.

```python
# python/models/asr/my_model.py
from .base import ASRModel
import numpy as np

class MyASR(ASRModel):
    def load(self) -> None:
        # load weights here (called lazily on first transcribe)
        ...

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        ...
        return "transcribed text"
```

2. **Register in `main.py`** — add your model name to `--asr` / `--tts` choices and instantiate it in `build_pipeline()`.

3. **Use it**:

```bash
python python/main.py --asr my_model
```

---

## Docker / VoxCPM

Build and run the VoxCPM TTS Docker container:

```bash
cd voxcpm-docker
docker build -t voxcpm-tts .
docker run -p 8080:8080 voxcpm-tts
```

The `VoxCPMTTS` client in `python/models/tts/voxcpm.py` will POST text to `{server_url}/synthesize` and write the returned audio bytes to disk.

> **Note:** The Docker container currently runs a local TTS script. An HTTP server endpoint (`/synthesize`) needs to be added to the container before `VoxCPMTTS` is fully functional.
