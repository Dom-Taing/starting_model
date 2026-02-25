# Speech Pipeline

Real-time, streaming, and file-based speech processing using a modular ASR + TTS architecture.

---

## Overview

This project provides a speech pipeline that records audio, transcribes it with an ASR model (wav2vec2 by default), optionally post-processes the text, and synthesizes speech from it using a TTS model (Piper by default). Each component is swappable via a clean OOP interface.

The primary use case is **accent clarification**: speak naturally into a mic, and the pipeline instantly re-speaks each utterance in a clear, consistent synthesized voice so others can understand you more easily.

---

## Architecture

```
Microphone / Audio File
        │
        ▼
  AudioRecorder / load_audio()
        │
        ▼
  AudioPreprocessor.process()      (band-pass filter → noise reduction)
        │
        ▼
    ASRModel.transcribe()          (Wav2Vec2ASR, WhisperASR, ...)
        │
        ▼
  TextProcessor.process()          (optional: normalize + autocorrect)
        │
        ▼
    TTSModel.synthesize()          (PiperTTS, VoxCPMTTS, ...)
        │
        ▼
   output.wav / immediate playback
```

### Stream mode (accent clarification)

In `--mode stream` the pipeline runs as a continuous live loop:

```
AUDIO THREAD   InputStream callback
                   → UtteranceSegmenter.push(chunk)   (RMS only, never blocks)
                   → if utterance boundary detected → queue.put(audio)

WORKER THREAD  (one utterance at a time, serial)
                   → AudioPreprocessor (bandpass → denoise) → ASR → TextProcessor → TTS → play → loop
```

Utterance boundaries are detected by trailing silence (default 0.8 s). Each utterance is played back immediately after it is transcribed and synthesized, with no overlap.

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

### Stream mode — accent clarification (recommended)

```bash
python python/main.py --mode stream --asr whisper --tts piper
```

Speak into the mic. Each time you pause, the pipeline transcribes your utterance and immediately plays it back in a clear synthesized voice. Press **ENTER** to stop.

Use **headphones** to prevent the mic picking up TTS playback.

```bash
# Tune for slower speakers (more pause tolerance)
python python/main.py --mode stream --asr whisper --tts piper --trailing-silence 1.2

# With VoxCPM (Docker server must be running)
python python/main.py --mode stream --asr whisper --tts voxcpm
```

| Flag | Default | Description |
|---|---|---|
| `--silence-threshold` | `0.01` | RMS level that counts as speech |
| `--trailing-silence` | `0.8` | Seconds of silence that ends an utterance |
| `--min-speech` | `0.3` | Minimum speech duration accepted (rejects coughs/noise) |

### Realtime transcription

```bash
python python/main.py
```

Press **ENTER** to stop recording. The pipeline transcribes in 3-second chunks, compiles the full text, and generates a TTS audio file.

### File transcription

```bash
python python/main.py --mode file --input Audio/test_1.wav
```

### Use Whisper Tiny for transcription

```bash
python python/main.py --asr whisper
```

Whisper handles accents and noisier audio better than wav2vec2 at the cost of slightly higher latency. The default variant is `openai/whisper-tiny` (39M parameters, English-only transcription mode).

### Use GPU for faster ASR

```bash
python python/main.py --gpu
```

### Use VoxCPM TTS (requires Docker server, see Docker / VoxCPM section)

```bash
python python/main.py --tts voxcpm
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
usage: main.py [-h] [--asr {wav2vec2,whisper}] [--tts {piper,voxcpm,none}]
               [--mode {realtime,file,stream}] [--input PATH] [--output PATH]
               [--sample-rate INT] [--chunk-duration FLOAT] [--gpu]
               [--piper-model PATH] [--voxcpm-url URL] [--autocorrect]
               [--silence-threshold RMS] [--trailing-silence SECS]
               [--min-speech SECS]
```

### ASR model comparison

| Model | Size | Speed | Notes |
|---|---|---|---|
| `wav2vec2` | ~360MB | Fast | Good for clean, clear speech |
| `whisper` | ~39MB (tiny) | Moderate | Better accent/noise robustness |

---

## Preprocessing

`AudioPreprocessor` (in `python/audio_utils.py`) runs two steps on raw audio before it reaches the ASR model:

| Step | What it does | Why |
|---|---|---|
| **Band-pass filter** | 4th-order Butterworth, 80 Hz – 8 kHz | Removes low-frequency rumble (HVAC, handling noise) and high-frequency hiss while keeping the full speech band intact |
| **Noise reduction** | `noisereduce` spectral subtraction | Suppresses stationary background noise (fan hum, room tone) to improve ASR accuracy |

Constructor parameters and their defaults:

| Parameter | Default | Description |
|---|---|---|
| `highpass_hz` | `80.0` | Low-frequency cut-off (Hz) |
| `lowpass_hz` | `8000.0` | High-frequency cut-off (Hz) |
| `filter_order` | `4` | Butterworth filter order |
| `debug` | `False` | Saves `debug_raw.wav` and `debug_filtered.wav` to disk for inspection |

**Planned additions:** voice activity detection (skip silent chunks), dynamic range normalisation, resampling guard.

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

VoxCPM runs as a Flask HTTP server inside Docker. The pipeline client (`python/models/tts/voxcpm.py`) sends text to it and receives WAV bytes back.

### Requirements

- Docker Desktop installed and running

### 1. Build the image

```bash
cd voxcpm-docker
docker build -t voxcpm-tts .
```

The build will:
- Install PyTorch (CPU), `voxcpm`, `soundfile`, and `flask`
- Apply a required attention-mask patch to the voxcpm source
- Copy `server.py` into the image

This takes a few minutes the first time due to the PyTorch download.

### 2. Run the server

```bash
docker run -p 8080:8080 \
  -v "$(pwd)/hf-cache:/root/.cache/huggingface" \
  voxcpm-tts
```

The `-v` flag mounts the local `hf-cache/` directory into the container so the model weights (`openbmb/VoxCPM1.5`) are cached on disk and not re-downloaded on every container start. On first run this will download the model (~2GB); subsequent starts load from the cache immediately.

#### Pin a consistent voice (recommended for stream mode)

By default VoxCPM picks a random speaker on each call. To use a fixed voice, provide a short (~5–10 s) reference WAV and its transcript:

```bash
docker run -p 8080:8080 \
  -v "$(pwd)/hf-cache:/root/.cache/huggingface" \
  -v /path/to/reference.wav:/ref/voice.wav \
  -e VOXCPM_REFERENCE_WAV=/ref/voice.wav \
  -e VOXCPM_REFERENCE_TEXT="Verbatim transcript of the reference clip." \
  voxcpm-tts
```

All synthesis calls will then clone from that reference, giving consistent output across utterances.

Wait for the log line:

```
Model ready.
 * Running on http://0.0.0.0:8080
```

### 3. Verify the server is up

```bash
curl http://localhost:8080/health
# {"status": "ok"}
```

### 4. Use it from the pipeline

```bash
python python/main.py --tts voxcpm
# or with a custom server address:
python python/main.py --tts voxcpm --voxcpm-url http://localhost:8080
```

### Endpoints

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/health` | — | `{"status": "ok"}` |
| POST | `/synthesize` | `{"text": "..."}` | `audio/wav` bytes |
