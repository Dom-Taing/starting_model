# Speech Processing Project

A real-time speech-to-text and text-to-speech pipeline using Wav2Vec2 and Piper TTS.

## Features

- **Real-time ASR**: Live speech-to-text transcription using Wav2Vec2
- **Text-to-Speech**: Generate speech from text using Piper TTS
- **Audio Recording**: Record from microphone with multiple modes
- **Spell Correction**: Optional text correction using SymSpell
- **Complete Pipeline**: Record → Transcribe → Correct → Generate Speech

## Setup

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd starting_model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Required Models

**Piper TTS Voice Model:**
```bash
mkdir -p voice_model
cd voice_model
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx.json
cd ..
```

**SymSpell Dictionary (Optional):**
```bash
wget https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell.FrequencyDictionary/frequency_dictionary_en_82_765.txt
```

### 3. Key Dependencies

- `torch` - PyTorch for ML models
- `transformers` - Hugging Face transformers (Wav2Vec2)
- `librosa` - Audio processing
- `sounddevice` - Microphone recording
- `piper-tts` - Text-to-speech
- `symspellpy` - Spell correction
- `scipy` - Scientific computing
- `numpy` - Numerical computing

## Usage

### Real-time Speech-to-Text with TTS
```bash
python3 python/transcribe_w2v2.py
```
- Starts real-time recording and transcription
- Press ENTER to stop
- Automatically generates TTS from transcribed text
- Creates 3 files: original recording, TTS output, corrected TTS

### ASR from Audio File
```bash
python3 python/transcribe_w2v2.py Audio/test_1.wav
```

### Record from Microphone (Fixed Duration)
```bash
python3 python/transcribe_w2v2.py --mic 10  # Record 10 seconds
```

### Test TTS Only
```bash
python3 python/transcribe_w2v2.py --tts
```

## File Structure

```
starting_model/
├── python/
│   └── transcribe_w2v2.py      # Main script
├── Audio/                      # Sample audio files
├── voice_model/                # Piper TTS models (.onnx files)
├── ASR.ipynb                   # Jupyter notebook
├── requirements.txt            # Python dependencies
├── frequency_dictionary_*.txt  # SymSpell dictionary
└── README.md                   # This file
```

## Output Files

- `realtime_recording_YYYYMMDD_HHMMSS.wav` - Original voice recording
- `tts_output_YYYYMMDD_HHMMSS.wav` - TTS from transcribed text
- `tts_corrected_YYYYMMDD_HHMMSS.wav` - TTS from corrected text

## Troubleshooting

**Audio Device Issues:**
```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

**Missing Voice Model:**
Make sure you've downloaded the Piper TTS model files to the `voice_model/` directory.

**Import Errors:**
Ensure your virtual environment is activated and all packages are installed via `requirements.txt`.