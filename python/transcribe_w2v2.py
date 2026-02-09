import sys
import torch
import librosa
from transformers import pipeline
import re
from symspellpy import SymSpell, Verbosity
import tempfile
import os
import subprocess
import wave
from piper import PiperVoice
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import threading
import time

# Load once at startup
sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# You need a frequency dictionary file.
# Easiest: download one once, then keep it locally for offline use.
# Example: frequency_dictionary_en_82_765.txt (commonly used with SymSpell)
# Put it in your project folder and load it:
sym.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

def normalize_caps(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip())
    letters = [c for c in t if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        t = t.lower()
    return t

def autocorrect_symspell(text: str) -> str:
    t = normalize_caps(text)

    # SymSpell works best on a full string for segmentation + correction
    # This will fix spacing issues too (e.g., "pipe line" -> "pipeline")
    result = sym.lookup_compound(t, max_edit_distance=2, ignore_non_words=True)
    corrected = result[0].term if result else t

    # Light casing: capitalize first letter
    if corrected:
        corrected = corrected[0].upper() + corrected[1:]

    # Add period if missing
    if corrected and corrected[-1] not in ".!?":
        corrected += "."

    return corrected


def load_audio(path, sr=16000):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio

def transcribe(audio_path: str) -> str:
    # Good baseline model
    model_id = "facebook/wav2vec2-base-960h"

    device = -1  # CPU
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        chunk_length_s=20,   # chunking helps longer files
        stride_length_s=4
    )

    audio = load_audio(audio_path, sr=16000)
    out = asr(audio, padding=True)
    return out["text"]


def text_to_speech(text: str, output_path: str = None) -> str:
    voice = PiperVoice.load("voice_model/en_US-lessac-medium.onnx")
    with wave.open(output_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)


def record_microphone(duration: float = 5.0, sample_rate: int = 16000, output_path: str = None) -> str:
    """
    Record audio from the laptop microphone.
    
    Args:
        duration: Recording duration in seconds (default: 5.0)
        sample_rate: Audio sample rate in Hz (default: 16000)
        output_path: Optional path to save the audio file. If None, creates a temp file.
    
    Returns:
        Path to the recorded audio file
    """
    if output_path is None:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            output_path = f.name
    
    try:
        print(f"Recording for {duration} seconds... Speak now!")
        
        # Record audio
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype='float64')
        sd.wait()  # Wait until recording is finished
        
        # Convert to 16-bit integer format for WAV file
        audio_int16 = np.int16(audio_data * 32767)
        
        # Save to WAV file
        wavfile.write(output_path, sample_rate, audio_int16)
        
        print(f"Recording complete! Saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Microphone recording error: {e}")
        return None


def record_microphone_stream(sample_rate: int = 16000, output_path: str = None) -> str:
    """
    Record audio from the laptop microphone using InputStream (proper sounddevice approach).
    Continuously records until user presses Enter.
    
    Args:
        sample_rate: Audio sample rate in Hz (default: 16000)
        output_path: Optional path to save the audio file. If None, creates a temp file.
    
    Returns:
        Path to the recorded audio file
    """
    if output_path is None:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            output_path = f.name
    
    try:
        print("Recording started... Press ENTER to stop recording!")
        
        # Storage for recorded data
        recorded_data = []
        
        # Flag to control recording
        recording = True
        
        def stop_recording():
            nonlocal recording
            input()  # Wait for Enter key
            recording = False
        
        # Start the input thread
        input_thread = threading.Thread(target=stop_recording)
        input_thread.daemon = True
        input_thread.start()
        
        # Audio callback function
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            # Store the incoming audio data
            recorded_data.append(indata.copy())
        
        # Create InputStream for continuous recording
        with sd.InputStream(samplerate=sample_rate, 
                           channels=1, 
                           dtype='float32',
                           callback=audio_callback):
            
            # Keep recording until user presses Enter
            while recording:
                time.sleep(0.1)  # Small sleep to prevent busy waiting
        
        print("Recording stopped!")
        
        # Combine all recorded chunks
        if recorded_data:
            audio_data = np.concatenate(recorded_data, axis=0)
            
            # Convert to 16-bit integer format for WAV file
            audio_int16 = np.int16(audio_data * 32767)
            
            # Save to WAV file
            wavfile.write(output_path, sample_rate, audio_int16.flatten())
            
            duration_seconds = len(audio_data) / sample_rate
            print(f"Recording complete! Duration: {duration_seconds:.2f}s, Saved to: {output_path}")
            return output_path
        else:
            print("No audio recorded")
            return None
        
    except Exception as e:
        print(f"Microphone recording error: {e}")
        return None


def realtime_speech_to_text(sample_rate: int = 16000, chunk_duration: float = 3.0):
    """
    Continuous recording with real-time speech-to-text transcription.
    Records audio and transcribes it live while recording.
    Also saves the complete recording to a file.
    
    Args:
        sample_rate: Audio sample rate in Hz (default: 16000)
        chunk_duration: Duration in seconds of audio chunks to transcribe (default: 3.0)
    """
    try:
        print("Real-time Speech-to-Text started...")
        print("Press ENTER to stop recording and transcription")
        print("-" * 50)
        
        # Initialize ASR pipeline once
        model_id = "facebook/wav2vec2-base-960h"
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=-1,  # CPU
            chunk_length_s=20,
            stride_length_s=4
        )
        
        # Audio buffer for chunks (for real-time transcription)
        audio_buffer = []
        chunk_frames = int(chunk_duration * sample_rate)
        
        # Complete recording storage (for saving the file)
        complete_recording = []
        
        # Store all transcribed text chunks
        transcribed_chunks = []
        
        # Control flags
        recording = True
        
        def stop_recording():
            nonlocal recording
            input()  # Wait for Enter key
            recording = False
        
        # Start the input thread
        input_thread = threading.Thread(target=stop_recording)
        input_thread.daemon = True
        input_thread.start()
        
        # Audio callback function
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            
            # Store all audio data for complete recording
            complete_recording.append(indata.copy())
            
            # Add new audio data to buffer for real-time processing
            audio_buffer.extend(indata.flatten())
            
            # When we have enough audio, transcribe it
            if len(audio_buffer) >= chunk_frames:
                # Extract chunk for transcription
                chunk_data = np.array(audio_buffer[:chunk_frames], dtype=np.float32)
                
                # Remove processed data from buffer (with overlap)
                overlap_frames = int(0.5 * sample_rate)  # 0.5 second overlap
                audio_buffer[:] = audio_buffer[chunk_frames - overlap_frames:]
                
                # Transcribe in a separate thread to avoid blocking audio
                def transcribe_chunk():
                    try:
                        result = asr(chunk_data, sampling_rate=sample_rate)
                        text = result["text"].strip()
                        if text:  # Only print if there's actual text
                            print(f"🎤 {text}")
                            transcribed_chunks.append(text)  # Store the transcribed text
                    except Exception as e:
                        print(f"Transcription error: {e}")
                
                # Run transcription in background thread
                transcribe_thread = threading.Thread(target=transcribe_chunk)
                transcribe_thread.daemon = True
                transcribe_thread.start()
        
        # Create InputStream for continuous recording
        with sd.InputStream(samplerate=sample_rate, 
                           channels=1, 
                           dtype='float32',
                           callback=audio_callback):
            
            # Keep recording until user presses Enter
            while recording:
                time.sleep(0.1)
        
        print("\nReal-time transcription stopped!")
        
        # Save the complete recording
        if complete_recording:
            # Create output filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"realtime_recording_{timestamp}.wav"
            
            # Combine all recorded data
            audio_data = np.concatenate(complete_recording, axis=0)
            
            # Convert to 16-bit integer format for WAV file
            audio_int16 = np.int16(audio_data * 32767)
            
            # Save to WAV file
            wavfile.write(output_path, sample_rate, audio_int16.flatten())
            
            duration_seconds = len(audio_data) / sample_rate
            print(f"Recording saved: {output_path} ({duration_seconds:.2f}s)")
            
            # Process the compiled text with TTS
            if transcribed_chunks:
                # 1. Compile all real-time transcribed text
                compiled_text = " ".join(transcribed_chunks)
                print(f"Compiled text: {compiled_text}")
                
                # 2. Generate speech from the compiled text using TTS
                try:
                    # Create TTS output filename
                    tts_output_path = f"tts_output_{timestamp}.wav"
                    text_to_speech(compiled_text, tts_output_path)
                    print(f"TTS generated: {tts_output_path}")
                    
                    return {
                        'original_audio': output_path,
                        'compiled_text': compiled_text,
                        'tts_audio': tts_output_path,
                        'duration': duration_seconds,
                        'chunk_count': len(transcribed_chunks)
                    }
                    
                except Exception as e:
                    print(f"TTS error: {e}")
                    return {
                        'original_audio': output_path,
                        'compiled_text': compiled_text,
                        'duration': duration_seconds,
                        'error': str(e)
                    }
            else:
                print("No text captured for TTS")
                return {
                    'original_audio': output_path,
                    'compiled_text': "",
                    'duration': duration_seconds
                }

        else:
            print("No audio was recorded")
            return None
            
    except Exception as e:
        print(f"Real-time transcription error: {e}")
        return None


def test_tts_pipeline(text: str):
    output_path = "output.wav"
    text_to_speech(text, output_path)
    print(f"Generated speech saved to {output_path}")

def test_asr_pipeline(audio_path: str):
    # Good baseline model
    model_id = "facebook/wav2vec2-base-960h"

    device = -1  # CPU
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        chunk_length_s=20,   # chunking helps longer files
        stride_length_s=4
    )

    audio = load_audio(audio_path, sr=16000)
    out = asr(audio, padding=True)
    # clean_text = autocorrect_symspell(out["text"])
    print("raw text: " + out["text"])
    # print("clean text: " + clean_text)


def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python transcribe_w2v2.py path/to/audio.(wav|mp3|m4a)")
    #     sys.exit(1)
    
    # audio_path = sys.argv[1]
    # test_asr_pipeline(audio_path)

    # test_tts_pipeline("Hello, this is a test of the text-to-speech pipeline.")

    # record_microphone(duration=5.0, sample_rate=16000, output_path="mic_recording.wav")

    # record_microphone_stream(sample_rate=16000, output_path="mic_recording.wav")

    realtime_speech_to_text(sample_rate=16000, chunk_duration=3.0)
if __name__ == "__main__":
    main()
