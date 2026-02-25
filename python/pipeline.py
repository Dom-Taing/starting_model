import datetime
import threading
import time

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd

from audio_utils import AudioPreprocessor, AudioRecorder, load_audio
from models.asr.base import ASRModel
from models.tts.base import TTSModel
from text_utils import TextProcessor


class SpeechPipeline:
    def __init__(
        self,
        asr: ASRModel,
        tts: TTSModel = None,
        sample_rate: int = 16000,
        text_processor: TextProcessor = None,
        audio_preprocessor: AudioPreprocessor = None,
    ):
        self.asr = asr
        self.tts = tts
        self.sample_rate = sample_rate
        self.text_processor = text_processor
        self.audio_preprocessor = audio_preprocessor or AudioPreprocessor()

    def transcribe_file(self, audio_path: str) -> str:
        audio = load_audio(audio_path, sr=self.sample_rate)
        audio = self.audio_preprocessor.process(audio, self.sample_rate)
        text = self.asr.transcribe(audio, self.sample_rate)
        if self.text_processor:
            text = self.text_processor.process(text)
        return text

    def synthesize(self, text: str, output_path: str) -> str:
        if self.tts is None:
            raise ValueError("No TTS model configured")
        return self.tts.synthesize(text, output_path)

    def realtime_transcription(self, chunk_duration: float = 3.0) -> dict:
        print("Real-time Speech-to-Text started...")
        print("Press ENTER to stop recording and transcription")
        print("-" * 50)

        chunk_frames = int(chunk_duration * self.sample_rate)
        audio_buffer = []
        complete_recording = []
        transcribed_chunks = []
        recording = True

        def stop_recording():
            nonlocal recording
            input()
            recording = False

        input_thread = threading.Thread(target=stop_recording)
        input_thread.daemon = True
        input_thread.start()

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            complete_recording.append(indata.copy())
            audio_buffer.extend(indata.flatten())

            if len(audio_buffer) >= chunk_frames:
                chunk_data = np.array(audio_buffer[:chunk_frames], dtype=np.float32)
                overlap_frames = int(0.5 * self.sample_rate)
                audio_buffer[:] = audio_buffer[chunk_frames - overlap_frames:]

                def transcribe_chunk():
                    try:
                        processed = self.audio_preprocessor.process(chunk_data, self.sample_rate)
                        text = self.asr.transcribe(processed, self.sample_rate).strip()
                        if text:
                            print(f"[live] {text}")
                            transcribed_chunks.append(text)
                    except Exception as e:
                        print(f"Transcription error: {e}")

                t = threading.Thread(target=transcribe_chunk)
                t.daemon = True
                t.start()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ):
            while recording:
                time.sleep(0.1)

        print("\nReal-time transcription stopped!")

        if not complete_recording:
            print("No audio was recorded")
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"realtime_recording_{timestamp}.wav"

        audio_data = np.concatenate(complete_recording, axis=0)
        audio_int16 = np.int16(audio_data * 32767)
        wavfile.write(output_path, self.sample_rate, audio_int16.flatten())
        duration_seconds = len(audio_data) / self.sample_rate
        print(f"Recording saved: {output_path} ({duration_seconds:.2f}s)")

        if not transcribed_chunks:
            print("No text captured")
            return {
                "original_audio": output_path,
                "compiled_text": "",
                "duration": duration_seconds,
            }

        compiled_text = " ".join(transcribed_chunks)
        if self.text_processor:
            compiled_text = self.text_processor.process(compiled_text)
        print(f"Compiled text: {compiled_text}")

        if self.tts is None:
            return {
                "original_audio": output_path,
                "compiled_text": compiled_text,
                "duration": duration_seconds,
                "chunk_count": len(transcribed_chunks),
            }

        try:
            tts_output_path = f"tts_output_{timestamp}.wav"
            self.tts.synthesize(compiled_text, tts_output_path)
            print(f"TTS generated: {tts_output_path}")
            return {
                "original_audio": output_path,
                "compiled_text": compiled_text,
                "tts_audio": tts_output_path,
                "duration": duration_seconds,
                "chunk_count": len(transcribed_chunks),
            }
        except Exception as e:
            print(f"TTS error: {e}")
            return {
                "original_audio": output_path,
                "compiled_text": compiled_text,
                "duration": duration_seconds,
                "error": str(e),
            }
