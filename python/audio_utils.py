import tempfile
import threading
import time

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd


class AudioPreprocessor:
    """
    Preprocessing stage applied to raw audio before it reaches the ASR model.

    Currently a pass-through. Add steps here as needed, for example:
    # TODO: noise reduction (e.g. noisereduce library)
    # TODO: silence trimming / voice activity detection
    # TODO: dynamic range normalisation
    # TODO: band-pass filtering to remove out-of-band noise
    # TODO: resampling guard (ensure sr matches model expectations)
    """

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        # Pass-through — audio is returned unchanged until preprocessing is implemented
        return audio


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def record(self, duration: float, output_path: str = None) -> str:
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name

        print(f"Recording for {duration} seconds... Speak now!")
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float64",
        )
        sd.wait()

        audio_int16 = np.int16(audio_data * 32767)
        wavfile.write(output_path, self.sample_rate, audio_int16)
        print(f"Recording complete! Saved to: {output_path}")
        return output_path

    def record_stream(self, output_path: str = None) -> str:
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name

        print("Recording started... Press ENTER to stop recording!")

        recorded_data = []
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
            recorded_data.append(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ):
            while recording:
                time.sleep(0.1)

        print("Recording stopped!")

        if recorded_data:
            audio_data = np.concatenate(recorded_data, axis=0)
            audio_int16 = np.int16(audio_data * 32767)
            wavfile.write(output_path, self.sample_rate, audio_int16.flatten())
            duration_seconds = len(audio_data) / self.sample_rate
            print(f"Recording complete! Duration: {duration_seconds:.2f}s, Saved to: {output_path}")
            return output_path

        print("No audio recorded")
        return None
