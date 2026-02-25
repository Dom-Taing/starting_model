import tempfile
import threading
import time

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
from scipy.signal import butter, sosfilt


class AudioPreprocessor:
    """
    Preprocessing stage applied to raw audio before it reaches the ASR model.

    Current steps:
      1. Band-pass filter — keeps the speech frequency band (80 Hz – 8 kHz),
         removing low-frequency rumble (HVAC, handling noise) and high-frequency
         hiss simultaneously. Uses a 4th-order Butterworth filter, which has a
         maximally flat passband (no ripple) before rolling off outside the band.

         Why not just a low-pass?
         A low-pass alone would attenuate high-frequency consonants (s, f, t, sh
         live at 3–8 kHz), hurting transcription accuracy. The band-pass cuts
         noise on both ends while leaving the full speech band intact.

    Future steps (TODO):
      - Voice activity detection: skip silent chunks before ASR
      - Dynamic range normalisation: consistent loudness across recordings
      - Resampling guard: ensure sr matches model expectations
    """

    def __init__(
        self,
        highpass_hz: float = 80.0,
        lowpass_hz: float = 8000.0,
        filter_order: int = 4,
    ):
        self.highpass_hz = highpass_hz
        self.lowpass_hz = lowpass_hz
        self.filter_order = filter_order

    def _bandpass(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        nyquist = sample_rate / 2.0
        low = self.highpass_hz / nyquist
        high = min(self.lowpass_hz / nyquist, 0.9999)
        sos = butter(self.filter_order, [low, high], btype="band", output="sos")
        return sosfilt(sos, audio).astype(np.float32)

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = self._bandpass(audio, sample_rate)
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
