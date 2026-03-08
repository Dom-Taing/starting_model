import tempfile
import threading
import time
import noisereduce as nr
import pyloudnorm as pyln

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd


class AudioPreprocessor:
    """
    Preprocessing stage applied to raw audio before it reaches the ASR model.

    Current steps:
      1. Background noise reduction via noisereduce (spectral subtraction).
      2. Silence trimming — removes leading/trailing silence based on RMS energy,
         with a small configurable pad preserved as headroom. Internal pauses
         are left intact.
      3. LUFS normalization — scales audio to -20 LUFS (ITU-R BS.1770) to match
         the loudness level used during model training, ensuring consistent input
         regardless of mic distance.

    Future steps (TODO):
      - Resampling guard: ensure sr matches model expectations
    """

    def __init__(
        self,
        target_lufs: float = -20.0,
        silence_threshold_dbfs: float = -50.0,
        silence_pad_s: float = 0.1,
        debug: bool = False,
    ):
        self.target_lufs = target_lufs
        self.silence_threshold_dbfs = silence_threshold_dbfs
        self.silence_pad_s = silence_pad_s
        self.debug = debug

    def _trim_silence(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        # Convert dBFS threshold to linear RMS: rms = 10^(dBFS/20)
        threshold = 10 ** (self.silence_threshold_dbfs / 20)
        win = max(1, int(0.01 * sample_rate))  # 10ms windows
        rms = np.array([
            np.sqrt(np.mean(audio[i:i + win] ** 2))
            for i in range(0, len(audio), win)
        ])
        speech_windows = np.where(rms > threshold)[0]
        if len(speech_windows) == 0:
            return audio  # all silence — leave untouched

        pad_frames = int(self.silence_pad_s * sample_rate)
        start = max(0, speech_windows[0] * win - pad_frames)
        end   = min(len(audio), speech_windows[-1] * win + win + pad_frames)
        return audio[start:end]

    def _lufs_normalize(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        meter = pyln.Meter(sample_rate)  # ITU-R BS.1770
        loudness = meter.integrated_loudness(audio.astype(np.float64))
        if not np.isfinite(loudness):
            return audio  # silence or too short to measure — skip
        normalized = pyln.normalize.loudness(audio.astype(np.float64), loudness, self.target_lufs)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.debug:
            raw = audio.copy()
        audio = nr.reduce_noise(y=audio, sr=sample_rate).astype(np.float32)
        audio = self._trim_silence(audio, sample_rate)
        audio = self._lufs_normalize(audio, sample_rate)
        if self.debug:
            wavfile.write("debug_raw.wav", sample_rate, raw)
            wavfile.write("debug_filtered.wav", sample_rate, audio)
            print("[debug] saved debug_raw.wav and debug_filtered.wav")
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
