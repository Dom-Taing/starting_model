"""
VoxCPM TTS HTTP server.

Loads the model once at startup, then serves synthesis requests.

POST /synthesize
    Body:  {"text": "Hello world"}
    Returns: audio/wav bytes

GET /health
    Returns: {"status": "ok"}
"""

import io
import os

import soundfile as sf
import torch
import torchaudio
from flask import Flask, jsonify, request, send_file
from voxcpm import VoxCPM

# torchaudio ≥ 2.6 requires torchcodec which is not installed in this image.
# Patch torchaudio.load to use soundfile (already present) so voice-cloning
# prompt loading works without adding a new dependency or rebuilding.
def _sf_load(path, *args, **kwargs):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr  # [channels, samples]

torchaudio.load = _sf_load

app = Flask(__name__)

print("Loading VoxCPM model...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5")
print("Model ready.")

# Optional fixed voice reference — set these env vars to pin a single speaker.
# VOXCPM_REFERENCE_WAV  : path to a short (~5-10s) WAV of the target voice
# VOXCPM_REFERENCE_TEXT : verbatim transcript of that WAV
_REFERENCE_WAV  = os.environ.get("VOXCPM_REFERENCE_WAV")  or None
_REFERENCE_TEXT = os.environ.get("VOXCPM_REFERENCE_TEXT") or None

if _REFERENCE_WAV:
    print(f"Using fixed voice reference: {_REFERENCE_WAV}")
else:
    print("No reference voice set — VoxCPM will pick a random speaker per call.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "'text' must not be empty"}), 400

    wav = model.generate(
        text=text,
        prompt_wav_path=_REFERENCE_WAV,
        prompt_text=_REFERENCE_TEXT,
        cfg_value=2.0,
        inference_timesteps=10,
        normalize=False,
        denoise=False,
        retry_badcase=True,
        retry_badcase_max_times=3,
        retry_badcase_ratio_threshold=6.0,
    )

    buf = io.BytesIO()
    sf.write(buf, wav, model.tts_model.sample_rate, format="WAV")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
