import argparse
import sys

from audio_utils import AudioPreprocessor
from models.asr.wav2vec2 import Wav2Vec2ASR
from models.asr.whisper import WhisperASR
from models.tts.piper import PiperTTS
from models.tts.voxcpm import VoxCPMTTS
from pipeline import SpeechPipeline
from text_utils import TextProcessor


def build_pipeline(args) -> SpeechPipeline:
    # ASR
    device = 0 if args.gpu else -1
    if args.asr == "wav2vec2":
        asr = Wav2Vec2ASR(device=device)
    elif args.asr == "whisper":
        asr = WhisperASR(device=device)
    else:
        print(f"Unknown ASR model: {args.asr}", file=sys.stderr)
        sys.exit(1)

    # TTS
    if args.tts == "piper":
        tts = PiperTTS(model_path=args.piper_model)
    elif args.tts == "voxcpm":
        tts = VoxCPMTTS(server_url=args.voxcpm_url)
    elif args.tts == "none":
        tts = None
    else:
        print(f"Unknown TTS model: {args.tts}", file=sys.stderr)
        sys.exit(1)

    # Text processor
    text_processor = TextProcessor() if args.autocorrect else None

    audio_preprocessor = AudioPreprocessor(debug=args.debug_audio)

    return SpeechPipeline(
        asr=asr,
        tts=tts,
        sample_rate=args.sample_rate,
        text_processor=text_processor,
        audio_preprocessor=audio_preprocessor,
    )


def main():
    parser = argparse.ArgumentParser(description="Speech pipeline: ASR + TTS")
    parser.add_argument("--asr", choices=["wav2vec2", "whisper"], default="wav2vec2",
                        help="ASR model to use (default: wav2vec2)")
    parser.add_argument("--tts", choices=["piper", "voxcpm", "none"], default="piper",
                        help="TTS model to use (default: piper)")
    parser.add_argument("--mode", choices=["realtime", "file", "stream"], default="realtime",
                        help="Pipeline mode (default: realtime)")
    parser.add_argument("--input", metavar="PATH",
                        help="Input audio file path (required for file mode)")
    parser.add_argument("--output", metavar="PATH", default="output.wav",
                        help="Output audio path (default: output.wav)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Audio sample rate in Hz (default: 16000)")
    parser.add_argument("--chunk-duration", type=float, default=3.0,
                        help="Chunk duration for realtime mode in seconds (default: 3.0)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (device=0) for ASR")
    parser.add_argument("--piper-model", default="voice_model/en_US-lessac-medium.onnx",
                        metavar="PATH", help="Path to Piper ONNX model")
    parser.add_argument("--voxcpm-url", default="http://localhost:8080",
                        metavar="URL", help="VoxCPM server URL")
    parser.add_argument("--autocorrect", action="store_true",
                        help="Enable SymSpell autocorrect on transcribed text")
    parser.add_argument("--silence-threshold", type=float, default=0.01,
                        metavar="RMS", help="RMS threshold for voice activity (default: 0.01)")
    parser.add_argument("--trailing-silence", type=float, default=0.8,
                        metavar="SECS", help="Seconds of silence that ends an utterance (default: 0.8)")
    parser.add_argument("--min-speech", type=float, default=0.3,
                        metavar="SECS", help="Minimum speech duration to accept (default: 0.3)")
    parser.add_argument("--debug-audio", action="store_true",
                        help="Save raw and filtered audio to debug_raw.wav / debug_filtered.wav (debug only)")

    args = parser.parse_args()

    if args.mode == "file" and not args.input:
        parser.error("--input is required for file mode")
    if args.mode == "stream" and args.tts == "none":
        parser.error("--tts none is not allowed with --mode stream; choose piper or voxcpm")

    pipeline = build_pipeline(args)

    if args.mode == "file":
        text = pipeline.transcribe_file(args.input)
        print(f"Transcription: {text}")
        if pipeline.tts is not None:
            out = pipeline.synthesize(text, args.output)
            print(f"TTS saved to: {out}")
    elif args.mode == "stream":
        pipeline.stream(
            silence_threshold=args.silence_threshold,
            trailing_silence_s=args.trailing_silence,
            min_speech_duration_s=args.min_speech,
        )
    else:
        result = pipeline.realtime_transcription(chunk_duration=args.chunk_duration)
        if result:
            print(f"\nFinal text: {result.get('compiled_text', '')}")


if __name__ == "__main__":
    main()
