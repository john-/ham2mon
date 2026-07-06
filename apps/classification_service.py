#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiteRT/TensorFlow Classification Service.
Runs as a separate subprocess to isolate signal handlers and threads.
"""

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

# Try importing LiteRT or TFLite runtimes
try:
    import ai_edge_litert.interpreter as litert

    Interpreter = litert.Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite

        Interpreter = tflite.Interpreter
    except ImportError:
        try:
            import tensorflow.lite as tflite

            Interpreter = tflite.Interpreter
        except ImportError:
            print(
                "Could not load any LiteRT/TFLite interpreter runtime. "
                "Please install one of: 'ai-edge-litert', 'tflite-runtime', or 'tensorflow'.",
                file=sys.stderr,
            )
            sys.exit(1)


class ServiceClassifier:
    def __init__(self, model_path: Path, audio_rate: int):
        self.audio_rate = audio_rate
        self.model = Interpreter(model_path=model_path.absolute().as_posix())
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

    def get_spectrogram(self, file_path: str):
        # 1. Decode WAV audio file using standard library wave module
        with wave.open(file_path, "rb") as w:
            nchannels, sampwidth, framerate, nframes = w.getparams()[:4]
            data = w.readframes(nframes)
            if sampwidth == 2:
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                audio = (
                    np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0
                ) / 128.0
            elif sampwidth == 4:
                audio = (
                    np.frombuffer(data, dtype=np.int32).astype(np.float32)
                    / 2147483648.0
                )
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth}")

            if nchannels > 1:
                audio = audio.reshape(-1, nchannels)[:, 0]

        # 2. Extract the target 2-second clip from the middle of the audio
        total_samples = len(audio)
        target_samples = self.audio_rate * 2  # 2 second clip
        middle = total_samples / 2
        start = int(middle - target_samples / 2)
        if start < 0:
            start = 0
        end = int(middle + target_samples / 2)
        waveform = audio[start:end]

        # 3. Apply zero padding if needed
        if len(waveform) < target_samples:
            pad_width = target_samples - len(waveform)
            waveform = np.pad(waveform, (0, pad_width), "constant")

        # 4. Compute STFT (Short-Time Fourier Transform) matching TF's behavior.
        # tf.signal.stft's default window_fn is a periodic Hann window of length M.
        # We mirror TF's parity-dependent raised_cosine_window quirk (where periodic
        # and symmetric coincide for odd window lengths) to remain compatible if
        # M is ever changed to an even number in the future.
        M = 255
        even = 1 - (M % 2)
        n = M + even - 1
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(M) / n)

        hop_length = 128
        num_frames = (len(waveform) - M) // hop_length + 1
        frames = []
        for i in range(num_frames):
            start_idx = i * hop_length
            frame = waveform[start_idx : start_idx + M]
            windowed_frame = frame * window
            # 256-point RFFT returns 129 bins
            rfft_out = np.fft.rfft(windowed_frame, n=256)
            frames.append(np.abs(rfft_out))

        spectrogram = np.array(frames, dtype=np.float32)
        spectro = np.expand_dims([spectrogram], axis=-1)
        return spectro

    def predict(self, spectrogram):
        if spectrogram is None:
            return None
        self.model.set_tensor(self.input_details[0]["index"], spectrogram)
        self.model.invoke()
        prediction = self.model.get_tensor(self.output_details[0]["index"])
        types = ("V", "D", "S")
        return types[np.argmax(prediction[0])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--audio_rate", type=int, required=True, help="Audio sample rate"
    )
    args = parser.parse_args()

    try:
        classifier = ServiceClassifier(Path(args.model), args.audio_rate)
    except Exception as e:
        print(f"Could not initialize ServiceClassifier: {e}", file=sys.stderr)
        sys.exit(1)

    # Ready signal to parent process
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    # Simple command loop: read filename, output classification
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            file_path = line.strip()
            if not file_path:
                continue

            try:
                spectrogram = classifier.get_spectrogram(file_path)
                result = classifier.predict(spectrogram)
            except Exception as pe:
                print(f"Prediction error for file '{file_path}': {pe}", file=sys.stderr)
                sys.stderr.flush()
                result = "S"  # default fallback to Skip if prediction fails

            sys.stdout.write(f"{result}\n")
            sys.stdout.flush()
        except KeyboardInterrupt:
            break
        except Exception:
            break


if __name__ == "__main__":
    main()
