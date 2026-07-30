#!/usr/bin/env python3
"""
LiteRT/TensorFlow Classification Service.
Runs as a separate subprocess to isolate signal handlers and threads.
"""

import argparse
import sys
import wave
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np


class InterpreterInstance(Protocol):
    """Protocol defining the structural interface of a TFLite/LiteRT Interpreter."""

    def allocate_tensors(self) -> None: ...
    def get_input_details(self) -> list[dict[str, object]]: ...
    def get_output_details(self) -> list[dict[str, object]]: ...
    def set_tensor(self, tensor_index: object, value: np.ndarray) -> None: ...
    def invoke(self) -> None: ...
    def get_tensor(self, tensor_index: object) -> np.ndarray: ...


InterpreterClass: object = None

try:
    import ai_edge_litert.interpreter as litert

    InterpreterClass = litert.Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite

        InterpreterClass = tflite.Interpreter
    except ImportError:
        try:
            import tensorflow.lite as tflite

            InterpreterClass = tflite.Interpreter
        except ImportError:
            _ = sys.stderr.write(
                "Could not load any LiteRT/TFLite interpreter runtime. "
                + "Please install an optional extra (e.g. `uv sync --extra ai-edge-litert` or `uv sync --extra tensorflow`).\n"
            )
            sys.exit(1)


class ServiceClassifier:
    """Subprocess TFLite classifier service wrapper."""

    audio_rate: int
    model: InterpreterInstance
    input_details: list[dict[str, object]]
    output_details: list[dict[str, object]]

    def __init__(self, model_path: Path, audio_rate: int) -> None:
        if not callable(InterpreterClass):
            raise TypeError("TFLite Interpreter class is not available")
        self.audio_rate = audio_rate
        # Instantiating dynamic interpreter class
        interpreter_obj: object = InterpreterClass(model_path=model_path.absolute().as_posix())
        self.model = cast(InterpreterInstance, interpreter_obj)
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

    def get_spectrogram(self, file_path: str) -> np.ndarray:
        """Decode WAV file and compute spectrogram matching TFLite model specs."""
        # 1. Decode WAV audio file using standard library wave module
        with wave.open(file_path, "rb") as w:
            nchannels, sampwidth, _framerate, nframes = w.getparams()[:4]
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
        start = max(int(middle - target_samples / 2), 0)
        end = int(middle + target_samples / 2)
        waveform = audio[start:end]

        # 3. Apply zero padding if needed
        if len(waveform) < target_samples:
            pad_width = target_samples - len(waveform)
            waveform = np.pad(waveform, (0, pad_width), "constant")

        # 4. Compute STFT (Short-Time Fourier Transform) matching TF's behavior.
        # tf.signal.stft's default window_fn is a periodic Hann window of length m_window.
        # We mirror TF's parity-dependent raised_cosine_window quirk (where periodic
        # and symmetric coincide for odd window lengths) to remain compatible if
        # m_window is ever changed to an even number in the future.
        m_window = 255
        even = 1 - (m_window % 2)
        n = m_window + even - 1
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(m_window) / n)

        hop_length = 128
        num_frames = (len(waveform) - m_window) // hop_length + 1
        frames: list[np.ndarray] = []
        for i in range(num_frames):
            start_idx = i * hop_length
            frame = waveform[start_idx : start_idx + m_window]
            windowed_frame = frame * window
            # 256-point RFFT returns 129 bins
            rfft_out = np.fft.rfft(windowed_frame, n=256)
            frames.append(np.abs(rfft_out))

        spectrogram = np.array(frames, dtype=np.float32)
        spectro = np.expand_dims([spectrogram], axis=-1)
        return spectro

    def predict(self, spectrogram: np.ndarray | None) -> Literal["V", "D", "S"] | None:
        """Run prediction on precomputed spectrogram."""
        if spectrogram is None:
            return None
        self.model.set_tensor(self.input_details[0]["index"], spectrogram)
        self.model.invoke()
        prediction = self.model.get_tensor(self.output_details[0]["index"])
        types: tuple[Literal["V"], Literal["D"], Literal["S"]] = ("V", "D", "S")
        pred_idx = int(np.argmax(cast(np.ndarray, prediction[0])))
        return types[pred_idx]


def main() -> None:
    """Subprocess main command loop."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--model", type=str, required=True, help="Path to model")
    _ = parser.add_argument(
        "--audio-rate", type=int, required=True, help="Audio sample rate"
    )
    args = parser.parse_args()

    model_arg = str(cast(str, args.model))
    rate_arg = int(cast(int, args.audio_rate))
    try:
        classifier = ServiceClassifier(Path(model_arg), rate_arg)
    except Exception as e:
        _ = sys.stderr.write(f"Could not initialize ServiceClassifier: {e}\n")
        sys.exit(1)

    # Ready signal to parent process
    _ = sys.stdout.write("READY\n")
    _ = sys.stdout.flush()

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
                _ = sys.stderr.write(f"Prediction error for file '{file_path}': {pe}\n")
                _ = sys.stderr.flush()
                result = "S"  # default fallback to Skip if prediction fails

            _ = sys.stdout.write(f"{result}\n")
            _ = sys.stdout.flush()
        except KeyboardInterrupt:
            break
        except Exception:
            break


if __name__ == "__main__":
    main()
