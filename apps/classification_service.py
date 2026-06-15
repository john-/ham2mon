#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow Classification Service.
Runs as a separate subprocess to isolate TensorFlow's signal handlers and threads.
"""

import sys
import argparse
from pathlib import Path
import numpy as np

try:
    import tensorflow as tf
except ImportError as error:
    print(f"tensorflow module did not load ({error})", file=sys.stderr)
    sys.exit(1)

class ServiceClassifier:
    def __init__(self, model_path: Path, audio_rate: int):
        self.audio_rate = audio_rate
        self.model = tf.lite.Interpreter(model_path=model_path.absolute().as_posix())
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

    def decode_audio(self, audio_binary):
        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)

    def get_spectrogram(self, file_path: str):
        audio_binary = tf.io.read_file(file_path)
        waveform = self.decode_audio(audio_binary)
        total_samples = tf.size(waveform).numpy()

        target_samples = self.audio_rate * 2  # 2 second clip
        middle = total_samples / 2
        start = int(middle - target_samples / 2)
        if start < 0:
            start = 0
        end = int(middle + target_samples / 2)
        waveform = waveform[start:end]

        zero_padding = tf.zeros([target_samples] - tf.shape(waveform), dtype=tf.float32)
        waveform = tf.cast(waveform, tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)

        spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)

        spectro = [spectrogram.numpy()]
        spectro = np.expand_dims(spectro, axis=-1)
        return spectro

    def predict(self, spectrogram):
        if spectrogram is None:
            return None
        self.model.set_tensor(self.input_details[0]['index'], spectrogram)
        self.model.invoke()
        prediction = self.model.get_tensor(self.output_details[0]['index'])
        types = ('V', 'D', 'S')
        return types[np.argmax(prediction[0])]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--audio_rate", type=int, required=True, help="Audio sample rate")
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
                result = "S" # default fallback to Skip if prediction fails

            sys.stdout.write(f"{result}\n")
            sys.stdout.flush()
        except KeyboardInterrupt:
            break
        except Exception:
            break

if __name__ == "__main__":
    main()
