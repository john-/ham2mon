#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 2023

Code is copied from:
    https://gitlab.com/john---/xmit_processor

@author: john
"""

from pathlib import Path
import numpy as np
import logging

try:
    import tensorflow as tf
except ImportError as error:
    raise Exception(f'tensorflow module did not load ({error})')

# stops log spamming for a harmless debug message
logging.getLogger("h5py").setLevel(logging.INFO)

class Classifier(object):

    def __init__(self, params: dict[str, bool], audio_rate: int):

        self.params = params
        self.audio_rate = audio_rate

        if all(value is False for value in self.params.values()):
            raise Exception('user does not want to classify audio')

        path = Path('model/model_1.tflite')

        try:
            path.resolve(strict=True)
            self.load_model(path)
            logging.info('Model loaded')
        except:
            raise
        
    def load_model(self, path: Path) -> None:
        self.model = tf.lite.Interpreter(model_path=path.absolute().as_posix())
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

    def is_wanted(self, file: str) -> str | None:

        spectrogram = self.get_spectrogram(file)
        detected_as = self.predict(spectrogram)

        wanted = detected_as if detected_as and self.params[detected_as] else None
        return wanted
        
    # convert the waveform into a spectrogram
    def get_spectrogram(self, file: str) -> tf.Tensor:

        audio_binary: tf.Tensor = tf.io.read_file(file)
        try:
            waveform: tf.Tensor = self.decode_audio(audio_binary)
        except Exception as error:
            logging.error(f'could not decode audio (try "-b 16" option or disable classification): {error}')
            raise

        total_samples: int = tf.size(waveform).numpy()
        sample_rate = self.audio_rate
        #length = round(total_samples/sample_rate, 3)

        clip_length = 2  # value may ve because model was trained with min. 2 sec clips
        target_samples = sample_rate * clip_length

        # Shorten to target length
        # TODO:  add this to preprocess for training and get rid of pre-shortening
        #        when creating training data
        #logging.debug('current elements: %d', tf.size(waveform).numpy())
        middle = total_samples/2
        start = int(middle - target_samples/2)
        if start < 0:
            start = 0
        end = int(middle + target_samples/2)
        waveform = waveform[start:end]
        #logging.debug('length: %d [ start: %d middle: %d end: %d]', tf.size(waveform).numpy(), start, middle, end)

        # Padding for files with less than 16000 samples (2 seconds of 8 Khz sample rate)
        zero_padding: tf.Tensor = tf.zeros([target_samples] - tf.shape(waveform), dtype=tf.float32)

        # Concatenate audio with padding so that all audio clips will be of the
        # same length
        waveform = tf.cast(waveform, tf.float32)
        equal_length: tf.Tensor = tf.concat([waveform, zero_padding], 0)
        spectrogram = tf.signal.stft(
            equal_length, frame_length=255, frame_step=128)

        spectrogram = tf.abs(spectrogram)

        spectro: tf.Tensor = []
        spectro.append(spectrogram.numpy())
        # logging.debug(f'{spectro =}')
        spectro = np.expand_dims(spectro, axis=-1) # TODO:  what is this dimension for?
        # logging.debug(f' after: {spectro =}')

        return spectro

    def decode_audio(self, audio_binary: tf.Tensor) -> tf.Tensor:
        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)

    def predict(self, spectrogram: tf.Tensor) -> str | None:
        if spectrogram is None:
            return None

        # prediction = model(spectrogram) # full TF

        input_data = spectrogram
        self.model.set_tensor(self.input_details[0]['index'], input_data)

        self.model.invoke()

        prediction = self.model.get_tensor(self.output_details[0]['index'])
        
        types = ['V', 'D', 'S']

        # this will extract probsbilties for each of the labels
        # predictions = {}
        # for index, a_pred in np.ndenumerate(prediction[0]):
        #     predictions[types[index[0]]] = round(float(a_pred), 3)

        return types[np.argmax(prediction[0])]
    
def main():
    """Test the classifier

    Sets up the classifier
    Classifies a couple files
    """

    audio_rate = 8000
    classifier_params={'V':True,'D':True,'S':True }
    try:
        classifier = Classifier(classifier_params, audio_rate)
    except Exception as error:
        print(f'classification disabled: {error}')
        classifier = False

    print('should be data (V): ' + classifier.is_wanted("test/voice.wav"))
    print('should be data (D): ' + classifier.is_wanted("test/data.wav"))
    print('should be skip (S): ' + classifier.is_wanted("test/skip.wav"))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
