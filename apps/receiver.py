#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:38:36 2015

@author: madengr
"""

from gnuradio import gr  # type: ignore
import osmosdr  # type: ignore
from gnuradio import blocks
from gnuradio import fft
from gnuradio.fft import window  # type: ignore
from gnuradio import audio
import os
import glob
import errno
import time
import numpy as np
import logging

from demodulators.NBFM import TunerDemodNBFM
from demodulators.AM import TunerDemodAM
from demodulators.WBFM import TunerDemodWBFM
from classification import Classifier
from channel_loggers import ChannelLogParams, ChannelLogger

class Receiver(gr.top_block):
    """Receiver for NBFM and AM modulation

    Controls hardware and instantiates multiple tuner/demodulators
    Generates FFT power spectrum for channel estimation

    Args:
        ask_samp_rate (float): Asking sample rate of hardware in sps (1E6 min)
        num_demod (int): Number of parallel demodulators
        type_demod (int): Type of demodulator (0=NBFM, 1=AM)
        hw_args (string): Argument string to pass to hardware
        freq_correction (int): Frequency correction in ppm
        record (bool): Record audio to file if True
        audio_bps (int): Audio bit depth in bps (bits/samples)

    Attributes:
        center_freq (int): Hardware RF center frequency in Hz
        samp_rate (int): Hardware sample rate in sps (1E6 min)
        gain_db (int): Hardware RF gain in dB
        squelch_db (int): Squelch in dB
        volume_dB (int): Volume in dB
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments

    def __init__(self, ask_samp_rate: int, num_demod: int, type_demod: int,
                 hw_args: str, freq_correction: int, record: bool, play: bool,
                 audio_bps: int, min_recording: float,
                 classifier_params: dict, channel_log_params: ChannelLogParams):

        # Call the initialization method from the parent class
        gr.top_block.__init__(self, "Receiver")

        # Make sure the 'wav' directory exists
        try:
            os.makedirs('wav/tmp')
        except OSError as error:  # will need to add something here for Win support
            if error.errno == errno.EEXIST:
                # remove any existing wav files
                for f in glob.glob('wav/tmp/*.wav'):
                    os.unlink(f)
            else:
                raise

        # Default values
        self.center_freq: int = int(144E6)
        self.samp_rate: int
        self.squelch_db = -60
        self.volume_db = 0
        audio_rate = 8000

        # Setup the USRP source, or use the USRP sim
        self.src = osmosdr.source(args="numchan=" + str(1) + " " + hw_args)
        self.src.set_sample_rate(ask_samp_rate)
        self.src.set_center_freq(self.center_freq)
        self.src.set_freq_corr(freq_correction)

        # Get the sample rate and center frequency from the hardware
        self.samp_rate = self.src.get_sample_rate()
        self.center_freq = self.src.get_center_freq()

        # Set the I/Q bandwidth to 80 % of sample rate
        self.src.set_bandwidth(0.8 * self.samp_rate)

        # NBFM channel is about 10 KHz wide
        # Want  about 3 FFT bins to span a channel
        # Use length FFT so 4 Msps / 1024 = 3906.25 Hz/bin
        # This also means 3906.25 vectors/second
        # Using below formula keeps FFT size a power of two
        # Also keeps bin size constant for power of two sampling rates
        # Use of 256 sets 3906.25 Hz/bin; increase to reduce bin size
        samp_ratio = self.samp_rate / 1E6
        fft_length = 256 * int(pow(2, np.ceil(np.log(samp_ratio)/np.log(2))))

        # -----------Flow for FFT--------------

        # Convert USRP steam to vector
        stream_to_vector = blocks.stream_to_vector(gr.sizeof_gr_complex*1,
                                                   fft_length)

        # Want about 1000 vector/sec
        amount = int(round(self.samp_rate/fft_length/1000))
        keep_one_in_n = blocks.keep_one_in_n(gr.sizeof_gr_complex*
                                             fft_length, amount)

        # Take FFT
        fft_vcc = fft.fft_vcc(fft_length, True,
                              window.blackmanharris(fft_length), True, 1)

        # Compute the power
        complex_to_mag_squared = blocks.complex_to_mag_squared(fft_length)

        # Video average and decimate from 1000 vector/sec to 10 vector/sec
        integrate_ff = blocks.integrate_ff(100, fft_length)

        # Probe vector
        self.probe_signal_vf = blocks.probe_signal_vf(fft_length)

        # Connect the blocks
        self.connect(self.src, stream_to_vector, keep_one_in_n,
                     fft_vcc, complex_to_mag_squared,
                     integrate_ff, self.probe_signal_vf)

        classifier: Classifier | None
        try:
          classifier = Classifier(classifier_params, audio_rate)
        except Exception as error:
            logging.info(f'classification disabled: {error}')
            classifier = None

        channel_logger = ChannelLogger.get_logger(channel_log_params)

        # -----------Flow for Demod--------------

        # Create N parallel demodulators as a list of objects
        # Default to NBFM demod
        self.demodulators = []
        for idx in range(num_demod):
            if type_demod == 0:
                self.demodulators.append(TunerDemodNBFM(self.samp_rate,
                                                        audio_rate, record,
                                                        audio_bps,
                                                        min_recording,
                                                        classifier,
                                                        channel_logger))
            elif type_demod == 1:
                self.demodulators.append(TunerDemodAM(self.samp_rate,
                                                      audio_rate, record,
                                                      audio_bps,
                                                      min_recording,
                                                      classifier,
                                                      channel_logger))
            elif type_demod == 2:
                self.demodulators.append(TunerDemodWBFM(self.samp_rate,
                                                        audio_rate, record,
                                                        audio_bps,
                                                        min_recording,
                                                        classifier,
                                                        channel_logger))
            else:
                raise Exception(f'Invalid demodulator type: {type_demod}')

        if play:
            # Create an adder
            add_ff = blocks.add_ff(1)

            # Connect the demodulators between the source and adder
            for idx, demodulator in enumerate(self.demodulators):
                self.connect(self.src, demodulator, (add_ff, idx))

            # Audio sink
            audio_sink = audio.sink(audio_rate)

            # Connect the summed outputs to the audio sink
            self.connect(add_ff, audio_sink)
        else:
            # Just connect each demodulator to the receiver source
            for demodulator in self.demodulators:
                self.connect(self.src, demodulator)


    def set_center_freq(self, center_freq: int) -> None:
        """Sets RF center frequency of hardware

        Args:
            center_freq (int): Hardware RF center frequency in Hz
        """
        # Tune the hardware
        self.src.set_center_freq(center_freq)

        # Update center frequency with hardware center frequency
        # Do this to account for slight hardware offsets
        self.center_freq = self.src.get_center_freq()

    def get_gain_names(self) -> list[dict]:
        """Get the list of supported gain elements
        """
        return self.src.get_gain_names()

    def filter_and_set_gains(self, all_gains: list[dict]) -> list[dict]:
        """Remove unsupported gains and set them
        Args:
            all_gains (list of dictionary): Supported gains in dB
        """
        gains: list[dict] = []
        names = self.get_gain_names()
        for gain in all_gains:
            if gain["name"] in names:
                gains.append(gain)
        return self.set_gains(gains)

    def set_gains(self, gains: list[dict]) -> list[dict]:
        """Set all the gains
        Args:
            gains (list of dictionary): Supported gains in dB
        """
        for gain in gains:
            self.src.set_gain(gain["value"], gain["name"])
            gain["value"] = self.src.get_gain(gain["name"])
        self.gains = gains
        return self.gains

    def set_squelch(self, squelch_db: int) -> None:
        """Sets squelch of all demodulators and clamps range

        Args:
            squelch_db (int): Squelch in dB
        """
        self.squelch_db = max(min(0, squelch_db), -100)
        for demodulator in self.demodulators:
            demodulator.set_squelch(self.squelch_db)

    def set_volume(self, volume_db: int) -> None:
        """Sets volume of all demodulators and clamps range

        Args:
            volume_db (int): Volume in dB
        """
        self.volume_db = max(min(20, volume_db), -20)
        for demodulator in self.demodulators:
            demodulator.set_volume(self.volume_db)

    def get_demod_freqs(self) -> list[int]:
        """Gets baseband frequencies of all demodulators

        Returns:
            List[float]: List of baseband center frequencies in Hz
        """
        center_freqs: list[int] = []
        for demodulator in self.demodulators:
            center_freqs.append(demodulator.center_freq)
        return center_freqs

    def __del__(self):
        """Called when the object is destroyed."""
        # Make a best effort attempt to clean up our wavfile if it's empty
        try:
            for f in glob.glob('wav/tmp/*.wav'):
                os.unlink(f)
            os.rmdir('wav/tmp')
        except Exception:
            pass  # oh well, we're dying anyway

async def main():
    """Test the receiver

    Sets up the hardware
    Tunes a couple of demodulators
    Prints the max power spectrum
    """

    import h2m_parser as prsr
    from h2m_types import ChannelMessage

        # Create parser object
    parser = prsr.CLParser()
    import sys


    if not len(sys.argv) > 1:
        parser.print_help() #pylint: disable=maybe-no-member
        raise SystemExit(1)

    # Create receiver object

    # use command line args for some things
    hw_args = parser.hw_args
    ask_samp_rate = parser.ask_samp_rate
    play = parser.play

    # hardcode the rest
    num_demod = 4
    type_demod = 0
    freq_correction = 0
    record = False
    audio_bps = 8
    min_recording=1.0
    classifier_params={'V':False,'D':False,'S':False }

    async def print_to_screen(msg: ChannelMessage):  # callback used when channel opens 
        print(f'Opened channel {msg.channel-1}')  # channel is a 1 based representation of the demod
    
    channel_log_params=ChannelLogParams(type='none', target='', timeout=0, notify_scanner=print_to_screen)
    receiver = Receiver(ask_samp_rate, num_demod, type_demod, hw_args,
                        freq_correction, record, play, audio_bps,
                        min_recording, classifier_params, channel_log_params)

    # Start the receiver and wait for samples to accumulate
    receiver.start()
    time.sleep(1)

    # Set frequency, gain, squelch, and volume
    center_freq = int(144.5E6)
    receiver.set_center_freq(center_freq)
    print("\n")
    print("Started %s at %.3f Msps" % (hw_args, receiver.samp_rate/1E6))
    print("RX at %.3f MHz" % (receiver.center_freq/1E6))
    # gain needs to be one supported by hw_args
    receiver.filter_and_set_gains([{ "name": "RF", "value": 10.0 }])
    for gain in receiver.gains:
        print("gain %s at %d dB" % (gain["name"], gain["value"]))
    receiver.set_squelch(-60)
    receiver.set_volume(0)
    print("%d demods of type %d at %d dB squelch and %d dB volume" % \
        (num_demod, type_demod, receiver.squelch_db, receiver.volume_db))

    # Create some baseband channels to tune based on 144 MHz center
    channels = np.zeros(num_demod)
    channels[0] = 144.39E6 - receiver.center_freq # APRS
    channels[1] = 144.6E6 - receiver.center_freq

    # Tune demodulators to baseband channels
    # If recording on, this creates empty wav file since manually tuning.
    for idx, demodulator in enumerate(receiver.demodulators):
        await demodulator.set_center_freq(channels[idx], center_freq)

    # Print demodulator info
    for idx, channel in enumerate(channels):
        print("Tuned demod %d to %.3f MHz" % (idx,
                                              (channel+receiver.center_freq)
                                              /1E6))

    while 1:
        # No need to go faster than 10 Hz rate of GNU Radio probe
        # Just do 1 Hz here
        time.sleep(1)

        # Grab the FFT data and print max value
        spectrum = receiver.probe_signal_vf.level()
        print("Max spectrum of %.3f" % (np.max(spectrum)))

    # Stop the receiver
    receiver.stop()
    receiver.wait()


if __name__ == '__main__':

    import asyncio

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
