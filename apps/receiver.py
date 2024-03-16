#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:38:36 2015

@author: madengr
"""

from gnuradio import gr  # type: ignore
import osmosdr  # type: ignore
from gnuradio import filter as grfilter # Don't redefine Python's filter()
from gnuradio import blocks
from gnuradio import fft
from gnuradio.fft import window  # type: ignore
from gnuradio import analog
from gnuradio import audio
import os
import glob
import errno
import time
import numpy as np
from gnuradio.filter import pfb  # type: ignore
from classification import Classifier
import logging
from utilities import baseband_to_frequency
import asyncio
import channel_loggers as loggers
from h2m_types import ChannelMessage

class BaseTuner(gr.hier_block2):
    """Some base methods that are the same between the known tuner types.

    See TunerDemodNBFM and TunerDemodAM for better documentation.
    """
    def __init__(self, classify: Classifier | None, channel_logger: loggers.ChannelLogger, channel: int) -> None:
        # Default values
        self.classify = classify
        self.channel_logger = channel_logger
        self.channel = channel
        self.last_heard: float = 0.0
        self.file_name: str | None = None
        self.log_task: asyncio.Task | None = None
        self.center_freq: int

    def set_last_heard(self, a_time: float) -> None:
        self.last_heard = a_time
        # channel_log active channel if at required interval
        # alternately use a timer or something that is created on demod start

    async def set_center_freq(self, center_freq: int, rf_center_freq: int) -> None:
        """Sets baseband center frequency and file name

        Sets baseband center frequency of frequency translating FIR filter
        Also sets file name of wave file sink
        If tuner is tuned to zero Hz then set to file name to None
        Otherwise set file name to tuned RF frequency in MHz

        Args:
            center_freq (int): Baseband center frequency in Hz
            rf_center_freq (int): RF center in Hz (for file name)
        """
        # address completed transmissions
        results: ChannelMessage | None
        if self.record:
            # Move file from tmp directory if it is long enough
            # and classified appropriately
            results = self._persist_wavfile(rf_center_freq)   # also get channel_log information
        elif self.center_freq != 0:
            # not recording files and center_freq has changed
            results = ChannelMessage(state='off',
                                     frequency=baseband_to_frequency(
                                         self.center_freq, rf_center_freq),
                                     channel=self.channel)
        else:
            # center_freq is 0
            results = None
            
        await self.channel_logger.log(results)  # off events or nothing to note

        # Set the frequency of the tuner
        self.center_freq = center_freq
        self.freq_xlating_fir_filter_ccc.set_center_freq(self.center_freq)

        # Set the file name if recording
        if self.center_freq == 0 or not self.record:
            # If tuner at zero Hz, or record false, then file name to None
            self.file_name = None
        else:
            self.time_stamp = time.time()  # used for file naming and checking max_recording length
            self.set_file_name(rf_center_freq)

        if (self.file_name is not None and self.record):
            self.blocks_wavfile_sink.open(self.file_name)

        if self.center_freq != 0:
            await self.channel_logger.log(ChannelMessage(state='on',
                                                   frequency=baseband_to_frequency(
                                                       self.center_freq, rf_center_freq),
                                                   channel=self.channel))

    def set_file_name(self, rf_center_freq: int) -> None:
        # Use frequency and time stamp for file name
        tstamp = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + "{:.3f}".format(self.time_stamp%1)[1:]
        file_freq = (rf_center_freq + self.center_freq)/1E6  # TODO: use utilities function
        file_freq = np.round(file_freq, 4)
        # avoid "chatter" of possibly unwanted files by working in tmp dir initially
        self.file_name = f'wav/tmp/{file_freq:.4f}_{tstamp}.wav'

    def _persist_wavfile(self, rf_center_freq: int) -> ChannelMessage | None:
        """Save the current wavfile if duration long enough"""
        if (not self.file_name or
                self.file_name is None):
            # currently recording and transmission started
            return None
        
        self.blocks_wavfile_sink.close()

        # base message used for channel log
        xmit_msg = ChannelMessage(state='off',
                                  frequency=baseband_to_frequency(
                                      self.center_freq, rf_center_freq),
                                  channel=self.channel)

        # Delete short wavfiles otherwise move ones that are long enough
        min_size = 44 + self.audio_bps*1000 * self.min_recording
        if os.stat(self.file_name).st_size <= min_size:
            os.unlink(self.file_name)
            xmit_msg.detail = 'Discarded short recording'
            return xmit_msg

        # If not classifying then move from tmp directory
        if not self.classify:
            name = self.file_name.replace('tmp/', '')
            os.rename(self.file_name, name)
            xmit_msg.file = name
            return xmit_msg

        # If user wants file of this classification
        # then move from tmp directory and rename
        # otherwise delete it
        is_wanted = self.classify.is_wanted(self.file_name)
        if  is_wanted:
            name = self.file_name.replace('tmp/', '')
            name = name.replace('.wav', '_' + is_wanted + '.wav')
            os.rename(self.file_name, name)
            
            # if using recent python3 have self.file_name be a Path
            # new_name = PurePath(self.file_name)
            # name = f'wav/{new_name.stem}_{is_wanted}{new_name.suffix}'
            xmit_msg.file = name
            xmit_msg.classification = is_wanted
            return xmit_msg
        else:
            os.unlink(self.file_name)
            xmit_msg.detail = 'Discarded unwanted classification'
            return  xmit_msg
    
    def set_squelch(self, squelch_db: int) -> None:
        """Sets the threshold for both squelches

        Args:
            squelch_db (int): Squelch in dB
        """
        self.analog_pwr_squelch_cc.set_threshold(squelch_db)

class TunerDemodNBFM(BaseTuner):
    """Tuner, demodulator, and recorder chain for narrow band FM demodulation

    Kept as it's own class so multiple can be instantiated in parallel
    Accepts complex baseband samples at 1 Msps minimum
    Frequency translating FIR filter tunes from -samp_rate/2 to +samp_rate/2
    The following sample rates assume 1 Msps input
    First two stages of decimation are 5 each for a total of 25
    Thus first two stages brings 1 Msps down to 40 ksps
    The third stage decimates by int(samp_rate/1E6)
    Thus output rate will vary from 40 ksps to 79.99 ksps
    The channel is filtered to 12.5 KHz bandwidth followed by squelch
    The squelch is non-blocking since samples will be added with other demods
    The quadrature demod is followed by a fourth stage of decimation by 5
    This brings the sample rate down to 8 ksps to 15.98 ksps
    The audio is low-pass filtered to 3.5 kHz bandwidth
    The polyphase resampler resamples by samp_rate/(decims[1] * decims[0]**3)
    This results in a constant 8 ksps, irrespective of RF sample rate
    This 8 ksps audio stream may be added to other demod streams
    The audio is run through an additional blocking squelch at -200 dB
    This stops the sample flow so squelched audio is not recorded to file
    The wav file sink stores 8-bit samples (default/grainy quality but compact)
    Default demodulator center frequency is 0 Hz
    This is desired since hardware DC removal reduces sensitivity at 0 Hz
    NBFM demod of LO leakage will just be 0 amplitude

    Args:
        samp_rate (int): Input baseband sample rate in sps (1E6 minimum)
        audio_rate (int): Output audio sample rate in sps (8 kHz minimum)
        record (bool): Record audio to file if True
        audio_bps (int): Audio bit depth in bps (bits/samples)
        min_recording (float): Minimum length of a recording in seconds

    Attributes:
        center_freq (int): Baseband center frequency in Hz
        record (bool): Record audio to file if True
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, samp_rate: int, audio_rate: int, record: bool,
                 audio_bps: int, min_recording: float, classify: Classifier | None,
                 channel_logger: loggers.ChannelLogger, channel: int):
        gr.hier_block2.__init__(self, "TunerDemodNBFM",
                                gr.io_signature(1, 1, gr.sizeof_gr_complex),
                                gr.io_signature(1, 1, gr.sizeof_float))

        super().__init__(classify, channel_logger, channel)

        # Default values
        self.center_freq = 0
        squelch_db = -60
        self.quad_demod_gain = 0.050
        self.file_name = None
        self.record = record
        self.audio_bps = audio_bps
        self.min_recording = min_recording

        # Decimation values for four stages of decimation
        decims = (5, int(samp_rate/1E6))

        # Low pass filter taps for decimation by 5
        low_pass_filter_taps_0 = \
            grfilter.firdes.low_pass(1, 1, 0.090, 0.010,
                    window.WIN_HAMMING)

        # Frequency translating FIR filter decimating by 5
        self.freq_xlating_fir_filter_ccc = \
            grfilter.freq_xlating_fir_filter_ccc(decims[0],
                                                 low_pass_filter_taps_0,
                                                 self.center_freq, samp_rate)

        # FIR filter decimating by 5
        fir_filter_ccc_0 = grfilter.fir_filter_ccc(decims[0],
                                                   low_pass_filter_taps_0)

        # Low pass filter taps for decimation from samp_rate/25 to 40-79.9 ksps
        # In other words, decimation by int(samp_rate/1E6)
        # 12.5 kHz cutoff for NBFM channel bandwidth
        low_pass_filter_taps_1 = grfilter.firdes.low_pass(
            1, samp_rate/decims[0]**2, 12.5E3, 1E3, window.WIN_HAMMING)

        # FIR filter decimation by int(samp_rate/1E6)
        fir_filter_ccc_1 = grfilter.fir_filter_ccc(decims[1],
                                                   low_pass_filter_taps_1)

        # Non blocking power squelch
        self.analog_pwr_squelch_cc = analog.pwr_squelch_cc(squelch_db,
                                                           1e-1, 0, False)

        # Quadrature demod with gain set for decent audio
        # The gain will be later multiplied by the 0 dB normalized volume
        self.analog_quadrature_demod_cf = \
            analog.quadrature_demod_cf(self.quad_demod_gain)

        # 3.5 kHz cutoff for audio bandwidth
        low_pass_filter_taps_2 = grfilter.firdes.low_pass(1,\
                        samp_rate/(decims[1] * decims[0]**2),\
                        3.5E3, 500, window.WIN_HAMMING)

        # FIR filter decimating by 5 from 40-79.9 ksps to 8-15.98 ksps
        fir_filter_fff_0 = grfilter.fir_filter_fff(decims[0],
                                                   low_pass_filter_taps_2)

        # Polyphase resampler allows arbitary RF sample rates
        # Takes 8-15.98 ksps to a constant 8 ksps for audio
        pfb_resamp = audio_rate/float(samp_rate/(decims[1] * decims[0]**3))
        pfb_arb_resampler_fff = pfb.arb_resampler_fff(pfb_resamp, taps=None,
                                                      flt_size=32)

        # Connect the blocks for the demod
        self.connect(self, self.freq_xlating_fir_filter_ccc)
        self.connect(self.freq_xlating_fir_filter_ccc, fir_filter_ccc_0)
        self.connect(fir_filter_ccc_0, fir_filter_ccc_1)
        self.connect(fir_filter_ccc_1, self.analog_pwr_squelch_cc)
        self.connect(self.analog_pwr_squelch_cc,
                     self.analog_quadrature_demod_cf)
        self.connect(self.analog_quadrature_demod_cf, fir_filter_fff_0)
        self.connect(fir_filter_fff_0, pfb_arb_resampler_fff)
        self.connect(pfb_arb_resampler_fff, self)

        # Need to set this to a very low value of -200 since it is after demod
        # Only want it to gate when the previous squelch has gone to zero
        analog_pwr_squelch_ff = analog.pwr_squelch_ff(-200, 1e-1, 0, True)

        # Connect the blocks for recording
        self.connect(pfb_arb_resampler_fff, analog_pwr_squelch_ff)

        # File sink with single channel and bits/sample
        if (self.record):
            self.blocks_wavfile_sink = blocks.wavfile_sink('/dev/null', 1,
                                                       audio_rate,
                                                       blocks.FORMAT_WAV,
                                                       blocks.FORMAT_PCM_16,
                                                       False)
            self.connect(analog_pwr_squelch_ff, self.blocks_wavfile_sink)
        else:
            null_sink1 = blocks.null_sink(gr.sizeof_float)
            self.connect(analog_pwr_squelch_ff, null_sink1)

    def set_volume(self, volume_db: int) -> None:
        """Sets the volume

        Args:
            volume_db (int): Volume in dB
        """
        gain = self.quad_demod_gain * 10**(volume_db/20.0)
        self.analog_quadrature_demod_cf.set_gain(gain)

class TunerDemodAM(BaseTuner):
    """Tuner, demodulator, and recorder chain for AM demodulation

    Kept as it's own class so multiple can be instantiated in parallel
    Accepts complex baseband samples at 1 Msps minimum
    Frequency translating FIR filter tunes from -samp_rate/2 to +samp_rate/2
    The following sample rates assume 1 Msps input
    First two stages of decimation are 5 each for a total of 25
    Thus first two stages brings 1 Msps down to 40 ksps
    The third stage decimates by int(samp_rate/1E6)
    Thus output rate will vary from 40 ksps to 79.99 ksps
    The channel is filtered to 12.5 KHz bandwidth followed by squelch
    The squelch is non-blocking since samples will be added with other demods
    The AGC sets level (volume) prior to AM demod
    The AM demod is followed by a fourth stage of decimation by 5
    This brings the sample rate down to 8 ksps to 15.98 ksps
    The audio is low-pass filtered to 3.5 kHz bandwidth
    The polyphase resampler resamples by samp_rate/(decims[1] * decims[0]**3)
    This results in a constant 8 ksps, irrespective of RF sample rate
    This 8 ksps audio stream may be added to other demod streams
    The audio is run through an additional blocking squelch at -200 dB
    This stops the sample flow so squelced audio is not recorded to file
    The wav file sink stores 8-bit samples (default/grainy quality but compact)
    Default demodulator center frequency is 0 Hz
    This is desired since hardware DC removal reduces sensitivity at 0 Hz
    AM demod of LO leakage will just be 0 amplitude

    Args:
        samp_rate (int): Input baseband sample rate in sps (1E6 minimum)
        audio_rate (int): Output audio sample rate in sps (8 kHz minimum)
        record (bool): Record audio to file if True
        audio_bps (int): Audio bit depth in bps (bits/samples)
        min_recording (float): Minimum length of a recording in seconds

    Attributes:
        center_freq (int): Baseband center frequency in Hz
        record (bool): Record audio to file if True
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-locals

    def __init__(self, samp_rate: int, audio_rate: int, record: bool,
                 audio_bps: int, min_recording: float, classify: Classifier | None,
                 channel_logger: loggers.ChannelLogger, channel: int):
        gr.hier_block2.__init__(self, "TunerDemodAM",
                                gr.io_signature(1, 1, gr.sizeof_gr_complex),
                                gr.io_signature(1, 1, gr.sizeof_float))

        super().__init__(classify, channel_logger, channel)

        # Default values
        self.center_freq = 0
        squelch_db = -60
        self.agc_ref = 0.1
        self.file_name = None
        self.record = record
        self.audio_bps = audio_bps
        self.min_recording = min_recording

        # Decimation values for four stages of decimation
        decims = (5, int(samp_rate/1E6))

        # Low pass filter taps for decimation by 5
        low_pass_filter_taps_0 = \
            grfilter.firdes.low_pass(1, 1, 0.090, 0.010,
                                     window.WIN_HAMMING)

        # Frequency translating FIR filter decimating by 5
        self.freq_xlating_fir_filter_ccc = \
            grfilter.freq_xlating_fir_filter_ccc(decims[0],
                                                 low_pass_filter_taps_0,
                                                 self.center_freq, samp_rate)

        # FIR filter decimating by 5
        fir_filter_ccc_0 = grfilter.fir_filter_ccc(decims[0],
                                                   low_pass_filter_taps_0)

        # Low pass filter taps for decimation from samp_rate/25 to 40-79.9 ksps
        # In other words, decimation by int(samp_rate/1E6)
        # 12.5 kHz cutoff for NBFM channel bandwidth
        low_pass_filter_taps_1 = grfilter.firdes.low_pass(
            1, samp_rate/decims[0]**2, 12.5E3, 1E3, window.WIN_HAMMING)

        # FIR filter decimation by int(samp_rate/1E6)
        fir_filter_ccc_1 = grfilter.fir_filter_ccc(decims[1],
                                                   low_pass_filter_taps_1)

        # Non blocking power squelch
        # Squelch level needs to be lower than NBFM or else choppy AM demod
        self.analog_pwr_squelch_cc = analog.pwr_squelch_cc(squelch_db,
                                                           1e-1, 0, False)

        # AGC with reference set for nomninal 0 dB volume
        # Paramaters tweaked to prevent impulse during squelching
        self.agc3_cc = analog.agc3_cc(1.0, 1E-4, self.agc_ref, 10, 1)
        self.agc3_cc.set_max_gain(65536)

        # AM demod with complex_to_mag()
        # Can't use analog.am_demod_cf() since it won't work with N>2 demods
        am_demod_cf = blocks.complex_to_mag(1)

        # 3.5 kHz cutoff for audio bandwidth
        low_pass_filter_taps_2 = grfilter.firdes.low_pass(1,\
                        samp_rate/(decims[1] * decims[0]**2),\
                        3.5E3, 500, window.WIN_HAMMING)

        # FIR filter decimating by 5 from 40-79.9 ksps to 8-15.98 ksps
        fir_filter_fff_0 = grfilter.fir_filter_fff(decims[0],
                                                   low_pass_filter_taps_2)

        # Polyphase resampler allows arbitary RF sample rates
        # Takes 8-15.98 ksps to a constant 8 ksps for audio
        pfb_resamp = audio_rate/float(samp_rate/(decims[1] * decims[0]**3))
        pfb_arb_resampler_fff = pfb.arb_resampler_fff(pfb_resamp, taps=None,
                                                      flt_size=32)

        # Connect the blocks for the demod
        self.connect(self, self.freq_xlating_fir_filter_ccc)
        self.connect(self.freq_xlating_fir_filter_ccc, fir_filter_ccc_0)
        self.connect(fir_filter_ccc_0, fir_filter_ccc_1)
        self.connect(fir_filter_ccc_1, self.analog_pwr_squelch_cc)
        self.connect(self.analog_pwr_squelch_cc, self.agc3_cc)
        self.connect(self.agc3_cc, am_demod_cf)
        self.connect(am_demod_cf, fir_filter_fff_0)
        self.connect(fir_filter_fff_0, pfb_arb_resampler_fff)
        self.connect(pfb_arb_resampler_fff, self)

        # Need to set this to a very low value of -200 since it is after demod
        # Only want it to gate when the previous squelch has gone to zero
        analog_pwr_squelch_ff = analog.pwr_squelch_ff(-200, 1e-1, 0, True)

        # Connect the blocks for recording
        self.connect(pfb_arb_resampler_fff, analog_pwr_squelch_ff)

        # File sink with single channel and 8 bits/sample
        if (self.record):
            self.blocks_wavfile_sink = blocks.wavfile_sink('/dev/null', 1,
                                                       audio_rate,
                                                       blocks.FORMAT_WAV,
                                                       blocks.FORMAT_PCM_16,
                                                       False)
            self.connect(analog_pwr_squelch_ff, self.blocks_wavfile_sink)
        else:
            null_sink1 = blocks.null_sink(gr.sizeof_float)
            self.connect(analog_pwr_squelch_ff, null_sink1)

    def set_volume(self, volume_db: int) -> None:
        """Sets the volume

        Args:
            volume_db (int): Volume in dB
        """
        agc_ref = self.agc_ref * 10**(volume_db/20.0)
        self.agc3_cc.set_reference(agc_ref)

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
                 classifier_params: dict, channel_log_params: loggers.ChannelLogParams):

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

        channel_logger = loggers.ChannelLogger.get_logger(channel_log_params)

        # -----------Flow for Demod--------------

        # Create N parallel demodulators as a list of objects
        # Default to NBFM demod
        self.demodulators = []
        for channel_idx in range(num_demod):
            if type_demod == 1:
                self.demodulators.append(TunerDemodAM(self.samp_rate,
                                                      audio_rate, record,
                                                      audio_bps,
                                                      min_recording,
                                                      classifier,
                                                      channel_logger,
                                                      channel_idx+1))
            else:
                self.demodulators.append(TunerDemodNBFM(self.samp_rate,
                                                        audio_rate, record,
                                                        audio_bps,
                                                        min_recording,
                                                        classifier,
                                                        channel_logger,
                                                        channel_idx+1))

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
        """Get the list of suppoted gain elements
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

def main():
    """Test the receiver

    Sets up the hardware
    Tunes a couple of demodulators
    Prints the max power spectrum
    """

    # Create receiver object
    ask_samp_rate = 2E6
    num_demod = 4
    type_demod = 0
    hw_args = "uhd"
    freq_correction = 0
    record = False
    play = True
    audio_bps = 8
    min_recording=1.0
    classifier_params={'V':False,'D':False,'S':False }
    receiver = Receiver(ask_samp_rate, num_demod, type_demod, hw_args,
                        freq_correction, record, play, audio_bps,
                        min_recording, classifier_params)

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
        demodulator.set_center_freq(channels[idx], center_freq)

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
    try:
        main()
    except KeyboardInterrupt:
        pass
