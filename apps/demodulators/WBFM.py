"""
@author: madengr
"""

from gnuradio import gr  # type: ignore
from gnuradio import filter as grfilter # Don't redefine Python's filter()
from gnuradio.fft import window  # type: ignore
from gnuradio import analog
from gnuradio.filter import pfb  # type: ignore
from gnuradio import blocks
import ctcss_tones as ct
import logging
from typing import Callable

from demodulators.BaseTuner import BaseTuner
from classification import Classifier
from channel_loggers import ChannelLogger

class TunerDemodWBFM(BaseTuner):
    """Tuner, demodulator, and recorder chain for wide band FM demodulation

    Kept as it's own class so multiple can be instantiated in parallel
    Accepts complex baseband samples at 1 Msps minimum
    Frequency translating FIR filter tunes from -samp_rate/2 to +samp_rate/2
    The following sample rates assume 1 Msps input
    First two stages of decimation are 4 each for a total of 16
    Thus first two stages brings 1 Msps down to 40 ksps
    The third stage decimates by int(samp_rate/1E6)
    Thus output rate will vary from 40 ksps to 79.99 ksps
    The channel is filtered to 25.0 KHz bandwidth followed by squelch
    The squelch is non-blocking since samples will be added with other demods
    The quadrature demod is followed by a fourth stage of decimation by 4
    This brings the sample rate down to 8 ksps to 15.98 ksps
    The audio is low-pass filtered to 3.5 kHz bandwidth
    The polyphase resampler resamples by samp_rate/(decims[1] * decims[0]**3)
    Audio rate is configurable at 8 or 16 ksps
    This results in a constant 8/16 ksps, irrespective of RF sample rate
    The audio may then be CTCSS squelch blocked if configured, with a tone
    The audio may then be high-pass filtered to remove CTCSS tones if configured
    This 8/16 ksps audio stream may be added to other demod streams
    The audio is run through an additional blocking squelch at -200 dB
    This stops the sample flow so squelched audio is not recorded to file
    The wav file sink stores 8-bit samples (default/grainy quality but compact)
    Default demodulator center frequency is 0 Hz
    This is desired since hardware DC removal reduces sensitivity at 0 Hz
    WBFM demod of LO leakage will just be 0 amplitude

    Args:
        samp_rate (float): Input baseband sample rate in sps (1E6 minimum)
        audio_rate (float): Output audio sample rate in sps (8 kHz minimum)
        record (bool): Record audio to file if True
        audio_bps (int): Audio bit depth in bps (bits/samples)
        min_file_size (int): Minimum saved wav file size
        ctcss_filter (bool): Filter on set CTCSS tone if True
        ctcss_tone_block (bool): Prevent CTCSS tones in audio output if True

    Attributes:
        center_freq (float): Baseband center frequency in Hz
        record (bool): Record audio to file if True
        time_stamp (int): Time stamp of demodulator start for timing run length
        ctcss_filter (bool): Filter on set CTCSS tone if True
        ctcss_tone (float): CTCSS tone frequency for filter in Hz
        ctcss_index (int): CTCSS tone list index for selected tone
        ctcss_tone_block (bool): Prevent CTCSS tones in audio output if True
        ctcss_level (float): CTCSS tone level required to break squelch (open)
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, samp_rate: int, audio_rate: int, record: bool,
                 audio_bps: int, min_recording: float, classify: Classifier | None,
                 notify_scanner: Callable, ctcss_filter: bool=False, ctcss_tone_block: bool=False):

        gr.hier_block2.__init__(self, "TunerDemodWBFM",
                                gr.io_signature(1, 1, gr.sizeof_gr_complex),
                                gr.io_signature(1, 1, gr.sizeof_float))

        super().__init__(classify, notify_scanner)
        
        # Default values
        self.center_freq = 0
        self.time_stamp = 0
        squelch_db = -60
        self.quad_demod_gain = 0.050
        self.file_name = None
        self.record = record
        self.audio_bps = audio_bps
        self.min_recording = min_recording
        self.ctcss_filter = ctcss_filter
        self.ctcss_tone_block = ctcss_tone_block
        self.ctcss_level = 0.001 # little value in configuring this from testing so far

        # Decimation values for four stages of decimation
        decims = (4, int(samp_rate/1E6))

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
        # 25.0 kHz cutoff for WBFM channel bandwidth
        low_pass_filter_taps_1 = grfilter.firdes.low_pass(
            1, samp_rate/decims[0]**2, 25.0E3, 1E3, window.WIN_HAMMING)

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

        # Need to set this to a very low value of -200 since it is after demod
        # Only want it to gate when the previous squelch has gone to zero
        analog_pwr_squelch_ff = analog.pwr_squelch_ff(-200, 1e-1, 0, True)

        # Connect CTCSS related blocks
        if (self.ctcss_filter):
            # create CTCSS squelch block for float audio values at audio rate
            self.analog_ctcss_squelch_ff_0 = analog.ctcss_squelch_ff(audio_rate, self.ctcss_tone, self.ctcss_level, 1000, 0, True)

        if (self.ctcss_tone_block):
            # create CTCSS tone filtering high-pass filter for float audio
            self.high_pass_filter_0 = filter.fir_filter_fff(
            1,
            firdes.high_pass(
                1,
                16000,
                300,
                10,
                window.WIN_HAMMING,
                6.76))

#        if (self.ctcss_tone_detect):
#            # create tone detector via band-pass filter into pll freq det and moving avg
#            self.band_pass_filter_0 = filter.fir_filter_fcc(
#            1,
#            firdes.complex_band_pass(
#                1,
#                audio_rate,
#                55,
#                265,
#                100,
#                window.WIN_HAMMING,
#                6.76))
#            self.analog_pll_freqdet_cf_0 = analog.pll_freqdet_cf(100*2.0*pi, pi, -pi)
#            self.blocks_keep_one_in_n_0_0 = blocks.keep_one_in_n(gr.sizeof_float*1, 5)
#            self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(fft_length)
#            self.blocks_moving_average_xx_0 = blocks.moving_average_ff(3200, 1/3200, 30, 1)
#            self.blocks_multiply_const_vxx_0_0 = blocks.multiply_const_ff(audio_rate)
#            self.blocks_multiply_const_vxx_0 = blocks.multiply_const_ff(1/(2*pi))

        # Need to set this to a very low value of -200 since it is after demod
        # Only want it to gate when the previous squelch has gone to zero
        analog_pwr_squelch_ff = analog.pwr_squelch_ff(-200, 1e-1, 0, True)

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

        if (self.ctcss_filter and ~self.ctcss_tone_block):
            # Connect the blocks for CTCSS squelch filtering, keeping tone in audio
            self.connect(pfb_arb_resampler_fff, self.analog_ctcss_squelch_ff_0)
            self.connect(self.analog_ctcss_squelch_ff_0, analog_pwr_squelch_ff)
        elif (self.ctcss_filter and self.ctcss_tone_block):
            # Connect the blocks for CTCSS squelch filtering, removing tone in audio
            self.connect(pfb_arb_resampler_fff, self.analog_ctcss_squelch_ff_0)
            self.connect(self.analog_ctcss_squelch_ff_0, self.high_pass_filter_0)
            self.connect(self.high_pass_filter_0, analog_pwr_squelch_ff)
        elif (self.ctcss_tone_block):
            # Connect the blocks for removing CTCSS tones from audio
            self.connect(pfb_arb_resampler_fff, self.high_pass_filter_0)
            self.connect(self.high_pass_filter_0, analog_pwr_squelch_ff)
#        elif (self.ctcss_tone_detect):
#            # Connect tone detection PLL
#            self.connect(pfb_arb_resampler_fff,
        else:
            # Connect without CTCSS
            self.connect(pfb_arb_resampler_fff, analog_pwr_squelch_ff)

        # Connect the blocks for recording
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

    def set_volume(self, volume_db):
        """Sets the volume

        Args:
            volume_db (float): Volume in dB
        """
        gain = self.quad_demod_gain * 10**(volume_db/20.0)
        self.analog_quadrature_demod_cf.set_gain(gain)

    def set_ctcss_tone(self, ctcss_tone):
        """Sets the CTCSS tone frequency by selecting the index from tone list that matches the input value

        Args:
            ctcss_tone (float): CTCSS tone frequency in Hz
        """
        self.ctcss_index = ct.ctcss_tones.index(ctcss_tone)
        if (~self.ctcss_index):
            self.ctcss_index = 0
        self.ctcss_tone = ct.ctcss_tones[self.ctcss_index]
        return self.ctcss_tone
