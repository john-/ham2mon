#!/usr/bin/env python
"""
Created on Fri Jul  3 13:38:36 2015

@author: madengr
"""

import glob
import logging
import os
from collections.abc import Callable

import numpy as np
from config import (
    GAIN_FIELDS,
    AudioConfig,
    GainConfig,
    HardwareConfig,
    ReceiverConfig,
)
from demodulators.AM import TunerDemodAM
from demodulators.NBFM import TunerDemodNBFM
from demodulators.WBFM import TunerDemodWBFM
from gnuradio import (
    audio,
    blocks,
    fft,
    gr,  # type: ignore
)
from gnuradio.fft import window  # type: ignore

from utilities import DEFAULT_AUDIO_RATE

logger = logging.getLogger(f"ham2mon.{__name__}")

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

    def __init__(self,
                 hardware_config: HardwareConfig,
                 receiver_config: ReceiverConfig,
                 audio_config: AudioConfig,
                 gain_config: GainConfig,
                 notify_scanner: Callable,
                 get_ctcss_info: Callable[[float], list[float]] | None = None,
                 source_type: str = "hardware", source_file: str | None = None,
                 wav_dir: str = "wav", center_freq: int = int(144E6)):

        # Call the initialization method from the parent class
        super().__init__("Receiver")

        self._source_type = source_type
        self._wav_dir = wav_dir

        # Make sure the 'wav' directory exists
        try:
            os.makedirs(os.path.join(self._wav_dir, 'tmp'), exist_ok=True)
        except OSError as error:  # will need to add something here for Win support
            logger.error(f"Could not create wav/tmp directory: {error}")
            raise

        # Clean up existing files without breaking makedirs or masking permission errors
        for f in glob.glob(os.path.join(self._wav_dir, 'tmp', '*.wav')):
            try:
                os.unlink(f)
            except OSError as error:
                logger.warning(f"Could not remove stale wav file: {f} ({error})")

        # Default values
        self.center_freq: int = center_freq
        self.samp_rate: int
        self.squelch_db = -60
        self.volume_db = 0
        self.gains: list[dict] = []
        audio_rate = DEFAULT_AUDIO_RATE

        # Extract configuration attributes from sub-domain config objects
        ask_samp_rate = int(hardware_config.sample_rate)
        num_demod = receiver_config.demodulators
        type_demod = receiver_config.mode
        hw_args = hardware_config.args
        freq_correction = hardware_config.freq_correction
        record = audio_config.record
        play = audio_config.play
        agc = gain_config.agc

        # Setup the USRP source, or use the USRP sim
        if self._source_type == "file":
            if not source_file:
                raise ValueError("source_file must be specified when source_type is 'file'")
            self.src, self.samp_rate, self.center_freq = self._init_file_source(
                source_file, ask_samp_rate, self.center_freq
            )
        else:
            self.src, self.samp_rate, self.center_freq = self._init_hardware_source(
                hw_args, ask_samp_rate, freq_correction, agc, self.center_freq
            )

        # NBFM channel is about 10 KHz wide
        # Want  about 3 FFT bins to span a channel
        # Use length FFT so 4 Msps / 1024 = 3906.25 Hz/bin
        # This also means 3906.25 vectors/second
        # Using below formula keeps FFT size a power of two
        # Also keeps bin size constant for power of two sampling rates
        # Use of 256 sets 3906.25 Hz/bin; increase to reduce bin size
        samp_ratio = self.samp_rate / 1E6
        # At exactly 1.0 Msps, np.ceil(np.log(1.0)/np.log(2)) = 0, giving 256 * 2^0 = 256.
        # For rates below 1.0 Msps, we floor the length to a minimum of 256.
        if samp_ratio <= 1.0:
            fft_length = 256
        else:
            fft_length = 256 * int(pow(2, np.ceil(np.log(samp_ratio)/np.log(2))))

        # -----------Flow for FFT--------------

        # Convert USRP steam to vector
        stream_to_vector = blocks.stream_to_vector(gr.sizeof_gr_complex*1,
                                                   fft_length)

        # Want about 1000 vector/sec
        amount = max(1, int(round(self.samp_rate/fft_length/1000)))
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

        self.get_ctcss_info = get_ctcss_info
        self.max_ctcss_tones = receiver_config.max_ctcss_tones

        # -----------Flow for Demod--------------

        # Create N parallel demodulators as a list of objects
        # Default to NBFM demod
        self.demodulators = []
        for idx in range(num_demod):
            if type_demod == 0:
                self.demodulators.append(TunerDemodNBFM(self.samp_rate,
                                                        audio_rate, record,
                                                        notify_scanner,
                                                        get_ctcss_info=self.get_ctcss_info,
                                                        max_ctcss_tones=self.max_ctcss_tones,
                                                        wav_dir=self._wav_dir))
            elif type_demod == 1:
                self.demodulators.append(TunerDemodAM(self.samp_rate,
                                                      audio_rate, record,
                                                      notify_scanner,
                                                      get_ctcss_info=self.get_ctcss_info,
                                                      max_ctcss_tones=self.max_ctcss_tones,
                                                      wav_dir=self._wav_dir))
            elif type_demod == 2:
                self.demodulators.append(TunerDemodWBFM(self.samp_rate,
                                                        audio_rate, record,
                                                        notify_scanner,
                                                        get_ctcss_info=self.get_ctcss_info,
                                                        max_ctcss_tones=self.max_ctcss_tones,
                                                        wav_dir=self._wav_dir))
            else:
                raise Exception(f'Invalid demodulator type: {type_demod}')


        if play:
            # Create an adder
            add_ff = blocks.add_ff(1)

            # Connect the demodulators between the source and adder
            for idx, demodulator in enumerate(self.demodulators):
                self.connect(self.src, demodulator, (add_ff, idx))

            # Audio sink
            try:
                audio_sink = audio.sink(audio_rate)
                # Connect the summed outputs to the audio sink
                self.connect(add_ff, audio_sink)
            except RuntimeError as error:
                logger.warning(f"Could not initialize audio sink (speaker output disabled): {error}")
                # Fall back to null sink to prevent application crash
                null_sink = blocks.null_sink(gr.sizeof_float)
                self.connect(add_ff, null_sink)
        else:
            # Just connect each demodulator to the receiver source
            for demodulator in self.demodulators:
                self.connect(self.src, demodulator)

    def _init_hardware_source(self, hw_args: str, ask_samp_rate: int, freq_correction: int, agc: bool, center_freq: int):
        import osmosdr  # type: ignore
        src = osmosdr.source(args="numchan=" + str(1) + " " + hw_args)
        src.set_sample_rate(ask_samp_rate)
        src.set_center_freq(center_freq)
        src.set_freq_corr(freq_correction)

        if agc:
            try:
                agc_is_set = src.set_gain_mode(agc, 0)
                assert agc == agc_is_set, f'set_gain_mode returned "{agc_is_set}"'
            except Exception as error:
                msg = f'Could not set AGC mode ({error})'
                logger.error(msg)
                raise Exception(msg)

        samp_rate = src.get_sample_rate()
        actual_center_freq = src.get_center_freq()
        src.set_bandwidth(0.8 * samp_rate)
        return src, samp_rate, actual_center_freq

    def _init_file_source(self, source_file: str, ask_samp_rate: int, center_freq: int):
        src = blocks.file_source(gr.sizeof_gr_complex, source_file, repeat=False)
        return src, ask_samp_rate, center_freq

    def set_center_freq(self, center_freq: int) -> None:
        """Sets RF center frequency of hardware

        Args:
            center_freq (int): Hardware RF center frequency in Hz
        """
        if self._source_type == "file":
            self.center_freq = center_freq
            return

        # Tune the hardware
        self.src.set_center_freq(center_freq)

        # Update center frequency with hardware center frequency
        # Do this to account for slight hardware offsets
        self.center_freq = self.src.get_center_freq()

    def start(self, max_noutput_items: int = 10000000) -> None:
        """Starts the top block and configures CTCSS selectors once topology is validated.

        Note: selectors are configured immediately after start(); demodulators begin
        at center_freq=0 so no meaningful audio flows before routing is applied.
        """
        super().start(max_noutput_items)
        for demod in self.demodulators:
            demod.configure_selectors()

    def get_gain_names(self) -> list[dict]:
        """Get the list of supported gain elements
        """
        if self._source_type == "file":
            return []
        return self.src.get_gain_names()

    def filter_and_set_gains(self, gain_config: GainConfig) -> list[dict]:
        """Validate explicit gains against hardware support, then apply all supported gains.

        When AGC is active, manual gain setting is skipped entirely — the hardware
        manages gain automatically and osmosdr ignores set_gain calls in AGC mode.

        Args:
            gain_config: GainConfig carrying user-supplied values (non-None) and defaults.
        Returns:
            list of {name, value} dicts for the hardware-confirmed active gains.
        Raises:
            ValueError: if the user explicitly configured a gain not supported by this hardware.
        """
        if self._source_type == "file":
            return []

        if gain_config.agc:
            logger.debug("AGC enabled — skipping manual gain setup")
            return []

        names = self.get_gain_names()

        # Strict validation: reject any gain the user explicitly named that this SDR cannot handle
        unsupported = [
            hw_name
            for field_name, hw_name in GAIN_FIELDS
            if gain_config.is_explicit(field_name) and hw_name not in names
        ]
        if unsupported:
            raise ValueError(
                f"Gain(s) {unsupported} are not supported by the connected SDR hardware. "
                f"Supported gains: {names}"
            )

        # Build the active gains list: only hardware-supported gains with effective values
        gains: list[dict] = [
            {"name": hw_name, "value": gain_config.get_value(field_name)}
            for field_name, hw_name in GAIN_FIELDS
            if hw_name in names
        ]
        return self.set_gains(gains)

    def set_gains(self, gains: list[dict]) -> list[dict]:
        """Set all the gains
        Args:
            gains (list of dictionary): Supported gains in dB
        """
        if self._source_type == "file":
            self.gains = gains
            return self.gains
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

    def get_demod_freq_map(self) -> dict[int, object]:
        """Returns a mapping of baseband center frequency to demodulator.

        Allows O(1) lookup of a demodulator by its tuned frequency, which is
        more efficient than the O(N) linear scan that would otherwise be needed
        when iterating over channels in the scanner sweep.

        Returns:
            dict[int, BaseTuner]: {center_freq_hz: demodulator} for all demodulators
        """
        return {d.center_freq: d for d in self.demodulators}

    def __del__(self):
        """Called when the object is destroyed."""
        # Make a best effort attempt to clean up our wavfile if it's empty
        try:
            if hasattr(self, '_wav_dir'):
                for f in glob.glob(os.path.join(self._wav_dir, 'tmp', '*.wav')):
                    os.unlink(f)
                os.rmdir(os.path.join(self._wav_dir, 'tmp'))
        except Exception:
            pass  # oh well, we're dying anyway
