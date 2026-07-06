"""
@author: madengr
"""

from gnuradio import gr  # type: ignore
from gnuradio import filter as grfilter
from gnuradio import analog
from gnuradio import blocks
from gnuradio.fft import window
from asyncio import Task

import time
import threading
import numpy as np
import os
import logging
from typing import Callable


from frequency_manager import ChannelMessage
from utilities import baseband_to_frequency
from classification import Classifier

class BaseTuner(gr.hier_block2):
    """Some base methods that are the same between the known tuner types.

    See TunerDemodNBFM and TunerDemodAM for better documentation.
    """

    _CTCSS_GRACE_PERIOD_S: float = 0.4  # Seconds to allow CTCSS detector to stabilise before flagging a mismatch
    _MAX_CTCSS_TONES: int = 3  # Cap on simultaneous CTCSS tones checked per channel. Not used yet
                                # (squelch still checks a single tone) -- reserved for the
                                # multi-tone detector work in frequency_manager.get_ctcss_tones().
    _channel_lock = threading.Lock()
    _channel_counter: int = 0  # incremented for each new demodulator

    def __init__(self, classify: Classifier | None, notify_scanner: Callable,
                 file_metadata: list[str] | None = None,
                 get_priority_info: Callable[[int], tuple[int | None, bool]] | None = None,
                 get_ctcss_info: Callable[[int], float | None] | None = None,
                 wav_dir: str = "wav", audio_rate: int = 8000) -> None:
        with BaseTuner._channel_lock:
            BaseTuner._channel_counter += 1
            self.channel = BaseTuner._channel_counter
        self.classify = classify
        self.notify_scanner = notify_scanner
        self.last_heard: float = 0.0
        self.time_stamp: float = 0.0
        self.file_name: str | None = None
        self.log_task: Task | None = None
        self.center_freq: int

        self.file_metadata: list[str] = file_metadata if file_metadata is not None else []
        self.get_priority_info = get_priority_info
        self.get_ctcss_info = get_ctcss_info
        self.wav_dir = wav_dir
        self.audio_rate = audio_rate
        self.ctcss_level = 0.0001

        # CTCSS squelch — gate=False so it outputs zeros instead of gating when
        # tone is absent. This keeps samples flowing through the HPF and prevents
        # history buffer freezing, while still muting the audio.
        self.analog_ctcss_squelch_ff = analog.ctcss_squelch_ff(self.audio_rate, 100.0, self.ctcss_level, 1000, 0, False)

        # Pre-calculate CTCSS high-pass filter taps (strips the sub-audible tone from
        # recorded audio when CTCSS is active; taps swapped to all-pass when not).
        self.ctcss_hp_taps = grfilter.firdes.high_pass(
            1,
            self.audio_rate,
            300.0,
            10.0,
            window.WIN_HAMMING
        )
        self.high_pass_filter_ctcss = grfilter.fir_filter_fff(1, [1.0])

        # Signal path topology (parallel bypass and CTCSS paths with adder):
        #
        #                   ┌──► bypass_gain ─────────────────────────┐
        #   self.ctcss_in ──┤                                         ├──► adder ──► hub ──► output
        #                   └──► ctcss_squelch ──► HPF ──► ctcss_gain ┘
        #
        # Both paths remain connected and active. The routing is controlled by
        # gain parameters on the bypass_gain and ctcss_gain blocks:
        #   bypass_gain=1.0 / ctcss_gain=0.0 → bypass path (CTCSS disabled)
        #   bypass_gain=0.0 / ctcss_gain=1.0 → CTCSS chain (CTCSS enabled)
        # Because samples continuously flow through Path 1 even when bypass mode
        # is selected, the HPF history buffers are never frozen and do not leak
        # previous transmissions when switching back to CTCSS.

        self.ctcss_in = blocks.multiply_const_ff(1.0)
        self._bypass_gain = blocks.multiply_const_ff(1.0)
        self._ctcss_gain = blocks.multiply_const_ff(0.0)
        self._ctcss_adder = blocks.add_ff()
        self._ctcss_hub = blocks.multiply_const_ff(1.0)   # fan-out hub for ctcss_out

        # Wire bypass path: input → bypass_gain → adder port 0.
        self.connect(self.ctcss_in, self._bypass_gain, (self._ctcss_adder, 0))

        # Wire CTCSS chain: input → ctcss_squelch → hpf → ctcss_gain → adder port 1.
        self.connect(self.ctcss_in,
                     self.analog_ctcss_squelch_ff,
                     self.high_pass_filter_ctcss,
                     self._ctcss_gain,
                     (self._ctcss_adder, 1))

        # Wire adder output into hub so subclasses can fan out to multiple consumers.
        self.connect(self._ctcss_adder, self._ctcss_hub)

        self.ctcss_out = self._ctcss_hub      # subclasses: ctcss_out → sink/hier-output

        self._ctcss_enabled = False
        self.ctcss_matched = False
        self.ctcss_checked = False
        self._ctcss_start_time = 0.0
        self.discard_current = False
        self._is_started = False

    def set_last_heard(self, a_time: float) -> None:
        self.last_heard = a_time
        # channel_log active channel if at required interval
        # alternately use a timer or something that is created on demod start

    async def set_center_freq(self, center_freq: int, rf_center_freq: int, avg_signal: int | None = None) -> None:
        """Sets baseband center frequency and file name

        Sets baseband center frequency of frequency translating FIR filter
        Also sets file name of wave file sink
        If tuner is tuned to zero Hz then set to file name to None
        Otherwise set file name to tuned RF frequency in MHz

        Args:
            center_freq (int): Baseband center frequency in Hz
            rf_center_freq (int): RF center in Hz (for file name)
            avg_signal (int, optional): Calculated average signal strength in dB
        """
        # address completed transmissions
        results: ChannelMessage | None
        if self.record:
            if self._ctcss_enabled and self.ctcss_checked and not self.ctcss_matched:
                # Fallback safety belt: if CTCSS was enabled and checked, but never matched before we tuned away,
                # discard the recording. The primary check is performed in scanner.py.
                self.discard_current = True
            # Move file from tmp directory if it is long enough
            # and classified appropriately
            results = self._persist_wavfile(rf_center_freq, avg_signal=avg_signal)   # also get channel_log information
        else:
            self.discard_current = False
            if self.center_freq != 0:
                # not recording files and center_freq has changed
                results = ChannelMessage(state='off',
                                         rf=baseband_to_frequency(
                                            self.center_freq, rf_center_freq),
                                         bb=self.center_freq,
                                         channel=self.channel)
            else:
                # center_freq is 0
                results = None

        await self.notify_scanner(results)  # off events or nothing to note

        # Set the frequency of the tuner
        self.center_freq = center_freq
        self.freq_xlating_fir_filter_ccc.set_center_freq(self.center_freq)

        # Dynamically configure CTCSS squelch tone and routing
        ctcss_tone = None
        if self.center_freq != 0 and self.get_ctcss_info is not None:
            rf_freq = baseband_to_frequency(self.center_freq, rf_center_freq)
            ctcss_tone = self.get_ctcss_info(rf_freq)

        self._apply_ctcss_config(ctcss_tone)

        # Set the file name if recording
        if self.center_freq == 0 or not self.record:
            # If tuner at zero Hz, or record false, then file name to None
            self.file_name = None
        else:
            self.time_stamp = time.time()  # used for file naming and checking max_recording length
            self.set_file_name(rf_center_freq)

        if (self.file_name is not None and self.record):
            self.blocks_wavfile_sink.open(self.file_name)
            if self._is_started and hasattr(self, '_rec_selector'):
                try:
                    self._rec_selector.set_output_index(1)
                except IndexError as e:
                    logging.warning("Failed to set recording selector index: %s", e)


        if self.center_freq != 0:
            await self.notify_scanner(ChannelMessage(state='on',
                                                         rf=baseband_to_frequency(
                                                            self.center_freq, rf_center_freq),
                                                         bb=self.center_freq,
                                                         channel=self.channel))

    def set_file_name(self, rf_center_freq: int) -> None:
        self.tstamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + "{:.3f}".format(self.time_stamp % 1)[1:]
        file_freq = (rf_center_freq + self.center_freq) / 1E6
        self.freq_str = f"{np.round(file_freq, 4):.4f}"
        self.file_name = f'{self.wav_dir}/tmp/{self.freq_str}_{self.tstamp_str}.wav'

    def connect_wav_sink(self, src_block) -> None:
        """Connects the given source block to the recording path, using a selector
        to route audio to a null sink when idle to keep the upstream pipeline
        and HPF buffers running and flushed.
        """
        if self.record:
            self._rec_selector = blocks.selector(gr.sizeof_float, 0, 0)
            self._rec_null_sink = blocks.null_sink(gr.sizeof_float)
            self.connect(src_block, self._rec_selector)
            self.connect((self._rec_selector, 0), self._rec_null_sink)
            self.connect((self._rec_selector, 1), self.blocks_wavfile_sink)
        else:
            null_sink = blocks.null_sink(gr.sizeof_float)
            self.connect(src_block, null_sink)

    def _persist_wavfile(self, rf_center_freq: int, avg_signal: int | None = None) -> ChannelMessage | None:
        if not self.file_name:
            return None

        if self._is_started and hasattr(self, '_rec_selector'):
            try:
                self._rec_selector.set_output_index(0)
            except IndexError as e:
                logging.warning("Failed to set recording selector index: %s", e)

        self.blocks_wavfile_sink.close()

        xmit_msg = ChannelMessage(state='off',
                                rf=baseband_to_frequency(self.center_freq, rf_center_freq),
                                bb=self.center_freq,
                                channel=self.channel,
                                signal_db=avg_signal)

        # Discard the file if flagged as mismatched CTCSS
        if self.discard_current:
            self.discard_current = False
            if self.file_name and os.path.exists(self.file_name):
                try:
                    os.unlink(self.file_name)
                except OSError as e:
                    logging.warning("Failed to delete mismatched CTCSS file %s: %s", self.file_name, e)
            xmit_msg.detail = 'Discarded mismatched CTCSS'
            return xmit_msg

        min_size = 44 + self.audio_bps * 1000 * self.min_recording
        if os.stat(self.file_name).st_size <= min_size:
            try:
                os.unlink(self.file_name)
            except OSError as e:
                logging.warning("Failed to delete short recording file %s: %s", self.file_name, e)
            xmit_msg.detail = 'Discarded short recording'
            return xmit_msg

        # Classify if enabled — determines whether file is wanted and adds label
        classification: str | None = None
        if self.classify:
            is_wanted, classification = self.classify.is_wanted(self.file_name)
            xmit_msg.classification = classification
            if not is_wanted:
                try:
                    os.unlink(self.file_name)
                except OSError as e:
                    logging.warning("Failed to delete unwanted classification file %s: %s", self.file_name, e)
                xmit_msg.detail = 'Discarded unwanted classification'
                return xmit_msg

        # Build final filename from stored components
        name_parts: list[str] = [self.freq_str]
        if classification is not None:
            name_parts.append(classification)
        if 'priority' in self.file_metadata and self.get_priority_info:
            priority, is_auto = self.get_priority_info(self.center_freq)
            if priority is not None:
                name_parts.append("PA" if is_auto else f"P{priority}")
        if 'strength' in self.file_metadata and avg_signal is not None:
            name_parts.append(f"{avg_signal}dB")
        name_parts.append(self.tstamp_str)

        new_name = f'{self.wav_dir}/{"_".join(name_parts)}.wav'
        os.rename(self.file_name, new_name)
        xmit_msg.file = new_name
        return xmit_msg

    def set_squelch(self, squelch_db: int) -> None:
        """Sets the threshold for both squelches

        Args:
            squelch_db (int): Squelch in dB
        """
        self.analog_pwr_squelch_cc.set_threshold(squelch_db)

    def set_ctcss_tone(self, ctcss_tone: float) -> None:
        """Sets the CTCSS tone frequency on the CTCSS squelch block.

        This is a public API for setting CTCSS squelch parameters.
        """
        self.ctcss_tone = ctcss_tone
        self.analog_ctcss_squelch_ff.set_frequency(ctcss_tone)

    def is_ctcss_mismatched(self) -> bool:
        """Returns True if there is an active signal (RF squelch open) but CTCSS tone does not match (CTCSS squelch closed)."""
        if not self._ctcss_enabled:
            return False

        self.ctcss_checked = True

        # If we already matched in this transmission, we are done checking
        if self.ctcss_matched:
            return False

        # Check if demodulator RF squelch is open and CTCSS squelch is open
        rf_open = self.analog_pwr_squelch_cc.unmuted()
        ctcss_open = self.analog_ctcss_squelch_ff.unmuted()

        # Update matching status
        if rf_open and ctcss_open:
            self.ctcss_matched = True

        # If we haven't matched yet, check if we've exceeded the grace period
        if not self.ctcss_matched:
            # Give the CTCSS block 0.4 seconds to detect the tone and stabilize
            if time.time() - self._ctcss_start_time < self._CTCSS_GRACE_PERIOD_S:
                return False
            # If 0.4s passed and still not matched while RF is open, it's a mismatch
            if rf_open:
                self.discard_current = True
                return True

        return False

    def configure_selectors(self) -> None:
        """Applies path routing based on self._ctcss_enabled.
        Must be called after the flowgraph starts.
        """
        self._is_started = True
        if self._ctcss_enabled:
            self._bypass_gain.set_k(0.0)
            self._ctcss_gain.set_k(1.0)
        else:
            self._bypass_gain.set_k(1.0)
            self._ctcss_gain.set_k(0.0)

        if hasattr(self, '_rec_selector'):
            try:
                self._rec_selector.set_output_index(1 if self.file_name is not None else 0)
            except IndexError as e:
                logging.warning("Failed to configure recording selector: %s", e)

    def _apply_ctcss_config(self, ctcss_tone: float | None) -> None:
        """Applies CTCSS configuration parameters and updates routing if started."""
        if ctcss_tone is not None:
            self._ctcss_enabled = True
            self._ctcss_start_time = time.time()  # Start grace period timer
            self.ctcss_matched = False
            self.ctcss_checked = False
            self.set_ctcss_tone(ctcss_tone)
            self.analog_ctcss_squelch_ff.set_level(self.ctcss_level)
            self.high_pass_filter_ctcss.set_taps(self.ctcss_hp_taps)
            if self._is_started:
                self._bypass_gain.set_k(0.0)
                self._ctcss_gain.set_k(1.0)
        else:
            self._ctcss_enabled = False
            # Swap taps to all-pass (1.0) when CTCSS is disabled so the HPF block
            # acts as a simple unity gain/identity filter.
            self.high_pass_filter_ctcss.set_taps([1.0])
            if self._is_started:
                self._bypass_gain.set_k(1.0)
                self._ctcss_gain.set_k(0.0)
