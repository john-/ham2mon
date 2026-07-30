"""
@author: madengr
"""

import logging
import threading
import time
from asyncio import Task
from collections.abc import Callable

from frequency_manager import ChannelMessage
from gnuradio import (
    analog,
    blocks,
    gr,  # type: ignore
)
from gnuradio import filter as grfilter
from gnuradio.fft import window
from utilities import baseband_to_frequency, format_freq_mhz, format_timestamp

logger = logging.getLogger(f"ham2mon.{__name__}")

class BaseTuner(gr.hier_block2):
    """Some base methods that are the same between the known tuner types.

    See TunerDemodNBFM and TunerDemodAM for better documentation.
    """

    _CTCSS_GRACE_PERIOD_S: float = 0.7  # Seconds to allow CTCSS detector to stabilise before flagging a mismatch
    _CTCSS_DETECTOR_LEN: int = 4000  # Goertzel block size (number of samples) for CTCSS detection
    _channel_lock = threading.Lock()
    _channel_counter: int = 0  # incremented for each new demodulator

    def __init__(self, notify_scanner: Callable,
                 get_ctcss_info: Callable[[float], list[float]] | None = None,
                 wav_dir: str = "wav", audio_rate: int = 8000,
                 max_ctcss_tones: int = 0) -> None:
        with BaseTuner._channel_lock:
            BaseTuner._channel_counter += 1
            self.channel = BaseTuner._channel_counter
        self.notify_scanner = notify_scanner
        self.last_heard: float = 0.0
        self.time_stamp: float = 0.0
        self.file_name: str | None = None
        self.log_task: Task | None = None
        self.center_freq: int

        # NOTE: kept the name `get_ctcss_info` (rather than renaming to
        # get_ctcss_tones) to avoid cascading the rename through receiver.py,
        # the NBFM/AM/WBFM subclasses, and all their test fixtures. The
        # contract has changed though: this now returns a list of valid
        # tones (possibly empty), not a single tone.
        self.get_ctcss_info = get_ctcss_info
        self.wav_dir = wav_dir
        self.audio_rate = audio_rate
        self.ctcss_level = 0.0001
        self.max_ctcss_tones = max_ctcss_tones
        self.matched_ctcss_tone: float | None = None
        self._active_tones: list[float] = []

        # Pre-calculate CTCSS high-pass filter taps (strips the sub-audible tone from
        # recorded/played audio when CTCSS is active; taps swapped to all-pass when not).
        self.ctcss_hp_taps = grfilter.firdes.high_pass(
            1,
            self.audio_rate,
            300.0,
            10.0,
            window.WIN_HAMMING
        )

        # Signal path topology (parallel bypass and up to max_ctcss_tones CTCSS
        # chains, all summed by an adder):
        #
        #                   ┌──► bypass_gain ───────────────────────────────────┐
        #   self.ctcss_in ──┼──► ctcss_squelch[0] ──► HPF[0] ──► ctcss_gain[0] ──┤
        #                   ├──► ctcss_squelch[1] ──► HPF[1] ──► ctcss_gain[1] ──┼──► adder ──► hub ──► output
        #                   └──► ctcss_squelch[2] ──► HPF[2] ──► ctcss_gain[2] ──┘
        #
        # All paths remain connected and active at all times (samples always flow
        # through every chain), so HPF history buffers are never frozen and don't
        # leak previous transmissions when a chain (re)activates. Routing is
        # controlled purely by the gain blocks:
        #   bypass_gain=1.0 / all ctcss_gain=0.0        → bypass path (CTCSS disabled)
        #   bypass_gain=0.0 / ctcss_gain[i]=1.0 for i<N  → CTCSS chains 0..N-1 active
        # Each squelch is built with gate=False, so a chain whose tone isn't
        # currently present outputs zeros rather than gating -- since at most one
        # configured tone can be present in a real transmission, only that chain
        # contributes non-zero samples to the sum, with no need to pick a "winning"
        # tone in software. A channel with 0 or 1 configured tones behaves exactly
        # as it did before this was generalized to N tones.

        self.ctcss_in = blocks.multiply_const_ff(1.0)
        self._bypass_gain = blocks.multiply_const_ff(1.0)
        self._ctcss_adder = blocks.add_ff()
        self._ctcss_hub = blocks.multiply_const_ff(1.0)   # fan-out hub for ctcss_out

        # Wire bypass path: input → bypass_gain → adder port 0.
        self.connect(self.ctcss_in, self._bypass_gain, (self._ctcss_adder, 0))

        # Wire one CTCSS chain per tone slot: input → squelch[i] → hpf[i] → gain[i] → adder port i+1.
        self._ctcss_squelches: list = []
        self._ctcss_hpfs: list = []
        self._ctcss_gains: list = []
        for i in range(self.max_ctcss_tones):
            # gate=False — see topology note above.
            squelch = analog.ctcss_squelch_ff(self.audio_rate, 100.0, self.ctcss_level, self._CTCSS_DETECTOR_LEN, 0, False)
            hpf = grfilter.fir_filter_fff(1, [1.0])
            gain = blocks.multiply_const_ff(0.0)
            self.connect(self.ctcss_in, squelch, hpf, gain, (self._ctcss_adder, i + 1))
            self._ctcss_squelches.append(squelch)
            self._ctcss_hpfs.append(hpf)
            self._ctcss_gains.append(gain)

        # Wire adder output into hub so subclasses can fan out to multiple consumers.
        self.connect(self._ctcss_adder, self._ctcss_hub)

        self.ctcss_out = self._ctcss_hub      # subclasses: ctcss_out → sink/hier-output

        self._ctcss_enabled = False
        self._active_tone_count = 0   # how many of the N chains are configured/active for the current channel
        self.ctcss_matched = False
        self.ctcss_checked = False
        self._ctcss_start_time = 0.0
        self.discard_current = False
        self._is_started = False

    def set_last_heard(self, a_time: float) -> None:
        self.last_heard = a_time
        # Log active channel if at required interval
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
            # Close active WAV recording if recording
            results = self._close_recording(rf_center_freq, avg_signal=avg_signal)   # also get activity logging information
        else:
            self.discard_current = False
            if self.center_freq != 0:
                # not recording files and center_freq has changed
                results = ChannelMessage(state='off',
                                         rf=float(baseband_to_frequency(
                                            self.center_freq, rf_center_freq)),
                                         bb=int(self.center_freq),
                                         channel=int(self.channel),
                                         matched_ctcss=float(self.matched_ctcss_tone) if self.matched_ctcss_tone is not None else None)
            else:
                # center_freq is 0
                results = None

        await self.notify_scanner(results)  # off events or nothing to note

        # Set the frequency of the tuner
        self.center_freq = center_freq
        self.freq_xlating_fir_filter_ccc.set_center_freq(self.center_freq)

        # Dynamically configure CTCSS squelch tones and routing
        ctcss_tones: list[float] = []
        if self.center_freq != 0 and self.get_ctcss_info is not None:
            rf_freq = baseband_to_frequency(self.center_freq, rf_center_freq)
            ctcss_tones = self.get_ctcss_info(rf_freq) or []

        self._apply_ctcss_config(ctcss_tones)

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
                    logger.warning("Failed to set recording selector index: %s", e)


        if self.center_freq != 0:
            await self.notify_scanner(ChannelMessage(state='on',
                                                         rf=float(baseband_to_frequency(
                                                            self.center_freq, rf_center_freq)),
                                                         bb=int(self.center_freq),
                                                         channel=int(self.channel)))

    def set_file_name(self, rf_center_freq: int) -> None:
        rf_freq = baseband_to_frequency(self.center_freq, rf_center_freq) * 1e6
        self.tstamp_str = format_timestamp(self.time_stamp)
        self.freq_str = format_freq_mhz(rf_freq)
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

    def _close_recording(self, rf_center_freq: int, avg_signal: int | None = None) -> ChannelMessage | None:
        """Close the active WAV file and return a pending ChannelMessage.

        Returns None if no recording was in progress.
        Returns a ChannelMessage with state='off', wav_tmp_path, and started_at set.
        """
        if not self.file_name:
            return None

        if self._is_started and hasattr(self, '_rec_selector'):
            try:
                self._rec_selector.set_output_index(0)
            except IndexError as e:
                logger.warning("Failed to set recording selector index: %s", e)

        self.blocks_wavfile_sink.close()

        return ChannelMessage(
            state='off',
            rf=float(baseband_to_frequency(self.center_freq, rf_center_freq)),
            bb=int(self.center_freq),
            channel=int(self.channel),
            signal_db=int(avg_signal) if avg_signal is not None else None,
            matched_ctcss=float(self.matched_ctcss_tone) if self.matched_ctcss_tone is not None else None,
            wav_tmp_path=self.file_name,
            discard=self.discard_current,
            started_at=self.time_stamp,
        )


    def set_squelch(self, squelch_db: int) -> None:
        """Sets the threshold for both squelches

        Args:
            squelch_db (int): Squelch in dB
        """
        self.analog_pwr_squelch_cc.set_threshold(squelch_db)

    def _set_ctcss_tone(self, index: int, ctcss_tone: float) -> None:
        """Sets the CTCSS tone frequency on chain `index`'s squelch block.

        This is the internal per-slot equivalent of the old (single-tone)
        public `set_ctcss_tone`. Not exposed publicly since nothing outside
        this class calls it.
        """
        self._ctcss_squelches[index].set_frequency(ctcss_tone)

    def is_ctcss_mismatched(self) -> bool:
        """Returns True if there is an active signal (RF squelch open) but none of the
        channel's configured CTCSS tones currently match (all their squelches closed)."""
        if not self._ctcss_enabled:
            return False

        self.ctcss_checked = True

        # If we already matched in this transmission, we are done checking
        if self.ctcss_matched:
            return False

        # Check if demodulator RF squelch is open and at least one active CTCSS
        # chain's squelch is open (i.e. its specific tone is present)
        rf_open = self.analog_pwr_squelch_cc.unmuted()
        matched_idx = None
        if rf_open:
            for i in range(self._active_tone_count):
                if self._ctcss_squelches[i].unmuted():
                    matched_idx = i
                    break

        # Update matching status
        if matched_idx is not None:
            self.ctcss_matched = True
            if matched_idx < len(self._active_tones):
                self.matched_ctcss_tone = self._active_tones[matched_idx]

        # If we haven't matched yet, check if we've exceeded the grace period
        if not self.ctcss_matched:
            # Give the CTCSS chains 0.4 seconds to detect the tone and stabilize
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
            for i, gain in enumerate(self._ctcss_gains):
                gain.set_k(1.0 if i < self._active_tone_count else 0.0)
        else:
            self._bypass_gain.set_k(1.0)
            for gain in self._ctcss_gains:
                gain.set_k(0.0)

        if hasattr(self, '_rec_selector'):
            try:
                self._rec_selector.set_output_index(1 if self.file_name is not None else 0)
            except IndexError as e:
                logger.warning("Failed to configure recording selector: %s", e)

    def _apply_ctcss_config(self, ctcss_tones: list[float] | None) -> None:
        """Applies CTCSS configuration parameters and updates routing if started.

        Args:
            ctcss_tones: Valid CTCSS tones for the channel at the currently tuned
                frequency. Empty/None means CTCSS is not configured for this
                channel. At most self.max_ctcss_tones are honored; any beyond that
                are logged and ignored.
        """
        tones = list(ctcss_tones) if ctcss_tones else []

        if len(tones) > self.max_ctcss_tones:
            logger.warning(
                f"Channel {self.channel}: {len(tones)} CTCSS tones configured but "
                f"only {self.max_ctcss_tones} are supported; the rest will be ignored")
            tones = tones[:self.max_ctcss_tones]

        self._active_tone_count = len(tones)
        self._active_tones = tones
        self.matched_ctcss_tone = None

        if tones:
            self._ctcss_enabled = True
            self._ctcss_start_time = time.time()  # Start grace period timer
            self.ctcss_matched = False
            self.ctcss_checked = False
            for i, tone in enumerate(tones):
                self._set_ctcss_tone(i, tone)
                self._ctcss_squelches[i].set_level(self.ctcss_level)
                # Re-applying taps on every retune resets the HPF's history, so a
                # newly (re)activated chain doesn't leak the previous channel's audio.
                self._ctcss_hpfs[i].set_taps(self.ctcss_hp_taps)
            # Any slots beyond the configured count stay/become inactive
            for i in range(len(tones), self.max_ctcss_tones):
                self._ctcss_hpfs[i].set_taps([1.0])
            if self._is_started:
                self._bypass_gain.set_k(0.0)
                for i, gain in enumerate(self._ctcss_gains):
                    gain.set_k(1.0 if i < len(tones) else 0.0)
        else:
            self._ctcss_enabled = False
            # Swap taps to all-pass (1.0) on every chain when CTCSS is disabled so
            # each HPF acts as a simple unity gain/identity filter.
            for hpf in self._ctcss_hpfs:
                hpf.set_taps([1.0])
            if self._is_started:
                self._bypass_gain.set_k(1.0)
                for gain in self._ctcss_gains:
                    gain.set_k(0.0)
