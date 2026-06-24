"""
@author: madengr
"""

from gnuradio import gr  # type: ignore
from asyncio import Task
import time
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

    channel: int = 0  # incremented for each new demodulator

    def __init__(self, classify: Classifier | None, notify_scanner: Callable,
                 file_metadata: list[str] | None = None,
                 get_priority_info: Callable[[int], tuple[int | None, bool]] | None = None,
                 wav_dir: str = "wav") -> None:
        BaseTuner.channel += 1

        # Default values
        self.classify = classify
        self.notify_scanner = notify_scanner
        self.channel = BaseTuner.channel
        self.last_heard: float = 0.0
        self.file_name: str | None = None
        self.log_task: Task | None = None
        self.center_freq: int

        self.file_metadata: list[str] = file_metadata if file_metadata is not None else []
        self.get_priority_info = get_priority_info
        self.wav_dir = wav_dir


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
            # Move file from tmp directory if it is long enough
            # and classified appropriately
            results = self._persist_wavfile(rf_center_freq, avg_signal=avg_signal)   # also get channel_log information
        elif self.center_freq != 0:
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

    def _persist_wavfile(self, rf_center_freq: int, avg_signal: int | None = None) -> ChannelMessage | None:
        if not self.file_name:
            return None

        self.blocks_wavfile_sink.close()

        xmit_msg = ChannelMessage(state='off',
                                rf=baseband_to_frequency(self.center_freq, rf_center_freq),
                                bb=self.center_freq,
                                channel=self.channel,
                                signal_db=avg_signal)

        min_size = 44 + self.audio_bps * 1000 * self.min_recording
        if os.stat(self.file_name).st_size <= min_size:
            os.unlink(self.file_name)
            xmit_msg.detail = 'Discarded short recording'
            return xmit_msg

        # Classify if enabled — determines whether file is wanted and adds label
        classification: str | None = None
        if self.classify:
            is_wanted, classification = self.classify.is_wanted(self.file_name)
            xmit_msg.classification = classification
            if not is_wanted:
                os.unlink(self.file_name)
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
