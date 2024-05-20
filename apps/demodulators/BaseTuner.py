"""
@author: madengr
"""

from gnuradio import gr  # type: ignore
from asyncio import Task
import time
import numpy as np
import os
import logging

from h2m_types import ChannelMessage
from utilities import baseband_to_frequency
from classification import Classifier
from channel_loggers import ChannelLogger

class BaseTuner(gr.hier_block2):
    """Some base methods that are the same between the known tuner types.

    See TunerDemodNBFM and TunerDemodAM for better documentation.
    """

    channel: int = 0  # incremented for each new demodulator

    def __init__(self, classify: Classifier | None, channel_logger: ChannelLogger) -> None:
        BaseTuner.channel += 1

        # Default values
        self.classify = classify
        self.channel_logger = channel_logger
        self.channel = BaseTuner.channel
        self.last_heard: float = 0.0
        self.file_name: str | None = None
        self.log_task: Task | None = None
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
        (is_wanted, classification) = self.classify.is_wanted(self.file_name)
        xmit_msg.classification = classification
        if  is_wanted:
            name = self.file_name.replace('tmp/', '')
            name = name.replace('.wav', '_' + classification + '.wav')
            os.rename(self.file_name, name)
            
            # if using recent python3 have self.file_name be a Path
            # new_name = PurePath(self.file_name)
            # name = f'wav/{new_name.stem}_{is_wanted}{new_name.suffix}'
            xmit_msg.file = name
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
