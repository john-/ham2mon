#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:38:36 2015

@author: madengr
"""
import receiver as recvr
import estimate
import h2m_parser as prsr
import time
import numpy as np
import sys
# import yaml
import logging
from numpy.typing import NDArray
from channel_loggers import ChannelLogParams, ChannelMessage, ChannelLogger
from classification import ClassifierParams
from center_frequency_provider import FrequencyGroup, FrequencyProvider
from frequency_manager import FrequencyManager, FrequencyList, FrequencyConfiguration, ChannelFrequency, ChannelList
from utilities import baseband_to_frequency, frequency_to_baseband
#import asyncio
from dataclasses import dataclass, field

@dataclass(kw_only=True)
class ClassificationCount:
    V: int = field(default=0)
    D: int = field(default=0)
    S: int = field(default=0)

class Scanner(object):
    """Scanner that controls receiver

    Estimates channels from FFT power spectrum that are above threshold
    Rounds channels to nearest 5 kHz
    Removes channels that are locked out
    Tunes demodulators to new channels  # gets overwritten by receiver value
    Holds demodulators on channels between scan cycles

    Args:
        ask_samp_rate (int): Asking sample rate of hardware in sps (1E6 min)
        num_demod (int): Number of parallel demodulators
        type_demod (int): Type of demodulator (0=NBFM, 1=AM)
        hw_args (string): Argument string to pass to hardware
        freq_correction (int): Frequency correction in ppm
        record (bool): Record audio to file if True
        auto_priority (bool): Automatically set priority channels
        frequency_configuration (FrequencyConfiguration): File name and other config information
        channel_log_params (ChannelLogParams): Parameters for logging channel activity
        audio_bps (int): Audio bit depth in bps (bits/samples)
        frequency_params (FrequencyGroup): Parameters for frequency provider
        spacing (int): granularity of frequency quantization
        min_recording (float): Minimum length of a recording in seconds
        max_recording (float): Maximum length of a recording in seconds
        classifier_params (ClassifierParams): Parameters for channel classification
        auto_priority (bool): Automatically set priority channels
        agc (bool): Automatic gain control

    Attributes:
        center_freq (int): Hardware RF center frequency in Hz
        samp_rate (int): Hardware sample rate in sps (1E6 min)
        gains : Enumerated gain types and values
        squelch_db (int): Squelch in dB
        volume_dB (int): Volume in dB
        threshold_dB (int): Threshold for channel detection in dB
        spectrum (numpy.ndarray): FFT power spectrum data in linear, not dB
        frequencies (FrequencyList): List of frequencies including baseband values
        channel_spacing (float):  Spacing that channels will be rounded
        lockout_file_name (string): Name of file with channels to lockout
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments

    def __init__(self, ask_samp_rate: int=int(4E6), num_demod: int=4, type_demod: int=0,
                 hw_args: str="uhd", freq_correction: int=0, record: bool=True,
                 frequency_configuration: FrequencyConfiguration | None=None,
                 channel_log_params: ChannelLogParams=ChannelLogParams(type='none', target='', timeout=0),
                 play: bool=True,
                 audio_bps: int=8, channel_spacing: int=5000,
                 frequency_params: FrequencyGroup=FrequencyGroup(sample_rate=int(4E6)),
                 min_recording: float=0, max_recording: float=0,
                 classifier_params: ClassifierParams=None,
                 auto_priority: bool=False, agc: bool=False):

        # Default values
        self.squelch_db = -60
        self.volume_db = 0
        self.threshold_db = 10
        self.record = record
        self.play = play
        self.audio_bps = audio_bps
        self.samp_rate: int
        self.frequency_params = frequency_params
        self.spectrum: NDArray = np.empty(0)
        self.frequencies: FrequencyList = []    # needed for the UI
        self.channels: ChannelList = []
        self._channels: ChannelList = []
        self.channel_log_params = channel_log_params
        self.channel_spacing = channel_spacing
        self.frequency_file_name = frequency_configuration.file_name  # used by the gui
        self.log_timeout_last = int(time.time())
        self.log_mode = ""
        self.hang_time: float = 1.0
        self.max_recording = max_recording
        self.xmit_stats: dict[float, ClassificationCount] = {}
        self.auto_priority = auto_priority

        self.channel_logger = ChannelLogger.get_logger(channel_log_params)

        # Create receiver object
        self.receiver = recvr.Receiver(ask_samp_rate, num_demod, type_demod,
                                       hw_args, freq_correction, record, play,
                                       audio_bps, min_recording, classifier_params,
                                       self.got_channel_activity, agc)

        # Get the hardware sample rate
        self.samp_rate = self.receiver.samp_rate

        self.frequency_params.notify_scanner = self.center_freq_changed
        self.frequency_params.sample_rate = self.samp_rate  # update with hardware sample rate

        self.frequency_provider = FrequencyProvider(self.frequency_params)

        # Set the initial center frequency here to allow setting min/max
        self.center_freq = self.frequency_provider.center_freq
        self.receiver.set_center_freq(self.center_freq)
        # Get the hardware center frequency in Hz
        self.center_freq = self.receiver.center_freq
        # Frequency provider info is used by the interface
        self.step = self.frequency_provider.step
        self.steps = self.frequency_provider.steps

        self.frequency_manager = FrequencyManager(frequency_configuration, self.channel_spacing)
        # self.frequencies = self.frequency_manager.frequencies

        # Start the receiver and wait for samples to accumulate
        self.receiver.start()
        time.sleep(1)

    def center_freq_changed(self):
        '''
        Callback used to propagate provider value with self.  Also,
        notify the interface that the center frequency changed.
        '''
        self.set_center_freq(self.frequency_provider.center_freq)

        self.frequency_params.notify_interface()
    
    async def scan_cycle(self) -> None:
        """Execute one scan cycle

        Should be called no more than 10 Hz rate
        Estimates channels from FFT power spectrum that are above threshold
        Rounds channels to nearest 5 kHz
        Moves priority channels in front
        Tunes demodulators to new channels
        Holds demodulators on channels between scan cycles
        Add metadata to channels for GUI and further processing
        Log recent active channels
        """

        raw_channels = self._get_raw_channels()

        self._channels = self._add_metadata(raw_channels)

        await self._process_current_demodulators(self._channels)

        await self._assign_channels_to_demodulators(self._channels)

        self.channels = self._channels

    def _get_raw_channels(self) -> NDArray:
        # Grab the FFT data, set threshold, and estimate baseband channels
        self.spectrum = self.receiver.probe_signal_vf.level()
        threshold = 10**(self.threshold_db/10.0)
        channels = np.array(
            estimate.channel_estimate(self.spectrum, threshold))

        # Convert channels from bin indices to baseband frequency in Hz
        channels = (channels-len(self.spectrum)/2) *\
            self.samp_rate/len(self.spectrum)

        # Round channels to channel spacing
        # Note this affects tuning the demodulators
        # 5000 Hz is adequate for NBFM
        # Note that channel spacing is with respect to the center + baseband offset,
        # not just the offset itself
        real_channels = channels + self.center_freq
        real_channels = np.round(
            real_channels / self.channel_spacing) * self.channel_spacing
        channels = (real_channels - self.center_freq).astype(int)
        # Remove 0 as this also represents an unassigned demodulator
        # As a result, valid signals at the center point will be ignored
        channels = channels[channels != 0]

        return channels

    async def _process_current_demodulators(self, channels: ChannelList) -> None:

        the_now = time.time()
        for idx in range(len(self.receiver.demodulators)):
            demodulator = self.receiver.demodulators[idx]
            if demodulator.center_freq == 0:
                continue

            # Stop locked out demodulator (lockout was just added via UI)
            if self.frequency_manager.locked_out(demodulator.center_freq):
                await demodulator.set_center_freq(0, self.center_freq)
                continue

            # Stop the demodulator if not being scanned and outside the hang time
            if any(channel.hanging and channel.bb == demodulator.center_freq for channel in channels):
                if the_now - demodulator.last_heard > self.hang_time:
                    await demodulator.set_center_freq(0, self.center_freq)
            else:
                demodulator.set_last_heard(the_now)

            # Stop any long running modulators
            if self.max_recording > 0:
                if time.time() - demodulator.time_stamp >= self.max_recording:
                    # clear the demodulator to reset file
                    await demodulator.set_center_freq(0, self.center_freq)

    async def _assign_channels_to_demodulators(self, channels: ChannelList) -> None:

        # assign channels to available demodulators
        for channel in [channel for channel in channels if not channel.hanging]:
        #for channel in channels:
            # If channel not in demodulators
            if channel.bb not in self.receiver.get_demod_freqs() and not channel.locked:
                # Sequence through each demodulator
                for idx in range(len(self.receiver.demodulators)):
                    demodulator = self.receiver.demodulators[idx]
                    # If channel is higher priority than what is being demodulated
                    if self.frequency_manager.is_higher_priority(channel.bb, demodulator.center_freq):
                        # Assigning channel to empty demodulator
                        await demodulator.set_center_freq(
                            channel.bb, self.center_freq)
                        break
                    else:
                        pass
            else:
                pass

    def _add_metadata(self, active_channels: NDArray) -> ChannelList:

        all_channels = active_channels   # start out with the active channels

        # If a demodulator is not in channel list than it is waiting for hang time to end
        # There is no activity on it so it was not in the scan
        demod_freqs = self.receiver.get_demod_freqs()
        for demod_freq in demod_freqs:
            if demod_freq != 0 and demod_freq not in all_channels:
               all_channels = np.append(all_channels, demod_freq)

        sweep: ChannelList = []
        for channel in all_channels:
            frequency = baseband_to_frequency(channel, self.receiver.center_freq)
            priority = self.frequency_manager.is_priority(channel)
            idx = 0 if priority is not None else len(sweep)  # priority channels up front
            sweep.insert(idx, ChannelFrequency(bb=channel,
                                      rf=frequency,
                                      locked=self.frequency_manager.locked_out(channel),
                                      active=channel in demod_freqs and channel in active_channels,
                                      priority=priority,
                                      hanging=channel in demod_freqs and channel not in active_channels,
                                      label=self.frequency_manager.get_label(frequency)))

        return sweep

    async def add_lockout(self, idx: int) -> None:
        # need the same subset here as in cursesgui.ChannelWindow so idx gets the right channel
        subset = [c for c in self.channels if c.active or c.hanging]
        try:
            self.frequencies = await self.frequency_manager.add(subset[idx].rf, {'locked': True})
        except IndexError:
            # user selected a digit but no channels in interface
            return
    
    async def clear_lockout(self) -> None:
        """
        Clears lockout channels and rebuilds based on config.  Usually called
        by the user interface ('l' key).
        """
        self.frequencies = await self.frequency_manager.load()

    async def load_frequencies(self) -> None:
        self.frequencies = await self.frequency_manager.load()

    def set_center_freq(self, center_freq: int) -> None:
        """Sets RF center frequency of hardware, update lockout
        baseband frequencies, and notify interface that things have changed

        Args:
            center_freq (int): Hardware RF center frequency in Hz
        """
        # Tune the receiver then update with actual frequency
        # and on frequency provider info
        self.receiver.set_center_freq(center_freq)
        self.center_freq = self.receiver.center_freq
        self.step = self.frequency_provider.step
        self.steps = self.frequency_provider.steps

        # Recreate baseband lockout since frequency is changing
        self.frequency_manager.set_center(self.center_freq)

    def filter_and_set_gains(self, all_gains: list[dict]) -> list[dict]:
        """Set the supported gains and return them

        Args:
            all_gains (list of dictionary): Supported gains in dB
        """
        self.gains = self.receiver.filter_and_set_gains(all_gains)
        return self.gains

    def set_gains(self, gains: list[dict]) -> list[dict]:
        """Set all the gains

        Args:
            gains (list of dictionary): Supported gains in dB
        """
        self.gains = self.receiver.set_gains(gains)
        return self.gains

    def set_squelch(self, squelch_db: int) -> None:
        """Sets squelch of all demodulators

        Args:
            squelch_db (int): Squelch in dB
        """
        self.receiver.set_squelch(squelch_db)
        self.squelch_db = self.receiver.squelch_db

    def set_volume(self, volume_db: int) -> None:
        """Sets volume of all demodulators

        Args:
            volume_db (int): Volume in dB
        """
        self.receiver.set_volume(volume_db)
        self.volume_db = self.receiver.volume_db

    def set_threshold(self, threshold_db: int) -> None:
        """Sets threshold in dB for channel detection

        Args:
            threshold_db (int): Threshold in dB
        """
        self.threshold_db = threshold_db

    async def got_channel_activity(self, msg: ChannelMessage) -> None:
        '''
        This callback is to let the demodulators inform us about a
        transmission.
    
        1. Log the activity via the currently configured channel logger
        2. If the channel is interesting, let the frequency provider know
            - It will hold the current center frequency open a bit longer
        3. assess if the channel should be an auto priority channel
        '''

        if msg is None:
            return

        # embellish the message with frequency information
        msg.label = self.frequency_manager.get_label(msg.rf)
        msg.priority = self.frequency_manager.is_priority(msg.bb)   # TODO: is_priority only takes base band frequency

        await self.channel_logger.log(msg)  # off events or nothing to note

        if self.interesting(msg):
            await self.frequency_provider.interesting_activity()

        await self.priority_assess(msg.rf, msg.classification)

    def interesting(self, msg: ChannelMessage) -> bool:
        '''
        What is interesting?
        1.  If recording and a wav file was created
        2.  If not recording and channel was set to active
        '''
        if (self.record and msg.file is not None) or \
            (not self.record and msg.state == 'on'):
            return True
        else:
            return False

    def stop(self) -> None:
        """Stop the receiver
        """
        self.receiver.stop()
        self.receiver.wait()

    async def priority_assess(self, freq: float, classification: str) -> None:
        '''
        Track classification of transmisions and use the ratio of wanted/unwanted to
        set the priority.
        '''

        if not self.auto_priority:
            return

        if classification is None:  # ignore start of transission and thrown away short ones
            return

        if freq not in self.xmit_stats:
            self.xmit_stats[freq] = ClassificationCount()
            setattr(self.xmit_stats[freq], classification, 1)
        else:
            setattr(self.xmit_stats[freq], classification,
                    getattr(self.xmit_stats[freq], classification) + 1)

        bb_freq = frequency_to_baseband(float(freq), self.center_freq, self.channel_spacing)

        metrics: ClassificationCount = self.xmit_stats[freq]
        if metrics.V > metrics.D and metrics.V > metrics.S:  # Flag voice frequency as priority if not already set
            if self.frequency_manager.is_priority(bb_freq) is None:
                logging.debug(f'adding {freq=} to priority list')
                self.frequencies = await self.frequency_manager.add(freq, {'priority': 1})
        else: # If not voice, remove from priority list if it currently a priority
            if self.frequency_manager.is_priority(bb_freq) is not None:
                logging.debug(f'removing {freq=} from the priority list')
                self.frequencies = await self.frequency_manager.add(freq, {'priority': None})

    async def clean_up(self) -> None:
        # cleanup terminating all demodulators
        for demod in self.receiver.demodulators:
            await demod.set_center_freq(0, self.center_freq)


async def main() -> None:
    """Test the scanner

    Gets options from parser
    Sets up the scanner
    Assigns a channel to lockout
    Executes scan cycles
    Prints channels as they change
    """

    # Create parser object
    parser = prsr.CLParser()

    if not len(sys.argv) > 1:
        parser.print_help() #pylint: disable=maybe-no-member
        raise SystemExit(1)

    # Create scanner object
    ask_samp_rate = parser.ask_samp_rate
    num_demod = parser.num_demod
    type_demod = parser.type_demod
    hw_args = parser.hw_args
    freq_correction = parser.freq_correction
    record = parser.record
    frequency_configuration = parser.frequency_configuration
    channel_log_params = parser.channel_log_params
    play = parser.play
    audio_bps = parser.audio_bps
    channel_spacing = parser.channel_spacing
    frequency_params = parser.frequency_params
    min_recording = 0
    max_recording = 0
    classifier_params = parser.classifier_params
    scanner = Scanner(ask_samp_rate, num_demod, type_demod, hw_args,
                        freq_correction, record, frequency_configuration,
                        channel_log_params, play,
                        audio_bps, channel_spacing, frequency_params,
                        min_recording, max_recording,
                        classifier_params)

    # Set frequency, gain, squelch, and volume
    print("\n")
    print("Started %s at %.3f Msps" % (hw_args, scanner.samp_rate/1E6))
    scanner.filter_and_set_gains(parser.gains)
    for gain in scanner.gains:
        print("gain %s at %d dB" % (gain["name"], gain["value"]))
    scanner.set_squelch(parser.squelch_db)
    scanner.set_volume(parser.volume_db)
    print("%d demods of type %d at %d dB squelch and %d dB volume" % \
        (num_demod, type_demod, scanner.squelch_db, scanner.volume_db))

    # Create this empty list to allow printing to screen
    old_freqs: list[float] = []

    while 1:
        # No need to go faster than 10 Hz rate of GNU Radio probe
        await asyncio.sleep(0.1)

        # Execute a scan cycle
        await scanner.scan_cycle()

        # Print the tuned channels if they have changed

        channels =  scanner.channels

        freqs = [freq.rf for freq in channels if freq.active]

        freqs.sort()
        if freqs != old_freqs:
            print(freqs)
        old_freqs = freqs


if __name__ == '__main__':

    import asyncio

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
