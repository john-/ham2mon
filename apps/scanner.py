#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:38:36 2015

@author: madengr
"""
import receiver as recvr
import estimate
import h2m_parser as prsr
from h2m_types import Channel
import time
import numpy as np
import sys
import yaml
import logging
from numpy.typing import NDArray
from channel_loggers import ChannelLogParams, ChannelMessage
from classification import ClassifierParams
from frequency_provider import FrequencyGroup, FrequencyProvider
from lockout_manager import LockoutManager, LockoutListRadioFreq
from utilities import baseband_to_frequency, frequency_to_baseband
#import asyncio
import typing
from pathlib import Path
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
        lockout_file_name (Path): Name of file with channels to lockout
        priority_file_name (string): Name of file with channels for priority
        auto_priority (bool): Automatically set priority channels
        audio_bps (int): Audio bit depth in bps (bits/samples)
        center_freq (int): initial center frequency for receiver (Hz)
        spacing (int): granularity of frequency quantization
        min_recording (float): Minimum length of a recording in seconds
        max_recording (float): Maximum length of a recording in seconds


    Attributes:
        center_freq (int): Hardware RF center frequency in Hz
        samp_rate (int): Hardware sample rate in sps (1E6 min)
        gains : Enumerated gain types and values
        squelch_db (int): Squelch in dB
        volume_dB (int): Volume in dB
        threshold_dB (int): Threshold for channel detection in dB
        spectrum (numpy.ndarray): FFT power spectrum data in linear, not dB
        lockout_channels (LockoutListRadioFreq): List of baseband lockout channels in MHz
        priority_channels list[int]: List of baseband priority channels in Hz
        channel_spacing (float):  Spacing that channels will be rounded
        lockout_file_name (string): Name of file with channels to lockout
        priority_file_name (string): Name of file with channels for priority
        auto_priority (bool): Automatically set priority channels
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments

    def __init__(self, ask_samp_rate: int=int(4E6), num_demod: int=4, type_demod: int=0,
                 hw_args: str="uhd", freq_correction: int=0, record: bool=True,
                 lockout_file_name: Path=Path(""), priority_file_name: str="",
                 channel_log_params: ChannelLogParams=ChannelLogParams(type='none', target='', timeout=0),
                 play: bool=True,
                 audio_bps: int=8, channel_spacing: int=5000,
                 frequency_params: FrequencyGroup=FrequencyGroup(sample_rate=int(4E6)),
                 min_recording: float=0, max_recording: float=0,
                 classifier_params: ClassifierParams=None, # dict[str, bool]={'V':False,'D':False,'S':False },
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
        self.lockout_channels: LockoutListRadioFreq = []
        self.priority_channels: list[int] = []
        self.channels: list[Channel] = []
        self.channel_log_params = channel_log_params
        self.channel_spacing = channel_spacing
        self.lockout_file_name = lockout_file_name  # used by lockout manager and the gui
        self.priority_file_name = priority_file_name
        self.log_timeout_last = int(time.time())
        self.log_mode = ""
        self.hang_time: float = 1.0
        self.max_recording = max_recording
        self.xmit_stats: dict[float, ClassificationCount] = {}
        self.auto_priority = auto_priority
        self.auto_priority_frequencies: list[float] = []

        channel_log_params.notify_scanner = self.got_provider_activity

        # Create receiver object
        self.receiver = recvr.Receiver(ask_samp_rate, num_demod, type_demod,
                                       hw_args, freq_correction, record, play,
                                       audio_bps, min_recording, classifier_params,
                                       channel_log_params, agc)

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

        self.lockout_manager = LockoutManager(lockout_file_name, self.center_freq, self.channel_spacing)
        self.lockout_channels = self.lockout_manager.lockouts

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

        channels = self._get_raw_channels()

        await self._process_current_demodulators(channels)

        await self._assign_channels_to_demodulators(channels)

        self._add_metadata(channels)

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

    async def _process_current_demodulators(self, channels: NDArray) -> None:

        the_now = time.time()
        for idx in range(len(self.receiver.demodulators)):
            demodulator = self.receiver.demodulators[idx]
            if demodulator.center_freq == 0:
                continue

            # Stop locked out demodulator
            if self.lockout_manager.locked_out(demodulator.center_freq):
                await demodulator.set_center_freq(0, self.center_freq)
                continue

            # Stop the demodulator if not being scanned and outside the hang time
            if demodulator.center_freq not in channels:
                if the_now - demodulator.last_heard > self.hang_time:
                    await demodulator.set_center_freq(0, self.center_freq)
            else:
                demodulator.set_last_heard(the_now)

            # Stop any long running modulators
            if self.max_recording > 0:
                if time.time() - demodulator.time_stamp >= self.max_recording:
                    # clear the demodulator to reset file
                    await demodulator.set_center_freq(0, self.center_freq)

    async def _assign_channels_to_demodulators(self, channels: NDArray) -> None:

        # assign channels to available demodulators
        for channel in channels:
            # If channel not in demodulators
            if channel not in self.receiver.get_demod_freqs() and not self.lockout_manager.locked_out(channel):
                # Sequence through each demodulator
                for idx in range(len(self.receiver.demodulators)):
                    demodulator = self.receiver.demodulators[idx]
                    # If channel is higher priority than what is being demodulated
                    if self.is_higher_priority(channel, demodulator.center_freq):
                        # Assigning channel to empty demodulator
                        await demodulator.set_center_freq(
                            channel, self.center_freq)
                        break
                    else:
                        pass
            else:
                pass

    def _add_metadata(self, active_channels: NDArray) -> None:

        demod_freqs = self.receiver.get_demod_freqs()

        # If a demodulator is not in channel list than it is waiting for hang time to end
        # There is no activity on it so it was not in the scan
        all_channels = active_channels   # start out with the active channels

        for demod_freq in demod_freqs:
            if demod_freq != 0 and demod_freq not in all_channels:
               all_channels = np.append(all_channels, demod_freq)

        sweep: list[Channel] = []
        for channel in all_channels:
            frequency = baseband_to_frequency(channel, self.receiver.center_freq)
            priority: bool = channel in self.priority_channels
            idx = 0 if priority else len(sweep)  # priority channels up front
            sweep.insert(idx, Channel(baseband=channel,
                                      frequency=frequency,
                                      locked=self.lockout_manager.locked_out(channel),
                                      active=channel in demod_freqs and channel in active_channels,
                                      priority=priority,
                                      hanging=channel in demod_freqs and channel not in active_channels))

        self.channels = sweep

    def is_higher_priority(self, channel: int, demod_freq: int) -> bool:

        if demod_freq == 0:
            return True

        try:
            channel_priority = self.priority_channels.index(channel)
        except ValueError:
            return False  # channel not in priority list so low priority
    
        try:
            demod_priority = self.priority_channels.index(demod_freq)
        except ValueError:
            return True   # there is a channel priority but no demod priority, therefore channel is higher priority

        if channel_priority < demod_priority:  # channel is higher priority than current demod frequency
            return True
        else:
            return False

    def add_lockout(self, idx: int) -> None:
        # need the same subset here as in cursesgui.ChannelWindow so idx gets the right channel
        subset = [c for c in self.channels if c.active or c.hanging]
        try:
            self.lockout_channels = self.lockout_manager.add(subset[idx].frequency)
        except IndexError:
            # user selected a digit but no channels in interface
            return
    
    def clear_lockout(self) -> None:
        """
        Clears lockout channels and rebuilds based on config.  Usually called
        by the user interface ('l' key).
        """
        self.lockout_channels = self.lockout_manager.load()

    def update_priority(self) -> None:
        """Updates priority channels
        """
        # Clear the priority channels
        self.priority_channels = []

        # Process priority file if it was provided
        if self.priority_file_name != "":
            # Open file, split to list, remove empty strings
            with open(self.priority_file_name) as priority_file:
                lines = list(filter(str.rstrip, priority_file))
            # Convert to baseband frequencies, round, and append if within BW
            for freq in lines:
                bb_freq = frequency_to_baseband(float(freq)/1E6, self.center_freq, self.channel_spacing)
                # bb_freq = float(freq) - self.center_freq
                # bb_freq = round(bb_freq/self.channel_spacing)*\
                #                        self.channel_spacing
                if abs(bb_freq) <= self.samp_rate/2.0:
                    self.priority_channels.append(bb_freq)
                else:
                    pass
        else:
            pass

        # Bring up channels with more voice than data/skip
        self.add_auto_priority_frequencies()

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

        # Update the priority since frequency is changing
        self.update_priority()

        # Recreate baseband lockout since frequency is changing
        self.lockout_manager.update(self.center_freq)
        #self.generate_baseband_lockout_channels()

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

    async def got_provider_activity(self, msg: ChannelMessage) -> None:
        '''
        The channel logger let us know about a possibly interesting transmission
    
        If so let the frequency provider know.  It will hold the channel
        open a bit longer.  Also, assess if the channel should be a priority channel.
        '''

        if self.interesting(msg):
            await self.frequency_provider.interesting_activity()

        await self.priority_assess(msg.frequency, msg.classification)

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
        Track classification of transmisions and use
        the ratio of wanted/unwanted to set the priority.
        '''

        if not self.auto_priority:
            return

        # TODO:  If the user has already defined a priority channel then this
        #        function will duplicate the priority.  Need to prevent this.
        #        When priority file format is changed maybe auto channels can be
        #        metadata added to the data structure/file.

        if classification is None:  # ignore start of transission and thrown away short ones
            return

        if freq not in self.xmit_stats:
            self.xmit_stats[freq] = ClassificationCount()
            setattr(self.xmit_stats[freq], classification, 1)
        else:
            setattr(self.xmit_stats[freq], classification,
                    getattr(self.xmit_stats[freq], classification) + 1)

        metrics: ClassificationCount = self.xmit_stats[freq]
        if metrics.V > metrics.D and metrics.V > metrics.S:
            if freq not in self.auto_priority_frequencies:
                logging.debug(f'adding {freq=} to priority list')
                self.auto_priority_frequencies.append(freq)
                self.update_priority()
        else:
            if freq in self.auto_priority_frequencies:
                logging.debug(f'removing {freq=} from the priority list')
                self.auto_priority_frequencies.remove(freq)
                self.update_priority()

    def add_auto_priority_frequencies(self) -> None:
        '''
        Add any frequencies that have a high ratio of wanted/unwanted
        to the priority list.
        '''

        for freq in self.auto_priority_frequencies:
            bb_freq = frequency_to_baseband(float(freq), self.center_freq, self.channel_spacing)
            if abs(bb_freq) <= self.samp_rate/2.0:
                self.priority_channels.append(bb_freq)
            else:
                pass


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
    lockout_file_name = parser.lockout_file_name
    priority_file_name = parser.priority_file_name
    channel_log_params = parser.channel_log_params
    play = parser.play
    audio_bps = parser.audio_bps
    channel_spacing = parser.channel_spacing
    frequency_params = parser.frequency_params
    min_recording = 0
    max_recording = 0
    classifier_params = parser.classifier_params
    scanner = Scanner(ask_samp_rate, num_demod, type_demod, hw_args,
                        freq_correction, record, lockout_file_name,
                        priority_file_name, channel_log_params, play,
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

        freqs = [freq.frequency for freq in channels if freq.active]

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
