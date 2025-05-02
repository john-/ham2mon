
from dataclasses import dataclass, field
from typing import Optional, TypeAlias  # TypeAlias needed for python < 3.12
from pathlib import Path
import yaml
import logging
from utilities import frequency_to_baseband

@dataclass(kw_only=True)
class BasebandFreqRange:
    lo: int
    hi: int

    def __post_init__(self):
        if self.lo >= self.hi:
            raise ValueError('Upper frequency must be larger than lower frequency')

@dataclass(kw_only=True)
class RadioFreqRange:
    lo: float
    hi: float

    def __post_init__(self):
        if self.lo >= self.hi:
            raise ValueError(
                'Upper frequency must be larger than lower frequency')

@dataclass(kw_only=True)
class FrequencyInfo:
    '''
    Metadata for frequencies specifcied in use configuration or as part of a channel
    '''
    label: str = field(default=None)
    locked: bool = field(default=False)
    priority: int | None = field(default=None)

@dataclass(kw_only=True)
class ConfigFrequency(FrequencyInfo):
    '''
    A frequency specified in the configuration file
    '''
    rf: float | RadioFreqRange
    bb: int | BasebandFreqRange | None = field(default=None)
    saved: bool = field(default=True)

@dataclass(kw_only=True)
class ChannelFrequency(FrequencyInfo):
    '''
    A frequency that is in use by a channel.  These are used by Scanner
    processing and the user interface.
    '''
    rf: float
    bb: int
    active: bool
    hanging: bool

@dataclass(kw_only=True)
class ChannelMessage(FrequencyInfo):
    '''
    It represents the state of a channel as it is processed by a demodulator.
    As a result, it will change over the time of the demodulation.  It will also
    be embellished by Scanner before channel logging.

    A channel message is sent to the scanner from the demodulator via a callback.
    '''
    state: str   # 'on' | 'off' | 'act'
    rf: float
    bb: int
    channel: int    # demodulator number (0-N)
    file: Optional[str] = None
    classification: Optional[str] = None
    detail: Optional[str] = None

FrequencyList: TypeAlias = list[ConfigFrequency]
ChannelList: TypeAlias = list[ChannelFrequency]

@dataclass(kw_only=True)
class FrequencyConfiguration:
    '''
    Used to load the freqency configuration file.
    '''
    file_name: Path
    disable_lockout: bool
    disable_priority: bool

class FrequencyManager:

    def __init__(self, config: FrequencyConfiguration, channel_spacing: int) -> None:

        self.channel_spacing = channel_spacing   # used for frequency conversions
        self.center_freq = None
        self.config = config

    async def load(self) -> FrequencyList:
        self.frequencies: FrequencyList = []
        file = self.config.file_name

        if not file.name:
            return []

        if not file.exists():
            raise FileNotFoundError(
                f'Frequency file does not exist: {file}')

        logging.debug(f'Loading frequencies from {file}')

        # Process lockout file if it was provided
        with file.open(mode='r') as file:
            frequencies_config = yaml.safe_load(file)

        if 'frequencies' in frequencies_config:
            for freq in frequencies_config['frequencies']:
                entry = {
                    'label': freq['label'],
                    'locked': freq['lockout'] if 'lockout' in freq and not self.config.disable_lockout else False,
                    'priority': freq['priority'] if 'priority' in freq and not self.config.disable_priority else None,
                }

                if 'single' in freq:
                    entry['rf'] = freq['single']
                elif 'lo' and 'hi' in freq:
                    entry['rf'] = RadioFreqRange(lo=freq['lo'], hi=freq['hi'])
                else:
                    raise ValueError(f'Invalid frequency entry: {freq}')

                self.frequencies.append(
                    ConfigFrequency(**entry)
                )

        return self.frequencies

    async def add(self, rf: float, type: dict) -> FrequencyList:
        '''
        Add frequency to channels if not already there.  Otherwise, modify existing
        occurance.

        Args:
            rf (float): Radio frequency of tuned channel
            type (dict): Dictionary of frequency attributes
        '''

        if not isinstance(rf, float):
            raise NotImplementedError('Frequency range not yet supported')

        # Return the frequencies that match this frequency
        matching_frequencies = [frequency for frequency in self.frequencies
            if isinstance(frequency, ConfigFrequency) and rf == frequency.rf]

        if len(matching_frequencies) > 0:  # if more than one occurance change them all
            for frequency in matching_frequencies:
                for field in ['label', 'locked', 'priority']:
                    if field in type:
                        frequency.__setattr__(field, type[field])
                        frequency.saved = False

        else:  # no occurances of this frequency in the list
            label = type['label'] if 'label' in type else None
            locked = type['locked'] if 'locked' in type else False
            priority = type['priority'] if 'priority' in type else None

            entry = {
                'saved': False,
                'label': label,
                'locked': locked,
                'priority': priority,
            }

            bb_freq = None if self.center_freq is None else frequency_to_baseband(rf, self.center_freq, self.channel_spacing)
            entry['rf'] = rf
            entry['bb'] = bb_freq
            self.frequencies.append(ConfigFrequency(**entry))

        return self.frequencies

    def set_center(self, center_freq: int) -> FrequencyList:
        '''
        When the center frequency changes, we need to regenerate the baseband frequencies.
        '''
        self.center_freq = center_freq
        self.generate_baseband_frequencies()

        return self.frequencies


    def locked_out(self, bb: int) -> bool:
        '''
        Compare the channel to frequency and range lockouts

        Args:
            bb (int): Baseband frequency of tuned channel

        TODO:  Maybe this should be a cached function
        TODO:  Maybe return what lockouts where found (for GUI)
        '''
        locked = False
        for frequency in self.frequencies:
            if not frequency.locked:
                continue
            if isinstance(frequency.bb, BasebandFreqRange):
                if frequency.bb.lo <= bb <= frequency.bb.hi:
                    locked = True
            else:  # is this frequency locked out?
                if bb == frequency.bb:
                    locked = True
        return locked

    def is_priority(self, bb: int) -> int | None:
        '''
        Compare the channel to frequency and range priorities

        Args:
            bb (int): Baseband frequency of tuned channel
        '''
        priority: int | None = None
        for frequency in self.frequencies:
            if frequency.priority is None:
                continue
            if isinstance(frequency.bb, BasebandFreqRange):
                if frequency.bb.lo <= bb <= frequency.bb.hi:
                    if priority is None or frequency.priority < priority:
                        priority = frequency.priority
            else:  # isingle channel
                if bb == frequency.bb:
                    if priority is None or frequency.priority < priority:
                        priority = frequency.priority

        return priority

    def is_higher_priority(self, channel_bb: int, demod_freq: int) -> bool:
        '''
        Compare priorities of the frequency of the channel at point in sweep to
        what is currently being demodulated

        Args:
            channel_bb (int): Baseband frequency of the tuned channel
            demod_freq (int): Baseband frequency of the demod frequency
        '''
        if demod_freq == 0:
            return True

        if self.config.disable_priority:
            return False

        channel_priority = self.is_priority(channel_bb)

        if channel_priority is None:
            return False  # there is no channel priority, therefore channel is lower priority

        demod_priority = self.is_priority(demod_freq)

        if demod_priority is None:
            return True   # there is a channel priority but no demod priority, therefore channel is higher priority

        if channel_priority < demod_priority:  # channel is higher priority than current demod frequency
            return True
        else:
            return False

    def generate_baseband_frequencies(self) -> None:
        '''
        Generate frequencies in baseband.  The scanner
        uses this as it tracks channels in baseband frequencies.
        '''
        for frequency in self.frequencies:
            if isinstance(frequency.rf, float):
                frequency.bb = frequency_to_baseband(frequency.rf,
                                                     self.center_freq,
                                                     self.channel_spacing)
            elif isinstance(frequency.rf, RadioFreqRange):
                frequency.bb = BasebandFreqRange(lo=frequency_to_baseband(frequency.rf.lo,
                                                                          self.center_freq,
                                                                          self.channel_spacing),
                                                 hi=frequency_to_baseband(frequency.rf.hi,
                                                                          self.center_freq,
                                                                          self.channel_spacing))
            else:
                logging.error(f'invalid frequency: {frequency}')

    def get_label(self, rf: float) -> str | None:
        '''
        Get the label for a frequency.  If there is not a label for the frequency then
        return the label for the range of frequencies (if any)

        Args:
            rf (float): Radio frequency of tuned channel
        '''
        range_label: str | None = None
        for freq_entry in self.frequencies:
            if isinstance(freq_entry.rf, float):
                if freq_entry.rf == rf and freq_entry.label is not None:
                    return freq_entry.label
            elif isinstance(freq_entry.rf, RadioFreqRange):
                if freq_entry.rf.lo <= rf <= freq_entry.rf.hi:
                    range_label = freq_entry.label

        return range_label

async def main() -> None:

    file = Path('./frequencies-example.yaml')
    channel_spacing = 5000
    config = FrequencyConfiguration(file_name=file, disable_lockout=False, disable_priority=False)
    frequency_manager = FrequencyManager(config, channel_spacing)
    await frequency_manager.load()

    print(f'Frequencies added: {len(frequency_manager.frequencies)}')

    # invalid range
    try:
        RadioFreqRange(lo=500, hi=400)
    except ValueError:
        print('Got the expected error for upper frequency less than lower frequency')

    # add a single frequency
    frequency = 500.0
    label = 'Test frequency'
    await frequency_manager.add(frequency, {'label': label, 'locked': True})

    added_frequency = frequency_manager.frequencies[-1]

    print(added_frequency)

    if added_frequency.rf == frequency:
        print('Frequency was added as expected')
    else:
        print('Frequency was NOT added (unexpected result)')

    if added_frequency.label == label:
        print('Frequency label was added as expected')
    else:
        print('Frequency label was NOT added (unexpected result)')

    frequency_manager.set_center(int(frequency*1E6))   # set the baseband frequencies
    if frequency_manager.frequencies[-1].bb == 0:
        print('Baseband frequency was set correctly')
    else:
        print('Baseband frequency was NOT set correctly (unexpected result)')

    # check if locked out
    if frequency_manager.locked_out(0):
        print('Frequency was locked out as expected')
    else:
        print('Frequency was NOT locked out (unexpected result)')

    # testing frequency range
    lo = 400.0
    hi = 500.0
    frequency_range = RadioFreqRange(lo=lo, hi=hi)
    try:
        await frequency_manager.add(frequency_range, {'locked': True, 'priority': True})
    except NotImplementedError:
        print('Got the expected error for frequency range')
    except:
        print('Got an unexpected error for frequency range')

    # frequency_manager.set_center(int(lo*1E6))   # set the baseband frequencies
    # # check if locked out
    # if frequency_manager.locked_out(0):
    #     print('Frequency was locked out by a range as expected')
    # else:
    #     print('Frequency was NOT locked out by a range (unexpected result)')

if __name__ == '__main__':

    import asyncio

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
