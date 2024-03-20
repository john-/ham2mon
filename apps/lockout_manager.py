
from dataclasses import dataclass, field
from typing import TypeAlias  # TypeAlias needed for python < 3.12
from pathlib import Path
import yaml
import logging
from utilities import frequency_to_baseband

@dataclass(kw_only=True)
class BasebandFreq:
    freq: int

@dataclass(kw_only=True)
class BasebandFreqRange:
    min: int
    max: int

    def __post_init__(self):
        if self.min >= self.max:
            raise ValueError('Upper frequency must be larger than lower frequency')

@dataclass(kw_only=True)
class RadioFreq:
    freq: float
    saved: bool = field(default=True)

@dataclass(kw_only=True)
class RadioFreqRange:
    min: float
    max: float
    saved: bool = field(default=True)

    def __post_init__(self):
        if self.min >= self.max:
            raise ValueError('Upper frequency must be larger than lower frequency')

LockoutListBbFreq: TypeAlias = list[BasebandFreq | BasebandFreqRange]  # the bb freqs used for scanner processing
LockoutListRadioFreq: TypeAlias = list[RadioFreq | RadioFreqRange]  # includes metadata for unsaved lockouts

class LockoutManager:

    def __init__(self, lockout_file: Path, center_freq: int, channel_spacing: int) -> None:

        logging.debug('Creating lockout manager')

        self.lockouts: LockoutListRadioFreq = []
        self.lockout_baseband: LockoutListBbFreq = []
        self.lockout_file = lockout_file
        self.center_freq = center_freq
        self.channel_spacing = channel_spacing   # used for frequency conversions

        self.load()

    def load(self) -> LockoutListRadioFreq:
        # Clear the lockout channels
        self.lockouts = []

        if not self.lockout_file.name:
            return []
        
        if not self.lockout_file.exists():
            raise FileNotFoundError(f'Lockout file does not exist: {self.lockout_file}')

        # Process lockout file if it was provided
        with self.lockout_file.open(mode='r') as file:
            lockout_config = yaml.safe_load(file)

        # Individual frequencies
        for freq in lockout_config['frequencies']:
            self.lockouts.append(RadioFreq(freq=freq))

        # Ranges of frequencies
        for a_range in lockout_config['ranges']:
            self.lockouts.append(RadioFreqRange(min=a_range['min'], max=a_range['max']))

        return self.lockouts

    def add(self, freq: float) -> LockoutListRadioFreq:
        '''
        Add frequency to lockout channels if not already there.

        Args:
            freq (float): Radio frequency of tuned channel
        '''
        if not [lockout for lockout in self.lockouts
                if isinstance(lockout, RadioFreq) and freq == lockout.freq]:
            self.lockouts.append(RadioFreq(freq=freq, saved=False))

        self.generate_baseband_lockout_channels()

        return self.lockouts
            
    def update(self, center_freq: int) -> None:
        self.center_freq = center_freq
        self.generate_baseband_lockout_channels()


    def locked_out(self, channel: int) -> bool:
        '''
        Compare the channel to frequency and range lockouts
        '''
        locked = False
        for lockout_channel in self.lockout_baseband:
            if isinstance(lockout_channel, BasebandFreqRange):
                if lockout_channel.min <= channel <= lockout_channel.max:
                    locked = True
            else:  # is this frequency locked out?
                if channel == lockout_channel.freq:
                    locked = True
        return locked

    def generate_baseband_lockout_channels(self) -> None:
        '''
        Generate lockout frequencies in baseband.  It is only
        used for locked_out().
        '''
        self.lockout_baseband = []
        for lockout in self.lockouts:
            if isinstance(lockout, RadioFreq):
                self.lockout_baseband.append(
                    BasebandFreq(freq=frequency_to_baseband(lockout.freq,
                                                            self.center_freq,
                                                            self.channel_spacing)))
            elif isinstance(lockout, RadioFreqRange):
                self.lockout_baseband.append(
                    BasebandFreqRange(min=frequency_to_baseband(lockout.min,
                                                                self.center_freq,
                                                                self.channel_spacing),
                                      max=frequency_to_baseband(lockout.max,
                                                                self.center_freq,
                                                                self.channel_spacing)))
            else:
                logging.error(f'invalid lockout: {lockout}')

def main() -> None:

    file = Path('/cart/ham2mon/apps/lockout-example.yaml')
    lockout_manager = LockoutManager(file, int(float(460.4E6)), 5000)

    print(f'lockouts: {lockout_manager.lockouts}')

    # invalid range
    try:
        RadioFreqRange(min=500, max=400)
    except ValueError:
        print('Got the expected error for upper frequency less than lower frequency')

if __name__ == '__main__':

    main()


