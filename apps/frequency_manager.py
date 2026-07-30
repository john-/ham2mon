"""
Handle frequency data used for internal proccessing and the user interface.

TODO: Save function
"""


from dataclasses import dataclass, field
from typing import Optional, TypeAlias  # TypeAlias needed for python < 3.12
from pathlib import Path
import os
import yaml
import logging
from utilities import frequency_to_baseband

logger = logging.getLogger(f"ham2mon.{__name__}")


@dataclass(kw_only=True)
class FrequencyInfo:
    '''
    Metadata for frequencies specified in use configuration or as part of a channel
    '''
    label: str = field(default=None)
    locked: bool = field(default=False)
    priority: int | None = field(default=None)
    ctcss: float | None = field(default=None)

    def __post_init__(self):
        if not isinstance(self.locked, bool):
            raise ValueError('Locked must be a boolean')

        if self.priority is not None:
            if not isinstance(self.priority, int) or self.priority < 1:
                raise ValueError('Priority must be an integer >= 1')

        if self.ctcss is not None:
            try:
                self.ctcss = float(self.ctcss)
            except (ValueError, TypeError):
                raise ValueError('CTCSS must be a float or integer representing frequency in Hz')
            if self.ctcss <= 0:
                raise ValueError('CTCSS must be a positive number')



@dataclass(kw_only=True, eq=False)
class ConfigFrequency(FrequencyInfo):
    '''
    A frequency specified in the configuration file.

    The baseband frequencies are not provided by the user.  They are
    calculated at run time
    '''
    single: float | None = field(default=None)
    lo: float | None = field(default=None)
    hi: float | None = field(default=None)

    bb_single: int | None = field(default=None)
    bb_lo: int | None = field(default=None)
    bb_hi: int | None = field(default=None)

    # 'add' means added through scanning
    mode: str | None = field(default=None)
    saved: bool = field(default=False)
    # if not a single it is a range
    is_single: bool | None = field(default=None)

    # All CTCSS tones considered valid for this frequency/range. Populated from
    # `ctcss` on construction, and may be extended later via FrequencyManager.add()
    # when the same frequency/range is declared again in config with a different
    # `ctcss` value (e.g. a repeater with a primary and a backup PL tone). `ctcss`
    # itself remains the first/primary tone, kept for backward compatibility with
    # code that only expects a single value (e.g. get_ctcss_info()).
    ctcss_tones: list[float] = field(default_factory=list, init=False, repr=False)
    ctcss_labels: list[str] = field(default_factory=list, init=False, repr=False)


    def calculate_baseband(self, center_freq: int, channel_spacing: int) -> None:
        if self.is_single:
            self.bb_single = frequency_to_baseband(
                self.single, center_freq, channel_spacing)
        else:
            self.bb_lo = frequency_to_baseband(
                self.lo, center_freq, channel_spacing)
            self.bb_hi = frequency_to_baseband(
                self.hi, center_freq, channel_spacing)

    def locks_out(self, bb: int) -> bool:
        if not self.locked:
            return False

        if self.is_single and self.bb_single == bb:
            return True
        elif not self.is_single and self.bb_lo <= bb <= self.bb_hi:
            return True
        else:
            return False

    def get_priority_at(self, bb: int) -> int | bool:

        if self.priority is None:
            return None

        if self.is_single and self.bb_single == bb:
            return self.priority
        elif not self.is_single and self.bb_lo <= bb <= self.bb_hi:
            return self.priority

        return None

    def __post_init__(self):
        # Call parent validation first
        super().__post_init__()

        # Validate frequency types
        self._validate_frequency_types()

        # Validate frequency specification (single or range)
        self._validate_frequency_specification()

        # Validate frequency values
        self._validate_frequency_values()

        # Set state
        self.is_single = self.single is not None
        self.ctcss_tones = [self.ctcss] if self.ctcss is not None else []
        self.ctcss_labels = [self.label] if (self.ctcss is not None and self.label is not None) else []


    def _validate_frequency_types(self):
        """Ensure all frequency values are floats if provided"""
        for attr_name, attr_value in [
            ('single', self.single),
            ('lo', self.lo),
            ('hi', self.hi)
        ]:
            if attr_value is not None and not isinstance(attr_value, float):
                raise ValueError(f'{attr_name} frequency must be a float')

    def _validate_frequency_specification(self):
        """Ensure frequency is specified correctly as either single or range"""
        has_single = self.single is not None
        has_lo = self.lo is not None
        has_hi = self.hi is not None

        # Check if any frequency is specified
        if not has_single and not has_lo and not has_hi:
            raise ValueError('Frequency must be specified as single or range')

        # Check for mixed single and range specifications
        if has_single and (has_lo or has_hi):
            raise ValueError(
                'Frequency cannot be specified as both single and range')

        # Check for incomplete range specification
        if has_lo != has_hi:  # XOR operation - one is set but not the other
            raise ValueError(
                'Both lo and hi must be specified for a frequency range')

    def _validate_frequency_values(self):
        """Ensure frequency values are valid"""
        # Check for negative frequencies
        if (self.single is not None and self.single < 0.0) or (self.lo is not None and self.lo < 0.0):
            raise ValueError('Frequencies must be positive numbers')

        # Check range order
        if self.lo is not None and self.hi is not None and self.lo >= self.hi:
            raise ValueError(
                'Upper frequency (hi) must be larger than lower frequency (lo)')

    def __eq__(self, other) -> bool:
        if not isinstance(other, ConfigFrequency):
            return NotImplemented

        if self.single and other.single:
            return self.single == other.single

        if self.lo and other.lo:
            return self.lo == other.lo and self.hi == other.hi


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
    matched_ctcss: float | None = field(default=None)  # actively matched CTCSS tone during a live transmission; None when idle or unmatched
    ctcss_tones: list[float] = field(default_factory=list)



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
    signal_db: Optional[int] = None
    matched_ctcss: float | None = field(default=None)  # CTCSS tone that matched and opened squelch for this transmission

    def __str__(self) -> str:
        rf_mhz = f"{self.rf:.3f} MHz" if self.rf else "?"
        ch_str = f"Ch {self.channel}"
        state_str = f"{self.state.upper():<3}"

        parts = [f"{ch_str}: {state_str}", rf_mhz]

        if self.label:
            parts.append(f"[{self.label}]")

        if self.file:
            parts.append(f"Saved: {os.path.basename(self.file)}")
        elif self.detail:
            parts.append(self.detail)

        if self.classification:
            parts.append(f"({self.classification})")

        if self.signal_db is not None:
            parts.append(f"[{self.signal_db} dB]")

        return " | ".join(parts)


FrequencyList: TypeAlias = list[ConfigFrequency]
ChannelList: TypeAlias = list[ChannelFrequency]


@dataclass(kw_only=True)
class FrequencyConfiguration:
    '''
    Used to load the freqency configuration file.
    '''
    file_name: Optional[Path] = None
    disable_lockout: bool
    disable_priority: bool
    max_ctcss_tones: int = 0


class FrequencyManager:

    def __init__(self, config: FrequencyConfiguration, channel_spacing: int) -> None:

        self.channel_spacing = channel_spacing   # used for frequency conversions
        self.center_freq = None
        self.config = config
        self.frequencies: FrequencyList = []

    async def process_frequencies_data(self, frequencies_config) -> FrequencyList:
        """Process pre-loaded frentryequencies configuration data."""

        if 'frequencies' in frequencies_config:
            for freq in frequencies_config['frequencies']:
                freq['saved'] = True
                await self.add(freq)

        return self.frequencies

    async def load(self) -> FrequencyList:
        """Load frequencies from the configured file."""
        self.frequencies = []

        if not self.config.file_name:
            return []

        file = self.config.file_name
        if not file.exists():
            raise FileNotFoundError(f'Frequency file does not exist: {file}')

        logger.debug(f'Loading frequencies from {file}')
        with file.open(mode='r') as file:
            try:
                frequencies_config = yaml.safe_load(file)
            except yaml.YAMLError as e:
                if hasattr(e, 'problem_mark'):
                    logger.error(
                        f'{e.problem_mark} {e.problem} {e.context if e.context else ""}')
                else:
                    logger.error(
                        f'Something went wrong while parsing yaml file: {file}')
                raise Exception(
                    "Invalid yaml frequency file (enable debugging for more info)")

            return await self.process_frequencies_data(frequencies_config)

    async def add(self, entry: dict) -> FrequencyList:
        '''
        Add frequency to channels if not already there.

        A frequency/range may legitimately be declared more than once in config
        when the only difference is the `ctcss` tone (e.g. a repeater that
        answers to a primary and a backup PL tone). In that case, the new tone
        is merged into the existing entry's `ctcss_tones` list rather than
        raising. Any other kind of duplicate (an identical entry, or a repeat
        that doesn't add a new tone) is still an error.

        Args:
            entry (dict): Dictionary of frequency attributes

            example:
                entry = {
                    'lo': 145.0, 'hi': 148.0,
                    'label':'Test range',
                    'locked':True,
                    'priority': 1
                }

        Returns:
                FrequencyList: List of frequencies

        Raises:
            ValueError: If the frequency already occurs in the list and is not
                a mergeable CTCSS-only variant of an existing entry.
        '''
        wanted = ConfigFrequency(**entry)

        if wanted.ctcss is not None:
            if self.config.max_ctcss_tones <= 0:
                raise ValueError(
                    f"CTCSS is disabled (max_ctcss_tones={self.config.max_ctcss_tones}) "
                    f"but frequency config specifies ctcss: {wanted.ctcss}")
            if len(wanted.ctcss_tones) > self.config.max_ctcss_tones:
                raise ValueError(
                    f"Frequency config specifies {len(wanted.ctcss_tones)} CTCSS tones "
                    f"but max_ctcss_tones is limited to {self.config.max_ctcss_tones}")

        # use the dataclass __eq__ functions to look for matches
        matching_frequencies = [existing for existing in self.frequencies
                                if wanted == existing]

        if len(matching_frequencies) > 0:
            existing = matching_frequencies[0]

            if wanted.ctcss is not None and wanted.ctcss not in existing.ctcss_tones:
                self._merge_ctcss_tone(existing, wanted, entry)
                return self.frequencies

            # Already one occurance, and no new tone to merge, so this is an error
            raise ValueError(
                f'Frequency {wanted} already occurs in list')

        # add the basband if center frequency has been set
        if self.center_freq:
            wanted.calculate_baseband(self.center_freq, self.channel_spacing)

        self.frequencies.append(wanted)

        return self.frequencies

    def _merge_ctcss_tone(self, existing: ConfigFrequency, wanted: ConfigFrequency, entry: dict) -> None:
        '''
        Merge an additional CTCSS tone into an already-loaded frequency/range.

        The first-seen entry's label/priority/locked remain authoritative for
        the merged entry. A conflicting, explicitly-specified priority on the
        duplicate entry is treated as a config error, since it's ambiguous
        which one should apply. (locked/label conflicts are intentionally not
        validated in this phase.)

        Args:
            existing (ConfigFrequency): The already-loaded entry to merge into
            wanted (ConfigFrequency): The newly parsed duplicate entry
            entry (dict): The raw dict passed to add(), used to distinguish
                "not specified" from "explicitly set" for conflict checks
        '''
        if self.config.max_ctcss_tones <= 0:
            raise ValueError(
                f"CTCSS is disabled (max_ctcss_tones={self.config.max_ctcss_tones}) "
                f"but frequency config specifies ctcss: {wanted.ctcss}")

        if len(existing.ctcss_tones) >= self.config.max_ctcss_tones:
            raise ValueError(
                f"Cannot merge CTCSS tone {wanted.ctcss} into "
                f"{existing.label!r}: exceeds max_ctcss_tones limit of {self.config.max_ctcss_tones}")

        if 'priority' in entry and existing.priority is not None and wanted.priority != existing.priority:
            raise ValueError(
                f"Cannot merge CTCSS tone {wanted.ctcss} into "
                f"{existing.label!r}: priority {wanted.priority} conflicts "
                f"with existing priority {existing.priority}")

        existing.ctcss_tones.append(wanted.ctcss)
        existing.ctcss_labels.append(wanted.label or existing.label or "")
        logger.debug(
            f'Merged CTCSS tone {wanted.ctcss} into existing frequency '
            f'{existing.label!r} (tones now {existing.ctcss_tones})')


    async def change(self, entry: dict) -> FrequencyList:
        '''
        Modify frequency or frequency range.

        Args:
            entry (dict): Dictionary of frequency attributes

        Returns:
            FrequencyList: List of frequencies

        Raises:
            ValueError: If the frequency is not found in the frequencies list
        '''
        # Create a temporary ConfigFrequency to use for comparison
        new_values = ConfigFrequency(**entry)

        # Find the matching frequency
        for frequency in self.frequencies:
            # Use the __eq__ method that's already defined in ConfigFrequency
            if frequency == new_values:
                # Update fields if they exist in the entry
                for field in ['label', 'priority', 'locked']:
                    if field in entry:
                        setattr(frequency, field, entry[field])

                # CTCSS tone merging in change()
                if 'ctcss' in entry and entry['ctcss'] is not None:
                    if entry['ctcss'] not in frequency.ctcss_tones:
                        self._merge_ctcss_tone(frequency, new_values, entry)

                return self.frequencies

        if 'mode' in entry and entry['mode'] == 'add':
            return await self.add(entry)

        raise ValueError(
            f'Frequency {entry} not found in frequencies list')

    def set_center(self, center_freq: int) -> FrequencyList:
        '''
        When the center frequency changes, we need to regenerate the baseband frequencies.

        Args:
            center_freq (int): Hardware RF center frequency in Hz
        '''
        self.center_freq = center_freq
        self.generate_baseband_frequencies()

        return self.frequencies

    def locked_out(self, bb: int) -> bool:
        '''
        Compare the channel to lockouts for each configured frequency.

        Args:
            bb (int): Baseband frequency of tuned channel

        TODO:  Maybe this should be a cached function
        TODO:  Maybe return what lockouts where found (for GUI)
        '''
        if self.config.disable_lockout:
            return False

        for frequency in self.frequencies:
            if frequency.locks_out(bb):
                return True

        return False

    def is_priority(self, bb: int) -> int | None:
        '''
        Compare the channel to frequency and range priorities

        A frequency can occur in multiple places.  For example, as a single frequnecy as
        well as in a range.  Therefore, we need to check all the frequency entries.

        Individual priorities take precedence over any priority assigned to a range
        that the frequency is a part of.

        Args:
            bb (int): Baseband frequency of tuned channel
        '''
        lowest: int | None = None
        for frequency in self.frequencies:
            priority = frequency.get_priority_at(bb)
            if priority is not None:
                if frequency.single:
                    return priority
                else:
                    if lowest is None or priority < lowest:
                        lowest = priority

        return lowest

    def get_priority_info(self, bb: int) -> tuple[int | None, bool]:
        '''
        Compare the channel to frequency and range priorities and return both
        the priority level and whether it was automatically assigned.

        Args:
            bb (int): Baseband frequency of tuned channel
        '''
        for frequency in self.frequencies:
            priority = frequency.get_priority_at(bb)
            if priority is not None and frequency.single:
                return priority, frequency.mode == 'add'

        lowest: int | None = None
        is_auto = False
        for frequency in self.frequencies:
            priority = frequency.get_priority_at(bb)
            if priority is not None and not frequency.single:
                if lowest is None or priority < lowest:
                    lowest = priority
                    is_auto = frequency.mode == 'add'

        return lowest, is_auto


    def get_ctcss_info(self, rf_freq: float) -> float | None:
        """Get the primary CTCSS tone frequency (in Hz) for the given absolute RF frequency.

        NOTE: if multiple tones are configured for this frequency (see
        get_ctcss_tones), this returns only the first/primary one. Kept for
        backward compatibility with callers that only support a single tone
        (currently: the single-tone squelch in BaseTuner). Callers that want to
        validate against all configured tones should use get_ctcss_tones().
        """
        # Check single frequencies first
        for frequency in self.frequencies:
            if frequency.is_single and abs(frequency.single - rf_freq) < 1e-4:
                if frequency.ctcss is not None:
                    return frequency.ctcss

        # Then check ranges
        for frequency in self.frequencies:
            if not frequency.is_single and frequency.lo <= rf_freq <= frequency.hi:
                if frequency.ctcss is not None:
                    return frequency.ctcss

        return None

    def get_ctcss_tones(self, rf_freq: float) -> list[float]:
        """Get all CTCSS tone frequencies (in Hz) configured for the given absolute RF frequency.

        Returns an empty list if the frequency isn't configured, or is
        configured without any CTCSS tone.
        """
        # Check single frequencies first
        for frequency in self.frequencies:
            if frequency.is_single and abs(frequency.single - rf_freq) < 1e-4:
                if frequency.ctcss_tones:
                    return list(frequency.ctcss_tones)

        # Then check ranges
        for frequency in self.frequencies:
            if not frequency.is_single and frequency.lo <= rf_freq <= frequency.hi:
                if frequency.ctcss_tones:
                    return list(frequency.ctcss_tones)

        return []

    def is_higher_priority(self, channel_bb: int, demod_freq: int) -> bool:
        '''
        Compare priorities of the frequency of the channel at point in sweep to
        what is currently being demodulated.

        Args:
            channel_bb (int): Baseband frequency of the tuned channel
            demod_freq (int): Baseband frequency of the current demodulator
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
            frequency.calculate_baseband(
                self.center_freq, self.channel_spacing)


    def get_label(self, rf: float, ctcss: float | None = None) -> str | None:
        '''
        Get the label for a frequency.  If there is not a label for the frequency then
        return the label for the range of frequencies (if any)

        Args:
            rf (float): Radio frequency of tuned channel
            ctcss (float, optional): Matched CTCSS tone frequency
        '''
        range_label: str | None = None
        for freq_entry in self.frequencies:
            if freq_entry.is_single:
                if freq_entry.single == rf:
                    if ctcss is not None and ctcss in freq_entry.ctcss_tones:
                        idx = freq_entry.ctcss_tones.index(ctcss)
                        if idx < len(freq_entry.ctcss_labels):
                            return freq_entry.ctcss_labels[idx]
                    return freq_entry.label
            else:
                if freq_entry.lo <= rf <= freq_entry.hi:
                    range_label = freq_entry.label
                    if ctcss is not None and ctcss in freq_entry.ctcss_tones:
                        idx = freq_entry.ctcss_tones.index(ctcss)
                        if idx < len(freq_entry.ctcss_labels):
                            range_label = freq_entry.ctcss_labels[idx]

        return range_label


async def main() -> None:  # pragma: no cover

    print('For testing this module use pytest')


if __name__ == '__main__':  # pragma: no cover

    import asyncio

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
