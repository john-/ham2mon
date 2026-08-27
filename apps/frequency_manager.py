"""
Handle frequency data used for internal proccessing and the user interface.

TODO: Save function
"""


import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias  # TypeAlias needed for python < 3.12

import yaml
from utilities import frequency_to_baseband

logger = logging.getLogger(f"ham2mon.{__name__}")

# Match tolerances shared across FrequencyManager RF/tone matching
# (resolve_banks, get_label, get_ctcss_info, get_ctcss_tones).
FREQ_MATCH_TOLERANCE_MHZ = 1e-4
TONE_MATCH_TOLERANCE_HZ = 0.5


@dataclass(kw_only=True)
class ToneRule:
    """Structured CTCSS tone rule for single frequencies or ranges."""
    ctcss: float
    label: str | None = None
    banks: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        try:
            self.ctcss = float(self.ctcss)
        except (ValueError, TypeError):
            raise ValueError("CTCSS must be a float or integer representing frequency in Hz")
        if self.ctcss <= 0:
            raise ValueError("CTCSS must be a positive number")
        if isinstance(self.banks, str):
            self.banks = [self.banks]


def _coerce_tone_rule(rule: ToneRule | dict[str, Any]) -> ToneRule:
    return rule if isinstance(rule, ToneRule) else ToneRule(**rule)


@dataclass(kw_only=True)
class FrequencyInfo:
    '''
    Metadata for frequencies specified in use configuration or as part of a channel
    '''
    label: str = field(default=None)
    locked: bool = field(default=False)
    priority: int | None = field(default=None)
    ctcss: float | None = field(default=None)
    banks: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.locked, bool):
            raise ValueError('Locked must be a boolean')

        if self.priority is not None:
            if not isinstance(self.priority, int) or self.priority < 1:
                raise ValueError('Priority must be an integer >= 1')

        if isinstance(self.banks, str):
            self.banks = [self.banks]

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

    # tones: list of CTCSS tone rules with optional per-tone labels and banks.
    # Config-domain only — ChannelFrequency and ChannelMessage do not carry tones.
    tones: list[ToneRule] = field(default_factory=list)

    # All CTCSS tones considered valid for this frequency/range. Populated from
    # the normalized `tones` rules, with any scalar `ctcss` prepended. `ctcss`
    # remains the first/primary tone, kept for backward compatibility with code
    # that only expects a single value (e.g. get_ctcss_info()).
    ctcss_tones: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        # Call parent validation first (banks norm, ctcss float cast/validation)
        super().__post_init__()

        # Convert any raw dict items under tones: into ToneRule instances
        if self.tones:
            self.tones = [_coerce_tone_rule(t) for t in self.tones]

        if self.ctcss is not None and not self.tones:
            # Normalize scalar ctcss into tones list if tones is empty
            self.tones = [ToneRule(ctcss=self.ctcss, label=self.label, banks=self.banks)]

        # Ensure any tone rule missing banks inherits parent entry's banks
        for tone_rule in self.tones:
            if not tone_rule.banks and self.banks:
                tone_rule.banks = list(self.banks)

        # Validate frequency types
        self._validate_frequency_types()

        # Validate frequency specification (single or range)
        self._validate_frequency_specification()

        # Validate frequency values
        self._validate_frequency_values()

        # Set state
        self.is_single = self.single is not None
        self.ctcss_tones = []
        if self.ctcss is not None:
            self.ctcss_tones.append(self.ctcss)
        for tone_rule in self.tones:
            if tone_rule.ctcss not in self.ctcss_tones:
                self.ctcss_tones.append(tone_rule.ctcss)


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
    file: str | None = None
    classification: str | None = None
    detail: str | None = None
    signal_db: int | None = None
    matched_ctcss: float | None = field(default=None)  # CTCSS tone that matched and opened squelch for this transmission
    wav_tmp_path: str | None = None
    discard: bool = False
    started_at: float | None = None
    duration_sec: float | None = None
    """Duration in seconds, populated by Scanner after the final WAV file is written."""

    def __str__(self) -> str:
        rf_mhz = f"{self.rf:.3f} MHz" if self.rf else "?"
        ch_str = f"Ch {self.channel}"
        state_str = f"{self.state.upper():<3}"

        parts = [f"{ch_str}: {state_str}", rf_mhz]

        if self.label:
            parts.append(f"[{self.label}]")

        if self.banks:
            parts.append(f"[{','.join(self.banks)}]")

        if self.file:
            parts.append(f"Saved: {os.path.basename(self.file)}")
        elif self.detail:
            parts.append(self.detail)

        if self.classification:
            parts.append(f"({self.classification})")

        if self.signal_db is not None:
            parts.append(f"[{self.signal_db} dB]")

        return " | ".join(parts)


@dataclass(frozen=True, kw_only=True)
class TransmissionRecord:
    """Immutable snapshot of a completed, kept transmission.

    Built by Scanner._process_completed_transmission() after os.rename() succeeds
    and passed to ActivityLogger.log() alongside the updated ChannelMessage.

    Attributes:
        rf: RF centre frequency in MHz (e.g. 460.1250).
        bb_hz:  Baseband offset from SDR centre in Hz.
        channel: Demodulator index (0-N).
        label: Frequency label from frequency-policy file, or None.
        priority: Numeric priority level, or None.
        matched_ctcss_hz: Matched CTCSS tone in Hz, or None.
        signal_db: Average signal strength in dB, or None.
        classification: Classifier result code (e.g. 'V', 'D', 'S'), or None.
        wav_path: Final absolute path to the saved WAV file.
        started_at: Unix timestamp when the transmission started.
        duration_sec: Duration of the transmission in seconds.
        metadata: Free-form dict populated by Phase 4 TransmissionComponent processors
            (e.g. transcription text, classifier confidence scores). Empty by default.
    """
    rf: float
    bb_hz: int
    channel: int
    label: str | None
    priority: int | None
    matched_ctcss_hz: float | None
    signal_db: int | None
    classification: str | None
    wav_path: str
    started_at: float
    duration_sec: float
    metadata: dict = field(default_factory=dict)  # type: ignore[reportGeneralTypeIssues]
    """Free-form metadata populated by TransmissionComponent processors in Phase 4
    (e.g. transcription text, classifier confidence scores)."""
    banks: list[str] = field(default_factory=list)
    """List of scanner bank tags assigned to this transmission."""


    def __str__(self) -> str:
        """Return a concise human-readable summary of the transmission."""
        parts = [f"{self.rf:.4f} MHz", f"{self.duration_sec:.1f}s"]
        if self.label:
            parts.append(f"[{self.label}]")
        if self.banks:
            parts.append(f"[{','.join(self.banks)}]")
        if self.priority is not None:
            parts.append(f"P{self.priority}")
        if self.classification:
            parts.append(self.classification)
        if self.matched_ctcss_hz is not None:
            parts.append(f"{self.matched_ctcss_hz:.1f}Hz")
        return " ".join(parts)


FrequencyList: TypeAlias = list[ConfigFrequency]
ChannelList: TypeAlias = list[ChannelFrequency]


@dataclass(kw_only=True)
class FrequencyConfiguration:
    '''
    Used to load the freqency configuration file.
    '''
    file_name: Path | None = None
    disable_lockout: bool
    disable_priority: bool
    max_ctcss_tones: int = 0


class FrequencyManager:

    def __init__(self, config: FrequencyConfiguration, channel_spacing: int) -> None:

        self.channel_spacing = channel_spacing   # used for frequency conversions
        self.center_freq = None
        self.config = config
        self.frequencies: FrequencyList = []
        self.active_banks: set[str] = set()

    def set_active_banks(self, banks: list[str] | set[str] | None) -> None:
        """Set the active bank tags for filtering and stepping inclusion."""
        if banks is None:
            self.active_banks = set()
        elif isinstance(banks, str):
            self.active_banks = {banks}
        else:
            self.active_banks = set(banks)

    def is_bank_active(self, bank_list: list[str]) -> bool:
        """Return True if active_banks is empty (promiscuous scan all mode),
        or if active_banks intersects with bank_list."""
        if not self.active_banks:
            return True
        return bool(self.active_banks.intersection(bank_list))

    def configured_banks(self) -> set[str]:
        """Union of all bank tags configured across loaded frequencies and tone rules."""
        banks: set[str] = set()
        for freq in self.frequencies:
            banks.update(freq.banks)
            for tone_rule in freq.tones:
                banks.update(tone_rule.banks)
        return banks

    def unknown_active_banks(self) -> set[str]:
        """Active bank tags that match no configured bank tag. The "SEARCH"
        and "UNTAGGED" dynamic tags are exempt. Returns empty set in
        promiscuous mode."""
        if not self.active_banks:
            return set()
        return self.active_banks - self.configured_banks() - {"SEARCH", "UNTAGGED"}

    def _warn_on_unmatched_active_banks(self) -> None:
        """Log a warning when active bank tags match no configured bank tag.

        Bank filtering is fail-closed (see is_bank_active), so a typo'd active
        bank (or a missing/empty frequency file) silently disables every
        channel — surface it at startup.
        """
        unknown = self.unknown_active_banks()
        if unknown:
            logger.warning(
                'Active bank(s) %s match no configured frequency/tone banks; bank filtering is fail-closed so channels in these will not be monitored. Check the --banks spelling and that -F/--frequencies points at a file defining matching banks.',
                sorted(unknown))

    def _untagged_fallback(self) -> list[str]:
        """Bank-filtering sentinel tag.

        "UNTAGGED" is only meaningful when the user opted into bank filtering
        (non-empty active_banks); it lets untagged hits fail-closed via
        is_bank_active().  In promiscuous/legacy mode (no --banks) return no
        tags so bank metadata stays empty for non-participants instead of
        injecting "UNTAGGED" into logs, fixed-field records, and JSON.
        """
        return ["UNTAGGED"] if self.active_banks else []

    def resolve_banks(self, rf: float, ctcss_hz: float | None = None) -> list[str]:
        """Resolve bank tags for a carrier hit at rf with decoded ctcss_hz
        using 5-tier precedence hierarchy:
        1. Explicit single entry tone match
        2. Explicit single entry base bank
        3. Range entry tone match
        4. Range entry base bank
        5. Dynamic fallback ("SEARCH" if active; else "UNTAGGED" when bank
           filtering is active, otherwise no tags)

        Note: "UNTAGGED" also arises from tiers 2 & 4 (and their pre-tuning
        union fallbacks) when a configured entry matches but sets no bank
        tags -- all such paths return _untagged_fallback(), the same gated
        sentinel as tier 5. Under bank filtering both an unbanked configured
        entry and an unmatched hit resolve to UNTAGGED and fail closed via
        is_bank_active(); in promiscuous mode both resolve to no tags.
        """
        # Tier 1 & 2: Check single frequencies first
        for freq in self.frequencies:
            if freq.is_single and freq.single is not None and abs(freq.single - rf) < FREQ_MATCH_TOLERANCE_MHZ:
                if ctcss_hz is not None:
                    for tone_rule in freq.tones:
                        if abs(tone_rule.ctcss - ctcss_hz) < TONE_MATCH_TOLERANCE_HZ:
                            return tone_rule.banks if tone_rule.banks else freq.banks
                    return freq.banks if freq.banks else self._untagged_fallback()
                else:
                    # Pre-tuning (no tone decoded yet): return Union of all possible bank tags
                    union_banks = set(freq.banks)
                    for tone_rule in freq.tones:
                        union_banks.update(tone_rule.banks)
                    return list(union_banks) if union_banks else self._untagged_fallback()

        # Tier 3 & 4: Check range entries
        for freq in self.frequencies:
            if not freq.is_single and freq.lo is not None and freq.hi is not None and freq.lo <= rf <= freq.hi:
                if ctcss_hz is not None:
                    for tone_rule in freq.tones:
                        if abs(tone_rule.ctcss - ctcss_hz) < TONE_MATCH_TOLERANCE_HZ:
                            return tone_rule.banks if tone_rule.banks else freq.banks
                    return freq.banks if freq.banks else self._untagged_fallback()
                else:
                    # Pre-tuning (no tone decoded yet): return Union of all possible bank tags
                    union_banks = set(freq.banks)
                    for tone_rule in freq.tones:
                        union_banks.update(tone_rule.banks)
                    return list(union_banks) if union_banks else self._untagged_fallback()

        # Tier 5: Outside configured entries
        if "SEARCH" in self.active_banks:
            return ["SEARCH"]
        return self._untagged_fallback()

    async def process_frequencies_data(self, frequencies_config: dict[str, Any]) -> FrequencyList:
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
            self._warn_on_unmatched_active_banks()
            return []

        file = self.config.file_name
        if not file.exists():
            raise FileNotFoundError(f'Frequency file does not exist: {file}')

        logger.debug(f'Loading frequencies from {file}')
        with file.open(mode='r') as file:
            try:
                frequencies_config: dict[str, Any] = yaml.safe_load(file)
            except yaml.YAMLError as e:
                if hasattr(e, 'problem_mark'):
                    logger.error(
                        f'{e.problem_mark} {e.problem} {e.context if e.context else ""}')
                else:
                    logger.error(
                        f'Something went wrong while parsing yaml file: {file}')
                raise Exception(
                    "Invalid yaml frequency file (enable debugging for more info)")

            _ = await self.process_frequencies_data(frequencies_config)
            self._warn_on_unmatched_active_banks()
            return self.frequencies

    async def add(self, entry: dict) -> FrequencyList:
        '''
        Add frequency to channels if not already there.

        Each frequency or range must be declared exactly once. To configure
        multiple CTCSS tones on a single frequency, use either a single
        ``ctcss:`` scalar (one tone) or the unified ``tones: [...]`` array
        (multiple tones with optional per-tone labels and banks).

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
            ValueError: If the frequency already occurs in the list.
        '''
        wanted = ConfigFrequency(**entry)

        if wanted.ctcss_tones:
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
            raise ValueError(
                f'Frequency {wanted} already occurs in list. Use tones: [...] on a single entry to configure multiple CTCSS tones.')

        # add the baseband if center frequency has been set
        if self.center_freq:
            wanted.calculate_baseband(self.center_freq, self.channel_spacing)

        self.frequencies.append(wanted)

        return self.frequencies



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

        Returns the scalar ``ctcss`` when configured; otherwise the first tone
        from the ``tones:`` list, so tones:-only entries still surface a primary
        tone. Kept single-valued for callers (e.g. the UI) that display one
        tone — use get_ctcss_tones() for the full set.
        """
        # Check single frequencies first
        for frequency in self.frequencies:
            if frequency.is_single and frequency.single is not None and abs(frequency.single - rf_freq) < FREQ_MATCH_TOLERANCE_MHZ:
                if frequency.ctcss is not None:
                    return frequency.ctcss
                if frequency.tones:
                    return frequency.tones[0].ctcss

        # Then check ranges
        for frequency in self.frequencies:
            if not frequency.is_single and frequency.lo is not None and frequency.hi is not None and frequency.lo <= rf_freq <= frequency.hi:
                if frequency.ctcss is not None:
                    return frequency.ctcss
                if frequency.tones:
                    return frequency.tones[0].ctcss

        return None

    def get_ctcss_tones(self, rf_freq: float) -> list[float]:
        """Get all CTCSS tone frequencies (in Hz) configured for the given absolute RF frequency.

        Returns an empty list if the frequency isn't configured, or is
        configured without any CTCSS tone.
        """
        # Check single frequencies first
        for frequency in self.frequencies:
            if frequency.is_single and frequency.single is not None and abs(frequency.single - rf_freq) < FREQ_MATCH_TOLERANCE_MHZ:
                tones = list(frequency.ctcss_tones)
                for tone_rule in frequency.tones:
                    if tone_rule.ctcss not in tones:
                        tones.append(tone_rule.ctcss)
                return tones

        # Then check ranges
        for frequency in self.frequencies:
            if not frequency.is_single and frequency.lo is not None and frequency.hi is not None and frequency.lo <= rf_freq <= frequency.hi:
                tones = list(frequency.ctcss_tones)
                for tone_rule in frequency.tones:
                    if tone_rule.ctcss not in tones:
                        tones.append(tone_rule.ctcss)
                return tones

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
                if freq_entry.single is not None and abs(freq_entry.single - rf) < FREQ_MATCH_TOLERANCE_MHZ:
                    if ctcss is not None:
                        for tone_rule in freq_entry.tones:
                            if abs(tone_rule.ctcss - ctcss) < TONE_MATCH_TOLERANCE_HZ and tone_rule.label:
                                return tone_rule.label
                    return freq_entry.label
            else:
                if freq_entry.lo is not None and freq_entry.hi is not None and freq_entry.lo <= rf <= freq_entry.hi:
                    range_label = freq_entry.label
                    if ctcss is not None:
                        for tone_rule in freq_entry.tones:
                            if abs(tone_rule.ctcss - ctcss) < TONE_MATCH_TOLERANCE_HZ and tone_rule.label:
                                range_label = tone_rule.label

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
