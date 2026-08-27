import logging
from pathlib import Path

import pytest
from frequency_manager import (
    ChannelMessage,
    ConfigFrequency,
    FrequencyConfiguration,
    FrequencyInfo,
    FrequencyManager,
    ToneRule,
    TransmissionRecord,
)

TEST_DIR = Path(__file__).parent

# To enable debug log, see pytest.ini and uncomment the "log_cli = true" line

CHANNEL_SPACING = 5000
LAST_ENTRY = -1


@pytest.fixture
async def fm_empty() -> FrequencyManager:

    config = FrequencyConfiguration(
        file_name=None, disable_lockout=False, disable_priority=False, max_ctcss_tones=3)
    channel_spacing = CHANNEL_SPACING

    frequency_manager = FrequencyManager(config, channel_spacing)

    # await frequency_manager.load()

    return frequency_manager


@pytest.fixture
async def fm_with_entries() -> FrequencyManager:

    config = FrequencyConfiguration(file_name=TEST_DIR / "frequency_config_for_testing.yaml", disable_lockout=False, disable_priority=False, max_ctcss_tones=3)
    channel_spacing = CHANNEL_SPACING

    frequency_manager = FrequencyManager(config, channel_spacing)

    # await frequency_manager.load()

    return frequency_manager


async def load_inline_config(frequency_manager: FrequencyManager, *entries: dict):
    """
    Feed frequency entries through the same process_frequencies_data() path used
    for real config files, without touching disk. Each positional dict is exactly
    what a `- label: ...` block under `frequencies:` would parse to, so a test's
    config can be written inline in the .py in a form that reads like the YAML
    it's meant to represent, e.g.:

        await load_inline_config(fm,
            {'label': 'Security Patrol dispatch', 'single': 462.400, 'ctcss': 100.0},
            {'label': 'Security Patrol dispatch (Backup)', 'single': 462.400, 'ctcss': 67.0},
        )

    Prefer this over calling frequency_manager.add() directly when the test's
    intent is "how does a config file with these entries behave" (e.g. loading,
    merging, duplicate handling) rather than "how does add() itself behave" in
    isolation (for the latter, calling add() directly, as most tests in this file
    do, is more direct).
    """
    return await frequency_manager.process_frequencies_data({'frequencies': list(entries)})


@pytest.mark.asyncio
async def test_process_frequencies_data():
    """Test the process_frequencies_data method directly with various inputs"""
    config = FrequencyConfiguration(
        disable_lockout=False, disable_priority=False)
    frequency_manager = FrequencyManager(config, CHANNEL_SPACING)

    # Test with valid data
    frequencies = await load_inline_config(
        frequency_manager,
        {'label': 'Test', 'single': 450.0},
        {'label': 'Range', 'lo': 460.0, 'hi': 470.0},
    )
    assert len(frequencies) == 2


@pytest.mark.parametrize("file, expected_exception, message", [
    (TEST_DIR / "frequency_config_not_found.yaml",
     FileNotFoundError, 'Frequency file does not exist'),
    (TEST_DIR / "invalid_frequency_config_format.yaml",
     Exception, 'Invalid yaml frequency file'),
    (TEST_DIR / "invalid_frequency_config_value_range.yaml",
     ValueError, 'must be larger than'),
    (TEST_DIR / "invalid_frequency_config_invalid_priority.yaml",
     ValueError, 'Priority must be an integer >= 1'),
    (TEST_DIR / "invalid_frequency_config_invalid_lockout.yaml",
     ValueError, 'Locked must be a boolean'),
    (TEST_DIR / "invalid_frequency_config_value_float_in_range.yaml",
     ValueError, 'frequency must be a float'),
    (TEST_DIR / "invalid_frequency_config_value_float_in_single.yaml",
     ValueError, 'frequency must be a float'),
    (TEST_DIR / "invalid_frequency_config_no_frequency.yaml",
     ValueError, 'Frequency must be specified as single or range'),
])
async def test_file_format_conditions(file, expected_exception, message):
    config = FrequencyConfiguration(file_name=file, disable_lockout=False, disable_priority=False)
    with pytest.raises(expected_exception, match=message):
        await FrequencyManager(config, CHANNEL_SPACING).load()


@pytest.mark.asyncio
async def test_file_load_no_errors(fm_with_entries):
    await fm_with_entries.load()
    initial_len = len(fm_with_entries.frequencies)
    # Loading twice should not raise any duplicate/already exists errors
    # and should be idempotent (i.e. length shouldn't change / no duplicates added)
    await fm_with_entries.load()
    assert len(fm_with_entries.frequencies) == initial_len


@pytest.mark.asyncio
async def test_check_existing_frequency_was_loaded(fm_with_entries):

    await fm_with_entries.load()

    frequencies = fm_with_entries.frequencies

    assert frequencies[2].single == 460.15


@pytest.mark.asyncio
async def test_check_existing_range_was_loaded(fm_with_entries):
    await fm_with_entries.load()

    frequencies = fm_with_entries.frequencies

    assert frequencies[1].lo == 450.0
    assert frequencies[1].hi == 470.0
    assert frequencies[1].label == 'A frequency range'
    assert frequencies[1].locked == True
    assert frequencies[1].priority == 2
    assert frequencies[1].saved == True


@pytest.mark.asyncio
async def test_add_single_frequency(fm_empty):

    FREQ = 500.0

    entry = {
        'single': FREQ,
        'label': 'Test frequency',
        'locked': True,
        'priority': 1
    }

    frequencies = await fm_empty.add(entry)
    # frequencies = await frequency_manager.add(FREQ, {'label': 'Test frequency', 'locked': True, 'priority': 1})

    added = frequencies[-1]

    assert added.single == FREQ
    assert added.label == 'Test frequency'
    assert added.locked == True
    assert added.saved == False
    assert added.priority == 1


@pytest.mark.asyncio
async def test_fail_negative_frequency(fm_empty):

    FREQ = -1.0

    entry = {
        'single': FREQ,
        'label': 'Test Frequency',
    }

    with pytest.raises(ValueError, match='Frequencies must be positive numbers'):
        await fm_empty.add(entry)


@pytest.mark.asyncio
async def test_fail_negative_frequency_in_range(fm_empty):

    FREQ = -1.0

    entry = {
        'lo': FREQ, 'hi': FREQ+1,
        'label': 'Test range',
    }

    with pytest.raises(ValueError, match='Frequencies must be positive numbers'):
        await fm_empty.add(entry)


@pytest.mark.asyncio
async def test_fail_invalid_key_in_frequency(fm_empty):

    FREQ = -1.0

    entry = {
        'lo': FREQ, 'blah': FREQ+1,
        'label': 'Test range',
    }

    with pytest.raises(TypeError, match='got an unexpected keyword argument'):
        await fm_empty.add(entry)


@pytest.mark.asyncio
async def test_fail_frequency_not_specified(fm_empty):

    entry = {
        'label': 'Test range',
    }

    with pytest.raises(ValueError, match='Frequency must be specified as single or range'):
        await fm_empty.add(entry)


@pytest.mark.asyncio
async def test_fail_frequency_both_single_and_range(fm_empty):

    entry = {
        'single': 500.0,
        'lo': 450.0, 'hi': 470.0,
        'label': 'Test range',
    }

    with pytest.raises(ValueError, match='Frequency cannot be specified as both single and range'):
        await fm_empty.add(entry)


@pytest.mark.asyncio
async def test_fail_frequency_partial_range(fm_empty):

    entry = {
        'hi': 470.0,
        'label': 'Test range',
    }

    with pytest.raises(ValueError, match='Both lo and hi must be specified for a frequency range'):
        await fm_empty.add(entry)


@pytest.mark.asyncio
async def test_fail_duplicate_add_single_frequency(fm_empty):

    FREQ = 500.0

    entry = {
        'single': FREQ,
        'label': 'Test frequency',
        'locked': True,
        'priority': 1
    }

    await fm_empty.add(entry)

    with pytest.raises(ValueError, match='already occurs in list'):
        await fm_empty.add(entry)


"""
Multiple CTCSS tones per frequency

Real-world config sometimes needs more than one CTCSS tone valid for the same
physical frequency (e.g. a repeater that answers to a primary and a backup PL
tone). Rather than requiring a new YAML shape, this is expressed by declaring
the same frequency/range twice, differing only by `ctcss` -- mirroring how
users already tend to write these configs:

    - label: "Security Patrol dispatch"
      single: 462.400
      priority: 1
      ctcss: 100.0
    - label: "Security Patrol dispatch (Backup)"
      single: 462.400
      priority: 1
      ctcss: 67.0
"""



@pytest.mark.asyncio
async def test_fail_duplicate_frequency(fm_empty):
    """Declaring the same frequency twice is always an error. Use tones: [...] for multiple tones."""

    FREQ = 462.400

    entry = {'label': 'Security Patrol dispatch', 'single': FREQ, 'ctcss': 100.0}
    await fm_empty.add(entry)

    # Same entry exactly → error
    with pytest.raises(ValueError, match='already occurs in list'):
        await fm_empty.add(entry)

    # Same frequency, different ctcss → also an error now
    with pytest.raises(ValueError, match='already occurs in list'):
        await fm_empty.add({'label': 'Backup', 'single': FREQ, 'ctcss': 67.0})


@pytest.mark.asyncio
async def test_fail_duplicate_range(fm_empty):
    """Declaring the same range twice is always an error."""

    await fm_empty.add({'label': 'Primary', 'lo': 450.0, 'hi': 451.0, 'ctcss': 100.0})

    with pytest.raises(ValueError, match='already occurs in list'):
        await fm_empty.add({'label': 'Backup', 'lo': 450.0, 'hi': 451.0, 'ctcss': 67.0})


@pytest.mark.asyncio
async def test_get_ctcss_tones_returns_all_configured_tones(fm_empty):

    FREQ = 462.400

    await fm_empty.add({
        'single': FREQ,
        'label': 'Security Patrol',
        'tones': [
            {'ctcss': 100.0, 'label': 'Primary'},
            {'ctcss': 67.0,  'label': 'Backup'},
        ]
    })

    fm_empty.set_center(FREQ*1e6)

    assert fm_empty.get_ctcss_tones(FREQ) == [100.0, 67.0]



@pytest.mark.asyncio
async def test_get_ctcss_tones_returns_tones_from_tones_array(fm_empty):
    """get_ctcss_tones must return CTCSS tones specified under the unified tones: [...] array."""
    await fm_empty.add({
        'single': 467.7125,
        'label': 'Base',
        'banks': ['FRS_FAMILY'],
        'tones': [
            {'ctcss': 67.0, 'label': 'Security', 'banks': ['SECURITY']},
            {'ctcss': 71.9, 'label': 'Operations', 'banks': ['OPERATIONS']},
        ]
    })
    tones = fm_empty.get_ctcss_tones(467.7125)
    assert set(tones) == {67.0, 71.9}


@pytest.mark.asyncio
async def test_get_ctcss_tones_empty_when_not_configured(fm_empty):

    await fm_empty.add({'label': 'No tone', 'single': 462.400})

    assert fm_empty.get_ctcss_tones(462.400) == []
    assert fm_empty.get_ctcss_tones(999.0) == []


@pytest.mark.asyncio
async def test_get_ctcss_info_still_returns_only_primary_tone(fm_empty):
    """
    get_ctcss_info() is kept single-valued on purpose for callers (e.g. the
    current single-tone squelch) that don't yet support multiple tones.
    """
    FREQ = 462.400

    await fm_empty.add({
        'single': FREQ,
        'label': 'Security Patrol',
        'ctcss': 100.0,
        'tones': [
            {'ctcss': 100.0, 'label': 'Primary'},
            {'ctcss': 67.0,  'label': 'Backup'},
        ]
    })

    assert fm_empty.get_ctcss_info(FREQ) == 100.0




@pytest.mark.asyncio
async def test_fail_priority_must_be_integer(fm_empty):

    FREQ = 500.0

    entry = {
        'single': FREQ,
        'label': 'Test frequency',
        'priority': "D"
    }

    with pytest.raises(ValueError, match='Priority must be an integer >= 1'):
        await fm_empty.add(entry)


@pytest.mark.asyncio
async def test_fail_priority_must_be_at_least_1(fm_empty):

    FREQ = 500.0

    entry = {
        'single': FREQ,
        'label': 'Test frequency',
        'priority': 0
    }

    with pytest.raises(ValueError, match='Priority must be an integer >= 1'):
        await fm_empty.add(entry)


@pytest.mark.asyncio
async def test_add_range_frequency(fm_empty):

    frequencies = await fm_empty.add({'lo': 200.0, 'hi': 300.0, 'label': 'Test range', 'locked': True, 'priority': 1})

    added = frequencies[LAST_ENTRY]

    assert added.lo == 200
    assert added.hi == 300
    assert added.label == 'Test range'
    assert added.locked == True
    assert added.saved == False
    assert added.priority == 1


@pytest.mark.asyncio
async def test_fail_add_duplicate_range(fm_empty):

    FREQ = 500.0

    entry = {
        'lo': FREQ, 'hi': FREQ+1,
        'label': 'Test range',
        'locked': True,
        'priority': 1
    }

    await fm_empty.add(entry)

    with pytest.raises(ValueError, match='already occurs in list'):
        await fm_empty.add(entry)


@pytest.mark.asyncio
async def test_set_center_frequency_before_adding(fm_empty):
    """
    Set the center frequency before adding in any frequencies
    """

    FREQ = 500.0

    entry = {
        'single': FREQ,
        'label': 'Test frequency',
    }

    # Set center frequency
    frequencies = fm_empty.set_center(FREQ*1e6)  # 500 MHz in Hz

    # Add a frequency
    await fm_empty.add(entry)

    # Verify baseband frequencies were generated
    assert frequencies[-1].bb_single is not None

    # The frequency we added should be at 0 Hz baseband (since it matches center freq)
    assert frequencies[-1].bb_single == 0


@pytest.mark.asyncio
async def test_set_center_frequency_after_adding(fm_empty):

    FREQ = 500.0

    entry = {
        'single': FREQ,
        'label': 'Single frequency',
    }

    # Add a single frequency before setting center frequency
    await fm_empty.add(entry)

    # Set center frequency to generate baseband frequencies
    fm_empty.set_center(FREQ*1e6)  # 500 MHz

    # Check that baseband frequencies were generated correctly
    assert isinstance(fm_empty.frequencies[-1].bb_single, int)
    # Should be at baseband center
    assert fm_empty.frequencies[LAST_ENTRY].bb_single == 0

    # Check a frequency that's offset from center
    await fm_empty.add({'single': 501.0, 'label': 'Offset frequency'})
    assert fm_empty.frequencies[LAST_ENTRY].bb_single == 1e6


@pytest.mark.asyncio
async def test_locked_out(fm_empty):

    FREQ = 500.0

    entry = {
        'single': FREQ,
        'label': 'Locked frequency',
        'locked': True
    }

    # Add a locked frequency
    await fm_empty.add(entry)

    # Set center frequency to generate baseband frequencies
    fm_empty.set_center(FREQ*1e6)

    # Check if the frequency is locked out
    assert fm_empty.locked_out(0) == True  # 0 Hz baseband = 500 MHz
    # Different frequency, not locked
    assert fm_empty.locked_out(CHANNEL_SPACING) == False


@pytest.mark.asyncio
async def test_frequency_not_locked_out(fm_empty):

    FREQ = 500.0

    entry = {
        'single': FREQ,
        'label': 'Locked frequency',
        'locked': False
    }

    # Add a non-locked frequency
    await fm_empty.add(entry)

    # Set center frequency to generate baseband frequencies
    fm_empty.set_center(FREQ*1e6)

    # Confirm the frequency is not locked out
    assert fm_empty.locked_out(0) == False  # 0 Hz baseband = 500 MHz


@pytest.mark.asyncio
async def test_locked_out_range(fm_empty):

    FREQ = 450.0

    entry = {
        'lo': FREQ, 'hi': FREQ+1,
        'label': 'Test range',
        'locked': True,
        'priority': 1
    }

    # Set center frequency to generate baseband frequencies
    fm_empty.set_center(FREQ*1e6)

    await fm_empty.add(entry)

    # Confirm that frequencies at start of range are locked out
    assert fm_empty.locked_out(0) == True  # at FREQ
    assert fm_empty.locked_out(CHANNEL_SPACING) == True

    # Confirm that frequenciesat the end of the range are locked out
    assert fm_empty.locked_out(1e6) == True  # at FREQ+1
    assert fm_empty.locked_out(1e6-CHANNEL_SPACING) == True

    # Confirm that frequencies outside the range are not locked out
    assert fm_empty.locked_out(-CHANNEL_SPACING) == False
    assert fm_empty.locked_out(1e6+CHANNEL_SPACING) == False


@pytest.mark.asyncio
async def test_is_priority(fm_empty):

    FREQ = 500.0

    # Add frequencies with different priorities
    await fm_empty.add({'single': FREQ, 'label': 'High priority', 'priority': 1})
    await fm_empty.add({'single': FREQ+1, 'label': 'Low priority', 'priority': 2})

    # Set center frequency
    fm_empty.set_center(FREQ*1e6)

    # Check priority values
    # TODO: Should is_priority return "not found" if no entry at that frequency? (e.g. 2e6)
    #       Or maybe it should log an info message?
    assert fm_empty.is_priority(0) == 1  # 500 MHz has priority 1
    assert fm_empty.is_priority(1e6) == 2  # 501 MHz has priority 2
    assert fm_empty.is_priority(
        2e6) is None  # 502 MHz has no priority


@pytest.mark.asyncio
async def test_frequency_no_priority_set(fm_empty):

    FREQ = 500.0

    entry = {
        'single': FREQ,
        'label': 'No priority',
    }

    # Add a no priority frequency
    await fm_empty.add(entry)

    # Set center frequency to generate baseband frequencies
    fm_empty.set_center(FREQ*1e6)

    # Confirm the frequency has no priority
    assert fm_empty.is_priority(0) == None  # 0 Hz baseband = 500 MHz


@pytest.mark.asyncio
async def test_is_priority_range(fm_empty):

    FREQ = 500.0

    await fm_empty.add({'lo': FREQ, 'hi': FREQ+1,
                        'label': 'High priority', 'priority': 1})

    # Set center frequency
    fm_empty.set_center(500e6)

    # Check priority values within range
    assert fm_empty.is_priority(0) == 1  # 120 MHz
    assert fm_empty.is_priority(-CHANNEL_SPACING) == None


"""
Testing precedence when frequency is also part of a range

The logic could be impacted by the order they are in the list so test both cases
"""


@pytest.mark.asyncio
async def test_individual_after_range_has_precedence(fm_empty):

    FREQ = 450.0

    range = {
        'lo': FREQ, 'hi': FREQ+1,
        'label': 'Test range',
        'priority': 1
    }

    single = {
        'single': FREQ+0.5,
        'label': 'Test frequency',
        'priority': 2
    }

    # insert the range and then the single frequency
    await fm_empty.add(range)
    await fm_empty.add(single)

    fm_empty.set_center((FREQ+0.5)*1e6)

    assert fm_empty.is_priority(0) == 2


@pytest.mark.asyncio
async def test_individual_before_range_has_precedence(fm_empty):

    FREQ = 450.0

    # insert frequency within the range
    single = {
        'single': FREQ+0.5,
        'label': 'Test frequency',
        'priority': 2
    }

    # insert range
    range = {
        'lo': FREQ, 'hi': FREQ+1,
        'label': 'Test range',
        'priority': 1
    }

    # insert the single and then the range
    await fm_empty.add(single)
    await fm_empty.add(range)

    fm_empty.set_center((FREQ+0.5)*1e6)

    assert fm_empty.is_priority(0) == 2


@pytest.mark.asyncio
async def test_return_highest_priority_range(fm_empty):
    """
    If a frequency is part of more than one range, the highest priority one should be used.
    """

    FREQ = 450.0

    # frequency within the ranges
    single = {
        'single': FREQ,
        'label': 'Test frequency',
    }

    # first range
    range1 = {
        'lo': FREQ, 'hi': FREQ+1,
        'label': 'Test range 1',
        'priority': 1
    }

    # first range
    range2 = {
        'lo': FREQ, 'hi': FREQ+2,
        'label': 'Test range 2',
        'priority': 2
    }

    # insert the single and then the range
    await fm_empty.add(single)
    await fm_empty.add(range1)
    await fm_empty.add(range2)

    fm_empty.set_center((FREQ)*1e6)

    assert fm_empty.is_priority(0) == 1


@pytest.mark.asyncio
async def test_is_higher_priority(fm_empty):

    FREQ = 500.0

    # Add frequencies with different priorities
    await fm_empty.add({'single': FREQ, 'label': 'High priority', 'priority': 1})
    await fm_empty.add({'single': FREQ+2, 'label': 'Low priority', 'priority': 2})

    # Set center frequency
    # center halfway between the 2 frequencies
    fm_empty.set_center(501e6)

    # Test priority comparisons
    # Priority 1 > Priority 2
    assert fm_empty.is_higher_priority(-1e6, 1e6) == True
    assert fm_empty.is_higher_priority(
        1e6, -1e6) == False  # Priority 2 < Priority 1
    assert fm_empty.is_higher_priority(
        1e6, 1e6) == False  # Same priority
    # Priority 1 > No priority
    assert fm_empty.is_higher_priority(-1e6, 4e6) == True
    assert fm_empty.is_higher_priority(
        4e6, -1e6) == False  # No priority < Priority 1

    # Test with demod_freq = 0 (special case)
    assert fm_empty.is_higher_priority(
        6e6, 0) == True  # Always true when demod_freq is 0


@pytest.mark.asyncio
async def test_priority_disabled():
    '''
    When using a frequency config and priority checking is disabled, check that priorities
    are ignored.
    '''

    FREQ = 500.0

    config = FrequencyConfiguration(
        disable_lockout=False, disable_priority=True)

    frequency_manager = FrequencyManager(config, CHANNEL_SPACING)

    await frequency_manager.load()

    # Add a priority frequency
    await frequency_manager.add({'single': FREQ, 'label': 'High priority', 'priority': 1})
    frequency_manager.set_center(FREQ*1e6)

    assert frequency_manager.is_higher_priority(0, 10) == False


@pytest.mark.asyncio
async def test_lockout_disabled():
    '''
    When using a frequency config and lockout checking is disabled, check that lockouts
    are ignored.
    '''
    FREQ = 500.0

    config = FrequencyConfiguration(
        disable_lockout=True, disable_priority=False)

    frequency_manager = FrequencyManager(config, CHANNEL_SPACING)

    await frequency_manager.load()
    frequency_manager.set_center(FREQ*1e6)

    # Add a locked out frequency
    await frequency_manager.add({'single': FREQ, 'label': 'Locked out', 'locked': True})

    assert frequency_manager.locked_out(0) == False


@pytest.mark.asyncio
async def test_get_label(fm_empty):

    FREQ = 500.0

    # Add a frequency with a label
    await fm_empty.add({'single': FREQ, 'label': 'Test label'})

    # Get the label
    label = fm_empty.get_label(FREQ)

    assert label == 'Test label'
    assert fm_empty.get_label(
        FREQ+1) is None  # No label for this frequency


@pytest.mark.asyncio
async def test_get_range_label(fm_empty):

    FREQ = 450.0

    # insert frequency within the range
    single = {
        'single': FREQ+0.5,
        'label': 'Specific label',
    }

    # insert range
    range = {
        'lo': FREQ, 'hi': FREQ+1,
        'label': 'General label',
    }

    await fm_empty.add(range)

    # Get labels for frequencies in the range
    assert fm_empty.get_label(FREQ+.5) == 'General label'

    # Add a specific frequency within the range
    await fm_empty.add(single)

    # The specific label should take precedence over the range label
    assert fm_empty.get_label(FREQ+0.5) == 'Specific label'

    assert fm_empty.get_label(
        FREQ) == 'General label'  # Still uses range label


@pytest.mark.asyncio
async def test_change_existing_frequency(fm_empty):

    FREQ = 500.0

    # Add a frequency
    await fm_empty.add({'single': FREQ, 'label': 'Original label', 'locked': True})

    # Modify the same frequency
    frequencies = await fm_empty.change({'single': FREQ, 'label': 'Modified label'})

    added = frequencies[LAST_ENTRY]

    assert added is not None
    assert added.label == 'Modified label'
    assert added.locked == True
    assert added.saved == False


@pytest.mark.asyncio
async def test_change_existing_range(fm_empty):

    FREQ = 500.0

    entry = {
        'lo': FREQ, 'hi': FREQ+1,
        'label': 'Test range',
        'locked': True,
        'priority': 1
    }

    # Add a frequency
    await fm_empty.add(entry)

    # Modify the same frequency
    frequencies = await fm_empty.change({'lo': entry['lo'], 'hi': entry['hi'], 'label': 'Modified label'})

    added = frequencies[LAST_ENTRY]

    assert added is not None
    assert added.label == 'Modified label'
    assert added.locked == True
    assert added.saved == False


@pytest.mark.asyncio
async def test_change_add_mode(fm_empty):
    """
    Test that changing a frequency in add mode works.  Add mode is when
    frequency does not exist but instead of error, it is added.
    """
    FREQ = 499.0

    # Modify the frequency that does not exist
    frequencies = await fm_empty.change({'single': FREQ, 'label': 'Frequency was added', 'locked': True, 'mode': 'add'})

    added = frequencies[LAST_ENTRY]

    assert added is not None
    assert added.label == 'Frequency was added'
    assert added.locked == True
    assert added.saved == False


@pytest.mark.asyncio
async def test_fail_change_nonexistant_single_frequency(fm_empty):

    FREQ = 499.0

    entry = {
        'single': FREQ,
        'label': 'Changing nonexistant frequency',
        'locked': True,
        'priority': 1
    }

    with pytest.raises(ValueError, match='not found in frequencies list'):
        await fm_empty.change(entry)


@pytest.mark.asyncio
async def test_fail_change_nonexistant_range(fm_empty):

    FREQ = 499.0

    entry = {
        'lo': FREQ, 'hi': FREQ+1,
        'label': 'Changing nonexistant frequency',
        'locked': True,
        'priority': 1
    }

    with pytest.raises(ValueError, match='not found in frequencies list'):
        await fm_empty.change(entry)


@pytest.mark.asyncio
async def test_get_priority_info(fm_empty):
    CENTER_FREQ = 500e6  # 500 MHz
    fm_empty.set_center(CENTER_FREQ)

    # 1. Add a single frequency with priority 1, mode='add' (auto-priority)
    await fm_empty.add({'single': 500.01, 'priority': 1, 'label': 'Auto Priority Single', 'mode': 'add'})

    # 2. Add a single frequency with priority 1, normal (not auto)
    await fm_empty.add({'single': 500.02, 'priority': 1, 'label': 'Priority Single'})

    # 3. Add a range frequency with priority 3
    await fm_empty.add({'lo': 500.03, 'hi': 500.05, 'priority': 3, 'label': 'Range Priority'})

    # Test single auto-priority: +10 kHz (10000 Hz)
    p, is_auto = fm_empty.get_priority_info(10000)
    assert p == 1
    assert is_auto is True

    # Test single normal priority: +20 kHz (20000 Hz)
    p, is_auto = fm_empty.get_priority_info(20000)
    assert p == 1
    assert is_auto is False

    # Test range priority: +40 kHz (40000 Hz)
    p, is_auto = fm_empty.get_priority_info(40000)
    assert p == 3
    assert is_auto is False

    # Test non-priority channel: +60 kHz (60000 Hz)
    p, is_auto = fm_empty.get_priority_info(60000)
    assert p is None
    assert is_auto is False


@pytest.mark.asyncio
async def test_max_ctcss_tones_disabled(fm_empty):
    """max_ctcss_tones=0 disables CTCSS entirely: adding a frequency with ctcss: raises."""
    from frequency_manager import FrequencyConfiguration, FrequencyManager

    config_0 = FrequencyConfiguration(
        file_name=None,
        disable_lockout=False,
        disable_priority=False,
        max_ctcss_tones=0
    )
    fm_0 = FrequencyManager(config_0, channel_spacing=5000)

    with pytest.raises(ValueError, match="CTCSS is disabled"):
        await fm_0.add({'single': 144.390, 'ctcss': 100.0, 'label': 'Frequency'})


@pytest.mark.asyncio
async def test_max_ctcss_tones_bypass_via_tones_raises_when_exceeding(fm_empty):
    """max_ctcss_tones must cap tones: -only entries too (not just scalar ctcss:).

    Regression guard: the count check previously keyed off wanted.ctcss, so a
    tones: array with more tones than the receiver supports slipped past config
    validation and was only truncated silently at runtime.
    """
    with pytest.raises(ValueError, match="max_ctcss_tones"):
        await fm_empty.add({
            'single': 467.7125,
            'label': 'FRS Family',
            'tones': [
                {'ctcss': 67.0, 'label': 'Tone 1'},
                {'ctcss': 71.9, 'label': 'Tone 2'},
                {'ctcss': 74.4, 'label': 'Tone 3'},
                {'ctcss': 77.0, 'label': 'Tone 4'},
            ],
        })


@pytest.mark.asyncio
async def test_max_ctcss_tones_disabled_rejects_tones_only_entry():
    """max_ctcss_tones=0 disables CTCSS even for tones: -only entries."""
    from frequency_manager import FrequencyConfiguration, FrequencyManager

    config_0 = FrequencyConfiguration(
        file_name=None,
        disable_lockout=False,
        disable_priority=False,
        max_ctcss_tones=0
    )
    fm_0 = FrequencyManager(config_0, channel_spacing=5000)

    with pytest.raises(ValueError, match="CTCSS is disabled"):
        await fm_0.add({
            'single': 144.390,
            'label': 'Frequency',
            'tones': [
                {'ctcss': 100.0, 'label': 'Primary'},
                {'ctcss': 141.3, 'label': 'Backup'},
            ],
        })


@pytest.mark.asyncio
async def test_change_does_not_merge_ctcss(fm_empty):
    """change() no longer merges CTCSS tones; ctcss key in entry is ignored for tone accumulation."""
    await fm_empty.add({'single': 144.390, 'ctcss': 100.0, 'label': 'Frequency'})
    assert fm_empty.frequencies[0].ctcss_tones == [100.0]

    # change() with a ctcss key should not accumulate tones
    await fm_empty.change({'single': 144.390, 'ctcss': 141.3})
    assert fm_empty.frequencies[0].ctcss_tones == [100.0]


@pytest.mark.asyncio
async def test_get_label_with_ctcss(fm_empty):
    """Test that get_label correctly resolves the label matching a specific CTCSS tone."""
    await fm_empty.add({
        'single': 144.390,
        'label': 'Tone 100',
        'ctcss': 100.0,
        'tones': [
            {'ctcss': 100.0, 'label': 'Tone 100'},
            {'ctcss': 141.3, 'label': 'Tone 141'},
            {'ctcss': 151.4, 'label': 'Tone 151'},
        ]
    })

    # get_label without ctcss returns the primary label
    assert fm_empty.get_label(144.390) == 'Tone 100'

    # get_label with specific ctcss returns the corresponding label
    assert fm_empty.get_label(144.390, 100.0) == 'Tone 100'
    assert fm_empty.get_label(144.390, 141.3) == 'Tone 141'
    assert fm_empty.get_label(144.390, 151.4) == 'Tone 151'

    # get_label with unmatched ctcss falls back to primary label
    assert fm_empty.get_label(144.390, 88.5) == 'Tone 100'


def test_tone_rule_validation():
    """Test ToneRule dataclass construction and validation."""
    rule = ToneRule(ctcss=67.0, label="Fire Tac", banks="FIRE_TAC")
    assert rule.ctcss == 67.0
    assert rule.label == "Fire Tac"
    assert rule.banks == ["FIRE_TAC"]

    with pytest.raises(ValueError, match="positive number"):
        ToneRule(ctcss=-5.0)

    with pytest.raises(ValueError, match="CTCSS must be a float"):
        ToneRule(ctcss="invalid")


def test_frequency_info_banks_and_tones_normalization():
    """Test FrequencyInfo/ConfigFrequency normalization for banks and tones."""
    # 1. Scalar bank string auto-promotion (FrequencyInfo base class)
    info1 = FrequencyInfo(banks="PUBLIC_SAFETY")
    assert info1.banks == ["PUBLIC_SAFETY"]

    # 2. Scalar ctcss auto-promotion to tones list (ConfigFrequency only — tones lives there)
    info2 = ConfigFrequency(single=462.400, ctcss=100.0, banks=["FIRE_TAC"])
    assert len(info2.tones) == 1
    assert info2.tones[0].ctcss == 100.0
    assert info2.tones[0].banks == ["FIRE_TAC"]

    # 3. Parent bank inheritance for tones omitting banks (ConfigFrequency)
    tone_no_bank = ToneRule(ctcss=141.3, label="Parks")
    info3 = ConfigFrequency(single=462.400, banks=["COMMERCIAL"], tones=[tone_no_bank])
    assert info3.tones[0].banks == ["COMMERCIAL"]


def test_transmission_record_banks_field():
    """Test TransmissionRecord includes banks list."""
    rec = TransmissionRecord(
        rf=460.125,
        bb_hz=0,
        channel=0,
        label="Test",
        priority=1,
        matched_ctcss_hz=67.0,
        signal_db=-40,
        classification="V",
        wav_path="/tmp/test.wav",
        started_at=1000.0,
        duration_sec=5.0,
        banks=["PUBLIC_SAFETY", "FIRE_TAC"],
    )
    assert rec.banks == ["PUBLIC_SAFETY", "FIRE_TAC"]


# ---------------------------------------------------------------------------
# Bank support: set_active_banks / is_bank_active
# ---------------------------------------------------------------------------

def test_set_active_banks_accepts_list(fm_empty):
    """set_active_banks stores the supplied list as a set."""
    fm_empty.set_active_banks(["PUBLIC_SAFETY", "RAILROAD"])
    assert fm_empty.active_banks == {"PUBLIC_SAFETY", "RAILROAD"}


def test_set_active_banks_accepts_set(fm_empty):
    fm_empty.set_active_banks({"FIRE_TAC"})
    assert fm_empty.active_banks == {"FIRE_TAC"}


def test_set_active_banks_accepts_none(fm_empty):
    """None resets to promiscuous (empty set)."""
    fm_empty.set_active_banks(["PUBLIC_SAFETY"])
    fm_empty.set_active_banks(None)
    assert fm_empty.active_banks == set()


def test_is_bank_active_empty_active_banks_is_promiscuous(fm_empty):
    """Empty active_banks means every bank matches (scan-all mode)."""
    fm_empty.set_active_banks(None)
    assert fm_empty.is_bank_active(["PUBLIC_SAFETY"]) is True
    assert fm_empty.is_bank_active([]) is True


def test_is_bank_active_intersection(fm_empty):
    fm_empty.set_active_banks(["PUBLIC_SAFETY"])
    assert fm_empty.is_bank_active(["PUBLIC_SAFETY", "LAW_ENFORCEMENT"]) is True
    assert fm_empty.is_bank_active(["RAILROAD"]) is False
    assert fm_empty.is_bank_active([]) is False


# ---------------------------------------------------------------------------
# Bank support: resolve_banks() — 5-tier precedence hierarchy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_banks_tier1_single_with_matching_tone(fm_empty):
    """Tier 1: explicit single entry + matching tone rule → tone rule's banks."""
    await fm_empty.add({
        'single': 462.5625,
        'banks': ['COMMERCIAL'],
        'tones': [
            {'ctcss': 67.0, 'label': 'Fire Tac 1', 'banks': ['FIRE_TAC']},
            {'ctcss': 141.3, 'label': 'City Parks', 'banks': ['PARKS_MAINT']},
        ],
    })

    result = fm_empty.resolve_banks(462.5625, ctcss_hz=67.0)
    assert result == ['FIRE_TAC']

    result = fm_empty.resolve_banks(462.5625, ctcss_hz=141.3)
    assert result == ['PARKS_MAINT']


@pytest.mark.asyncio
async def test_resolve_banks_tier1_tone_inherits_parent_bank_when_tone_has_no_banks(fm_empty):
    """Tier 1: tone rule without its own banks inherits the parent entry's banks."""
    await fm_empty.add({
        'single': 462.5625,
        'banks': ['COMMERCIAL'],
        'tones': [
            {'ctcss': 141.3, 'label': 'City Parks'},   # no banks key → should inherit COMMERCIAL
        ],
    })

    result = fm_empty.resolve_banks(462.5625, ctcss_hz=141.3)
    assert result == ['COMMERCIAL']


@pytest.mark.asyncio
async def test_resolve_banks_tier2_single_csq_no_tone(fm_empty):
    """Tier 2: explicit single entry, no decoded tone → base entry banks."""
    await fm_empty.add({
        'single': 460.125,
        'banks': ['PUBLIC_SAFETY', 'LAW_ENFORCEMENT'],
    })

    result = fm_empty.resolve_banks(460.125, ctcss_hz=None)
    assert set(result) == {'PUBLIC_SAFETY', 'LAW_ENFORCEMENT'}


@pytest.mark.asyncio
async def test_resolve_banks_tier2_single_with_tone_no_ctcss_decoded(fm_empty):
    """Tier 2: single entry has tone rules, but no CTCSS was decoded → union of base and tone banks returned."""
    await fm_empty.add({
        'single': 462.5625,
        'banks': ['COMMERCIAL'],
        'tones': [
            {'ctcss': 67.0, 'banks': ['FIRE_TAC']},
        ],
    })

    # No ctcss_hz decoded → returns Union of base and tone rules
    result = fm_empty.resolve_banks(462.5625, ctcss_hz=None)
    assert set(result) == {'COMMERCIAL', 'FIRE_TAC'}


@pytest.mark.asyncio
async def test_resolve_banks_tier2_single_untagged_fallback(fm_empty):
    """Tier 2: single entry exists but has no banks → no tags (promiscuous)."""
    await fm_empty.add({'single': 460.050, 'label': 'Untagged channel'})

    result = fm_empty.resolve_banks(460.050, ctcss_hz=None)
    assert result == []


@pytest.mark.asyncio
async def test_resolve_banks_tier3_range_with_matching_tone(fm_empty):
    """Tier 3: frequency inside a range + matching tone rule → tone rule's banks."""
    await fm_empty.add({
        'lo': 462.200,
        'hi': 462.400,
        'banks': ['COMMERCIAL'],
        'tones': [
            {'ctcss': 67.0, 'label': 'Fire Tac Segment', 'banks': ['FIRE_TAC']},
        ],
    })

    result = fm_empty.resolve_banks(462.300, ctcss_hz=67.0)
    assert result == ['FIRE_TAC']


@pytest.mark.asyncio
async def test_resolve_banks_tier4_range_csq(fm_empty):
    """Tier 4: frequency inside a range, no tone decoded → range's base banks."""
    await fm_empty.add({
        'lo': 462.200,
        'hi': 462.400,
        'banks': ['COMMERCIAL'],
    })

    result = fm_empty.resolve_banks(462.300, ctcss_hz=None)
    assert result == ['COMMERCIAL']


@pytest.mark.asyncio
async def test_resolve_banks_tier4_range_untagged_fallback(fm_empty):
    """Tier 4: frequency inside an untagged range → no tags (promiscuous)."""
    await fm_empty.add({'lo': 462.200, 'hi': 462.400, 'label': 'Untagged range'})

    result = fm_empty.resolve_banks(462.300)
    assert result == []


@pytest.mark.asyncio
async def test_resolve_banks_tier5_unconfigured_returns_no_tags_in_promiscuous(fm_empty):
    """Tier 5: frequency outside all configured entries → no tags (no SEARCH,
    no bank filtering active)."""
    await fm_empty.add({'single': 460.125, 'banks': ['PUBLIC_SAFETY']})

    result = fm_empty.resolve_banks(999.999)
    assert result == []


@pytest.mark.asyncio
async def test_resolve_banks_tier5_unconfigured_returns_search_when_active(fm_empty):
    """Tier 5: SEARCH active + unconfigured frequency → ['SEARCH'] returned."""
    await fm_empty.add({'single': 460.125, 'banks': ['PUBLIC_SAFETY']})
    fm_empty.set_active_banks(['SEARCH'])

    result = fm_empty.resolve_banks(999.999)
    assert result == ['SEARCH']


@pytest.mark.asyncio
async def test_resolve_banks_single_takes_precedence_over_range(fm_empty):
    """Single entry (Tier 1/2) always beats a containing range (Tier 3/4)."""
    await fm_empty.add({'lo': 460.000, 'hi': 461.000, 'banks': ['RAILROAD']})
    await fm_empty.add({'single': 460.500, 'banks': ['PUBLIC_SAFETY']})

    result = fm_empty.resolve_banks(460.500)
    assert result == ['PUBLIC_SAFETY']


@pytest.mark.asyncio
async def test_resolve_banks_tone_match_tolerance(fm_empty):
    """Tone matching uses a ±0.5 Hz tolerance; exact and near-exact both hit; far miss falls back."""
    await fm_empty.add({
        'single': 462.5625,
        'banks': ['COMMERCIAL'],
        'tones': [{'ctcss': 67.0, 'banks': ['FIRE_TAC']}],
    })

    # Exact match
    assert fm_empty.resolve_banks(462.5625, ctcss_hz=67.0) == ['FIRE_TAC']
    # Within tolerance
    assert fm_empty.resolve_banks(462.5625, ctcss_hz=67.4) == ['FIRE_TAC']
    # Outside tolerance → falls back to base entry banks (Tier 2)
    assert fm_empty.resolve_banks(462.5625, ctcss_hz=68.0) == ['COMMERCIAL']


# ---------------------------------------------------------------------------
# Bank support: "SEARCH" dynamic tag — opt-in unconfigured spectrum scanning
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_tag_in_active_banks_resolves_unconfigured_hits(fm_empty):
    """SEARCH in active_banks causes unconfigured spectrum hits to resolve as ['SEARCH']."""
    await fm_empty.add({'single': 460.125, 'banks': ['PUBLIC_SAFETY']})
    fm_empty.set_active_banks(['PUBLIC_SAFETY', 'SEARCH'])

    # Configured hit still resolves normally
    assert fm_empty.resolve_banks(460.125) == ['PUBLIC_SAFETY']

    # Unconfigured hit resolves as SEARCH (not UNTAGGED)
    assert fm_empty.resolve_banks(999.000) == ['SEARCH']


@pytest.mark.asyncio
async def test_search_tag_not_active_unconfigured_returns_untagged(fm_empty):
    """Without SEARCH in active_banks, unconfigured spectrum hits return UNTAGGED."""
    await fm_empty.add({'single': 460.125, 'banks': ['PUBLIC_SAFETY']})
    fm_empty.set_active_banks(['PUBLIC_SAFETY'])

    assert fm_empty.resolve_banks(999.000) == ['UNTAGGED']


@pytest.mark.asyncio
async def test_search_tag_alone_in_active_banks(fm_empty):
    """SEARCH as the only active bank: resolve_banks still returns SEARCH for unconfigured
    spectrum and resolves configured entries normally."""
    await fm_empty.add({'single': 460.125, 'banks': ['PUBLIC_SAFETY']})
    fm_empty.set_active_banks(['SEARCH'])

    # resolve_banks: unconfigured hit → SEARCH
    assert fm_empty.resolve_banks(999.000) == ['SEARCH']

    # resolve_banks: configured single still resolves correctly (SEARCH not in active_banks does
    # not suppress configured resolution; it only affects Tier 5 unconfigured hits)
    assert fm_empty.resolve_banks(460.125) == ['PUBLIC_SAFETY']


@pytest.mark.asyncio
async def test_search_tag_promiscuous_mode_search_inactive(fm_empty):
    """In promiscuous mode (empty active_banks), SEARCH is inactive and
    unconfigured spectrum returns no tags."""
    await fm_empty.add({'single': 460.125, 'banks': ['PUBLIC_SAFETY']})
    fm_empty.set_active_banks(None)  # promiscuous

    assert fm_empty.resolve_banks(999.000) == []


@pytest.mark.asyncio
async def test_resolve_banks_untagged_sentinel_gated_on_bank_filtering(fm_empty):
    """The UNTAGGED sentinel appears only when bank filtering is active.

    Without --banks (empty active_banks), untagged hits resolve to no tags so
    bank metadata stays empty for non-participants.  With --banks active, the
    same hits resolve to UNTAGGED so is_bank_active() can fail them closed.
    """
    await fm_empty.add({'single': 460.125, 'label': 'Untagged channel'})    # Promiscuous / legacy: no sentinel injected
    assert fm_empty.resolve_banks(460.125) == []
    assert fm_empty.resolve_banks(999.000) == []

    # Bank filtering active: untagged hits resolve to the sentinel
    fm_empty.set_active_banks(['PUBLIC_SAFETY'])
    assert fm_empty.resolve_banks(460.125) == ['UNTAGGED']
    assert fm_empty.resolve_banks(999.000) == ['UNTAGGED']

    # Untagged sentinel never matches a non-dynamic active bank (fail-closed)
    assert fm_empty.is_bank_active(['UNTAGGED']) is False


def test_channel_message_str_omits_bank_bracket_when_no_tags() -> None:
    """The syslog debug line (str(msg)) must skip the bank field when no bank
    information is available instead of printing [UNTAGGED]."""
    no_banks = ChannelMessage(state='off', rf=460.125, bb=0, channel=3)
    text = str(no_banks)
    assert '[UNTAGGED]' not in text
    assert '[PUBLIC_SAFETY]' not in text

    tagged = ChannelMessage(state='off', rf=460.125, bb=0, channel=3,
                            banks=['PUBLIC_SAFETY'])
    assert '[PUBLIC_SAFETY]' in str(tagged)


# ---------------------------------------------------------------------------
# Mutation-killing boundary tests (plugging gaps found by mutmut)
# ---------------------------------------------------------------------------

def test_set_active_banks_accepts_bare_string(fm_empty):
    """set_active_banks with a bare string wraps it in a set (isinstance str branch).
    Kills mutant that sets active_banks = None instead of {banks}.
    """
    fm_empty.set_active_banks("PUBLIC_SAFETY")
    assert fm_empty.active_banks == {"PUBLIC_SAFETY"}
    # Confirm it actually filters correctly — not promiscuous
    assert fm_empty.is_bank_active(["PUBLIC_SAFETY"]) is True
    assert fm_empty.is_bank_active(["RAILROAD"]) is False


@pytest.mark.asyncio
async def test_resolve_banks_single_proximity_threshold(fm_empty):
    """Frequency 200 Hz outside the ±100 Hz (1e-4 MHz) single-match window
    must NOT match the entry. Kills mutations that widen the threshold to
    <= 1e-4 or < 1.0001 MHz.
    """
    await fm_empty.add({'single': 460.1250, 'banks': ['PUBLIC_SAFETY']})

    # Exactly on target → must match
    assert fm_empty.resolve_banks(460.1250) == ['PUBLIC_SAFETY']
    # 200 Hz away (0.0002 MHz) → outside 100 Hz window → must NOT match
    assert fm_empty.resolve_banks(460.1252) == []
    assert fm_empty.resolve_banks(460.1248) == []


@pytest.mark.asyncio
async def test_resolve_banks_range_boundary_inclusive_at_lo_and_hi(fm_empty):
    """A frequency exactly at lo or hi must be inside the range (inclusive <=).
    Kills mutations that change lo <= to lo < or <= hi to < hi.
    """
    await fm_empty.add({'lo': 462.200, 'hi': 462.400, 'banks': ['COMMERCIAL']})

    # Exactly at lo — must be inside (<=, not <)
    assert fm_empty.resolve_banks(462.200) == ['COMMERCIAL']
    # Exactly at hi — must be inside (<= not <)
    assert fm_empty.resolve_banks(462.400) == ['COMMERCIAL']
    # Just outside both ends — must NOT match
    assert fm_empty.resolve_banks(462.199) == []
    assert fm_empty.resolve_banks(462.401) == []


@pytest.mark.asyncio
async def test_resolve_banks_single_tone_tolerance_exact_boundary(fm_empty):
    """Tone ±0.5 Hz is the exclusive upper boundary: abs(diff) must be
    strictly < 0.5 to match. Kills mutations that change < 0.5 to <= 0.5
    or < 1.5 on single-entry tone rules (Tier 1).
    """
    await fm_empty.add({
        'single': 462.5625,
        'banks': ['COMMERCIAL'],
        'tones': [{'ctcss': 67.0, 'banks': ['FIRE_TAC']}],
    })

    # 0.49 Hz away — strictly inside window → must match tone
    assert fm_empty.resolve_banks(462.5625, ctcss_hz=67.49) == ['FIRE_TAC']
    assert fm_empty.resolve_banks(462.5625, ctcss_hz=66.51) == ['FIRE_TAC']
    # Exactly 0.5 Hz away — NOT strictly < 0.5 → must NOT match, falls back to base
    assert fm_empty.resolve_banks(462.5625, ctcss_hz=67.5) == ['COMMERCIAL']
    assert fm_empty.resolve_banks(462.5625, ctcss_hz=66.5) == ['COMMERCIAL']
    # 0.9 Hz away — well outside → must NOT match (kills < 1.5 widening mutation)
    assert fm_empty.resolve_banks(462.5625, ctcss_hz=67.9) == ['COMMERCIAL']


@pytest.mark.asyncio
async def test_resolve_banks_range_tone_tolerance_exact_boundary(fm_empty):
    """Same ±0.5 Hz exclusive boundary check, but for range-entry tone rules
    (Tier 3). Kills mutations that change < 0.5 to <= 0.5 or < 1.5.
    """
    await fm_empty.add({
        'lo': 462.200,
        'hi': 462.400,
        'banks': ['COMMERCIAL'],
        'tones': [{'ctcss': 67.0, 'banks': ['FIRE_TAC']}],
    })

    # 0.49 Hz away — strictly inside window → must match tone
    assert fm_empty.resolve_banks(462.300, ctcss_hz=67.49) == ['FIRE_TAC']
    assert fm_empty.resolve_banks(462.300, ctcss_hz=66.51) == ['FIRE_TAC']
    # Exactly 0.5 Hz away — NOT strictly < 0.5 → falls back to base range banks
    assert fm_empty.resolve_banks(462.300, ctcss_hz=67.5) == ['COMMERCIAL']
    assert fm_empty.resolve_banks(462.300, ctcss_hz=66.5) == ['COMMERCIAL']
    # 0.9 Hz away — well outside → must NOT match (kills < 1.5 widening mutation)
    assert fm_empty.resolve_banks(462.300, ctcss_hz=67.9) == ['COMMERCIAL']


# ---------------------------------------------------------------------------
# get_label / get_ctcss_info match-tolerance consistency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_label_rf_match_tolerance(fm_empty):
    """get_label must match RF within the same strict <1e-4 MHz window as
    resolve_banks (not exact equality). Kills regressions to `==` matching.
    """
    await fm_empty.add({'single': 460.1250, 'label': 'Patrol'})

    # Exactly on target → matches
    assert fm_empty.get_label(460.1250) == 'Patrol'
    # 50 Hz away (0.00005 MHz) → strictly inside window → matches
    assert fm_empty.get_label(460.12505) == 'Patrol'
    # 200 Hz away (0.0002 MHz) → outside window → no match
    assert fm_empty.get_label(460.1252) is None
    assert fm_empty.get_label(460.1248) is None


@pytest.mark.asyncio
async def test_get_label_uses_tone_tolerance_for_tone_rules(fm_empty):
    """get_label tone matching uses the same ±0.5 Hz exclusive tolerance as
    resolve_banks: near tones hit, exactly-0.5-away and far tones fall back to
    the entry label. Kills regressions that return the exact-value tone label
    only (old ctcss_labels path) or widen the tolerance.
    """
    await fm_empty.add({
        'single': 462.5625,
        'label': 'Base',
        'tones': [
            {'ctcss': 67.0, 'label': 'Fire Tac'},
            {'ctcss': 100.0, 'label': 'Police'},
        ],
    })

    # 0.49 Hz away — strictly inside → matches the tone label
    assert fm_empty.get_label(462.5625, 67.49) == 'Fire Tac'
    assert fm_empty.get_label(462.5625, 99.51) == 'Police'
    # Exactly 0.5 Hz away — NOT strictly < 0.5 → falls back to entry label
    assert fm_empty.get_label(462.5625, 67.5) == 'Base'
    # Well outside → falls back to entry label
    assert fm_empty.get_label(462.5625, 68.0) == 'Base'
    # No ctcss provided → entry label
    assert fm_empty.get_label(462.5625) == 'Base'


@pytest.mark.asyncio
async def test_get_ctcss_info_tones_only_returns_first_tone(fm_empty):
    """get_ctcss_info must surface a primary tone for tones:-only entries
    (previously returned None when no scalar ctcss was set)."""
    await fm_empty.add({
        'single': 462.400,
        'label': 'Security Patrol',
        'tones': [
            {'ctcss': 100.0, 'label': 'Primary'},
            {'ctcss': 67.0, 'label': 'Backup'},
        ]
    })

    assert fm_empty.get_ctcss_info(462.400) == 100.0


@pytest.mark.asyncio
async def test_get_ctcss_info_none_when_no_tone(fm_empty):
    """get_ctcss_info returns None for entries with no CTCSS tone configured."""
    await fm_empty.add({'single': 462.400, 'label': 'No tone'})

    assert fm_empty.get_ctcss_info(462.400) is None
    assert fm_empty.get_ctcss_info(999.0) is None


# ---------------------------------------------------------------------------
# Active-bank startup sanity check (typo detection)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_active_banks_matches_configured_set(fm_empty):
    """unknown_active_banks must flag only requested banks with no configured
    match, across both top-level and tone-rule banks."""
    await fm_empty.add({
        'single': 462.5625,
        'banks': ['COMMERCIAL'],
        'tones': [
            {'ctcss': 67.0, 'banks': ['FIRE_TAC']},
            {'ctcss': 71.9, 'banks': ['SECURITY']},
        ],
    })
    await fm_empty.add({'single': 467.7125, 'banks': ['OPERATIONS']})

    fm_empty.set_active_banks(['FIRE_TAC', 'OPERATIONS', 'FIRE_TAK', 'NOPE'])

    assert fm_empty.unknown_active_banks() == {'FIRE_TAK', 'NOPE'}


def test_unknown_active_banks_exempts_search(fm_empty):
    """SEARCH is a pseudo-bank, not a configured tag — must never be flagged."""
    fm_empty.set_active_banks(['SEARCH'])
    assert fm_empty.unknown_active_banks() == set()


def test_unknown_active_banks_empty_in_promiscuous_mode(fm_empty):
    """Promiscuous mode (no active banks) means nothing to validate."""
    fm_empty.set_active_banks(None)
    assert fm_empty.unknown_active_banks() == set()


@pytest.mark.asyncio
async def test_load_warns_on_unmatched_active_banks(fm_empty, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """A typo'd active bank (vs. configured tags) must log a WARNING on load()."""
    freqs_file = tmp_path / "freqs.yaml"
    freqs_file.write_text(
        "frequencies:\n"
        "  - single: 460.125\n"
        "    label: Patrol\n"
        "    banks: [FIRE_TAC]\n"
    )

    fm_empty.config.file_name = freqs_file
    fm_empty.set_active_banks(["FIRE_TAK"])

    with caplog.at_level(logging.WARNING, logger="ham2mon.frequency_manager"):
        await fm_empty.load()

    assert any(
        "FIRE_TAK" in message and "match no configured" in message
        for message in caplog.messages
    )


@pytest.mark.asyncio
async def test_load_no_warning_when_all_active_banks_match(fm_empty, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Matching active banks produce no startup warning on load()."""
    freqs_file = tmp_path / "freqs.yaml"
    freqs_file.write_text(
        "frequencies:\n"
        "  - single: 460.125\n"
        "    label: Patrol\n"
        "    banks: [FIRE_TAC]\n"
    )

    fm_empty.config.file_name = freqs_file
    fm_empty.set_active_banks(["FIRE_TAC"])

    with caplog.at_level(logging.WARNING, logger="ham2mon.frequency_manager"):
        await fm_empty.load()

    assert not any("match no configured" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_load_warns_on_unmatched_active_banks_without_file(fm_empty, caplog: pytest.LogCaptureFixture):
    """--banks X without -F/--frequencies silently resolves everything to
    UNTAGGED, which never matches X — load() must warn at startup."""
    fm_empty.set_active_banks(["FIRE_TAC"])

    with caplog.at_level(logging.WARNING, logger="ham2mon.frequency_manager"):
        await fm_empty.load()

    assert any(
        "FIRE_TAC" in message and "match no configured" in message
        for message in caplog.messages
    )


@pytest.mark.asyncio
async def test_load_no_warning_without_file_when_search_only(fm_empty, caplog: pytest.LogCaptureFixture):
    """--banks SEARCH without a frequency file is functional (Tier 5 SEARCH
    matches), so no startup warning."""
    fm_empty.set_active_banks(["SEARCH"])

    with caplog.at_level(logging.WARNING, logger="ham2mon.frequency_manager"):
        await fm_empty.load()

    assert not any("match no configured" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_load_no_warning_without_file_when_untagged_only(fm_empty, caplog: pytest.LogCaptureFixture):
    """--banks UNTAGGED without a frequency file is functional (Tier 5 UNTAGGED
    matches), so no startup warning."""
    fm_empty.set_active_banks(["UNTAGGED"])

    with caplog.at_level(logging.WARNING, logger="ham2mon.frequency_manager"):
        await fm_empty.load()

    assert not any("match no configured" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_active_bank_without_frequencies_resolves_untagged(fm_empty):
    """Without a frequency file, every hit resolves to UNTAGGED which never
    matches a non-dynamic active bank — the fail-closed no-op that the
    load()-time warning exists to surface."""
    fm_empty.set_active_banks(["FIRE_TAC"])

    assert fm_empty.resolve_banks(460.125) == ["UNTAGGED"]
    assert fm_empty.is_bank_active(["UNTAGGED"]) is False
