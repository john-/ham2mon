import pytest
from frequency_manager import (
    FrequencyManager, FrequencyConfiguration,
)
from pathlib import Path

# To enable debug log, see pytest.ini and uncomment the "log_cli = true" line

CHANNEL_SPACING = 5000
LAST_ENTRY = -1


@pytest.fixture
async def fm_empty() -> FrequencyManager:

    config = FrequencyConfiguration(
        file_name=None, disable_lockout=False, disable_priority=False)
    channel_spacing = CHANNEL_SPACING

    frequency_manager = FrequencyManager(config, channel_spacing)

    # await frequency_manager.load()

    return frequency_manager


@pytest.fixture
async def fm_with_entries() -> FrequencyManager:

    config = FrequencyConfiguration(file_name=Path(
        "tests/frequency_config_for_testing.yaml"), disable_lockout=False, disable_priority=False)
    channel_spacing = CHANNEL_SPACING

    frequency_manager = FrequencyManager(config, channel_spacing)

    # await frequency_manager.load()

    return frequency_manager


@pytest.mark.asyncio
async def test_process_frequencies_data():
    """Test the process_frequencies_data method directly with various inputs"""
    config = FrequencyConfiguration(
        disable_lockout=False, disable_priority=False)
    frequency_manager = FrequencyManager(config, CHANNEL_SPACING)

    # Test with valid data
    valid_config = {
        'frequencies': [
            {'label': 'Test', 'single': 450.0},
            {'label': 'Range', 'lo': 460.0, 'hi': 470.0}
        ]
    }
    frequencies = await frequency_manager.process_frequencies_data(valid_config)
    assert len(frequencies) == 2


@pytest.mark.parametrize("file, expected_exception, message", [
    ("tests/frequency_config_not_found.yaml",
     FileNotFoundError, 'Frequency file does not exist'),
    ("tests/invalid_frequency_config_format.yaml",
     Exception, 'Invalid yaml frequency file'),
    ("tests/invalid_frequency_config_value_range.yaml",
     ValueError, 'must be larger than'),
    ("tests/invalid_frequency_config_invalid_priority.yaml",
     ValueError, 'Priority must be an integer >= 1'),
    ("tests/invalid_frequency_config_invalid_lockout.yaml",
     ValueError, 'Locked must be a boolean'),
    ("tests/invalid_frequency_config_value_float_in_range.yaml",
     ValueError, 'frequency must be a float'),
    ("tests/invalid_frequency_config_value_float_in_single.yaml",
     ValueError, 'frequency must be a float'),
    ("tests/invalid_frequency_config_no_frequency.yaml",
     ValueError, 'Frequency must be specified as single or range'),
])
async def test_file_format_conditions(file, expected_exception, message):
    config = FrequencyConfiguration(file_name=Path(
        file), disable_lockout=False, disable_priority=False)
    with pytest.raises(expected_exception, match=message):
        await FrequencyManager(config, CHANNEL_SPACING).load()


@pytest.mark.asyncio
async def test_file_load_no_errors(fm_with_entries):

    await fm_with_entries.load()


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
