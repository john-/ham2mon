"""Tests for Scanner and TransmissionRecord (Phase 2)."""
import os
import wave
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from conftest import make_test_scanner
from frequency_manager import ChannelFrequency, ChannelMessage, TransmissionRecord
from scanner import Scanner
from utilities import (
    DEFAULT_AUDIO_RATE,
    WAV_HEADER_BYTES,
    wav_bytes_per_sec,
    wav_duration_sec,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write_wav(path: str, num_samples: int, sample_rate: int = DEFAULT_AUDIO_RATE,
               bit_depth: int = 16) -> None:
    """Write a minimal PCM WAV file with the given number of silent samples.

    Defaults to DEFAULT_AUDIO_RATE (8000 Hz) and 16-bit depth to match the
    audio format used by ham2mon demodulators.
    """
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(bit_depth // 8)
        wf.setframerate(sample_rate)
        wf.writeframes(b'\x00' * (num_samples * (bit_depth // 8)))


# ---------------------------------------------------------------------------
# Existing Phase 1 tests
# ---------------------------------------------------------------------------

def test_scanner_get_signal_strength_offset() -> None:
    class DummyScanner:
        samp_rate: int
        spectrum: np.ndarray
        _get_signal_strength: Callable[[int], float]

    scanner = DummyScanner()
    scanner.samp_rate = 4000000
    # Bind the real _get_signal_strength method to our dummy instance
    scanner._get_signal_strength = Scanner._get_signal_strength.__get__(  # type: ignore[method-assign]
        scanner, DummyScanner
    )

    # 1. Test weak signal: linear power of 10.0 (10 dB raw power)
    # Calibrated: 10 dB - 70 dB = -60 dB
    scanner.spectrum = np.array([10.0])
    val = scanner._get_signal_strength(0)  # type: ignore[call-arg]
    assert np.isclose(val, -60.0)

    # 2. Test strong signal: linear power of 1000.0 (30 dB raw power)
    # Calibrated: 30 dB - 70 dB = -40 dB
    scanner.spectrum = np.array([1000.0])
    val = scanner._get_signal_strength(0)  # type: ignore[call-arg]
    assert np.isclose(val, -40.0)

    # 3. Test empty/None spectrum: should return -100.0
    scanner.spectrum = np.empty(0)
    val = scanner._get_signal_strength(0)  # type: ignore[call-arg]
    assert np.isclose(val, -100.0)

    # 4. Test zero/negative power: should return -200.0
    scanner.spectrum = np.array([0.0])
    val = scanner._get_signal_strength(0)  # type: ignore[call-arg]
    assert np.isclose(val, -200.0)


# ---------------------------------------------------------------------------
# Phase 2 tests — TransmissionRecord
# ---------------------------------------------------------------------------

def test_transmission_record_built_on_kept_wav(tmp_path: Path) -> None:
    """_process_completed_transmission returns a TransmissionRecord for a kept WAV."""
    wav_dir = str(tmp_path / "wav")
    os.makedirs(wav_dir)
    tmp_wav = str(tmp_path / "tmp_460125000_20250101T120000.wav")

    # Write a 2-second WAV (2 * 8000 = 16000 samples, 16-bit → 32000 bytes of data)
    _write_wav(tmp_wav, num_samples=16_000)

    started_at = 1_700_000_000.0
    msg = ChannelMessage(
        state='off',
        rf=460.125,
        bb=0,
        channel=0,
        wav_tmp_path=tmp_wav,
        started_at=started_at,
        signal_db=-42,
        matched_ctcss=100.0,
    )

    scanner = make_test_scanner(wav_dir=wav_dir)
    scanner.frequency_manager.get_label = MagicMock(return_value="FIRE1")
    scanner.frequency_manager.is_priority = MagicMock(return_value=1)
    msg.label = "FIRE1"
    msg.priority = 1

    _msg, record = scanner._process_completed_transmission(msg)  # type: ignore[union-attr]

    assert record is not None
    assert isinstance(record, TransmissionRecord)
    assert record.rf == pytest.approx(460.125)
    assert record.bb_hz == 0
    assert record.channel == 0
    assert record.label == "FIRE1"
    assert record.priority == 1
    assert record.matched_ctcss_hz == 100.0
    assert record.signal_db == -42
    assert record.classification is None
    assert record.wav_path == _msg.file  # type: ignore[union-attr]
    assert record.started_at == started_at
    assert record.metadata == {}
    # Duration: wav_bytes_per_sec(16)=16000 bytes/sec; 16000 samples × 2 bytes = 32000 data bytes
    # => 32000 / 16000 = 2.0 s; consistent with wav_duration_sec(WAV_HEADER_BYTES + 32000, 16)
    expected_duration = wav_duration_sec(WAV_HEADER_BYTES + wav_bytes_per_sec(16) * 2, 16)
    assert record.duration_sec == pytest.approx(expected_duration, abs=0.01)


def test_transmission_record_duration_set_on_msg(tmp_path: Path) -> None:
    """duration_sec must be set on both msg and the returned record."""
    wav_dir = str(tmp_path / "wav")
    os.makedirs(wav_dir)
    tmp_wav = str(tmp_path / "tmp_460125000_20250101T120001.wav")
    _write_wav(tmp_wav, num_samples=8_000)  # 1-second recording

    msg = ChannelMessage(
        state='off', rf=460.125, bb=0, channel=0,
        wav_tmp_path=tmp_wav, started_at=1_700_000_000.0,
    )

    scanner = make_test_scanner(wav_dir=wav_dir)
    _msg, record = scanner._process_completed_transmission(msg)  # type: ignore[union-attr]

    assert record is not None
    assert _msg.duration_sec == pytest.approx(1.0, abs=0.01)  # type: ignore[union-attr]
    assert record.duration_sec == pytest.approx(1.0, abs=0.01)
    # Verify helpers agree: wav_duration_sec(WAV_HEADER_BYTES + 8000 samples * 2 bytes, 16) = 1.0
    assert record.duration_sec == pytest.approx(
        wav_duration_sec(WAV_HEADER_BYTES + wav_bytes_per_sec(16) * 1, 16)
    )


def test_transmission_record_none_on_ctcss_discard(tmp_path: Path) -> None:
    """CTCSS-mismatch discard path must return record=None."""
    wav_dir = str(tmp_path / "wav")
    os.makedirs(wav_dir)
    tmp_wav = str(tmp_path / "tmp_ctcss_discard.wav")
    _write_wav(tmp_wav, num_samples=16_000)

    msg = ChannelMessage(
        state='off', rf=460.125, bb=0, channel=0,
        wav_tmp_path=tmp_wav, discard=True,
    )
    scanner = make_test_scanner(wav_dir=wav_dir)
    _msg, record = scanner._process_completed_transmission(msg)  # type: ignore[union-attr]

    assert record is None
    assert not os.path.exists(tmp_wav), "Discarded WAV must be deleted"


def test_transmission_record_none_on_short_recording(tmp_path: Path) -> None:
    """Short-recording discard path must return record=None."""
    wav_dir = str(tmp_path / "wav")
    os.makedirs(wav_dir)
    tmp_wav = str(tmp_path / "tmp_short.wav")
    # Write an effectively empty WAV (just the 44-byte header, no samples)
    _write_wav(tmp_wav, num_samples=0)

    msg = ChannelMessage(
        state='off', rf=460.125, bb=0, channel=0,
        wav_tmp_path=tmp_wav,
    )
    scanner = make_test_scanner(wav_dir=wav_dir)
    _msg, record = scanner._process_completed_transmission(msg)  # type: ignore[union-attr]

    assert record is None
    assert not os.path.exists(tmp_wav), "Discarded WAV must be deleted"


def test_transmission_record_is_immutable() -> None:
    """TransmissionRecord must be frozen (FrozenInstanceError on mutation)."""
    record = TransmissionRecord(
        rf=460.125,
        bb_hz=0,
        channel=0,
        label=None,
        priority=None,
        matched_ctcss_hz=None,
        signal_db=None,
        classification=None,
        wav_path="/wav/some_file.wav",
        started_at=1_700_000_000.0,
        duration_sec=2.0,
    )
    with pytest.raises(FrozenInstanceError):
        record.rf = 461.0  # pyright: ignore[reportAttributeAccessIssue]


def test_transmission_record_str() -> None:
    """TransmissionRecord.__str__ must include key fields."""
    record = TransmissionRecord(
        rf=460.125,
        bb_hz=0,
        channel=2,
        label="FIRE1",
        priority=1,
        matched_ctcss_hz=100.0,
        signal_db=-42,
        classification="V",
        wav_path="/wav/some_file.wav",
        started_at=1_700_000_000.0,
        duration_sec=3.5,
    )
    s = str(record)
    assert "460.1250" in s
    assert "3.5s" in s
    assert "FIRE1" in s
    assert "P1" in s
    assert "V" in s
    assert "100.0Hz" in s


# ---------------------------------------------------------------------------
# hold_scan_on / Scanner.interesting matrix tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "record, state, msg_file, classification, hold_scan_on, expected_interesting",
    [
        # Non-recording mode tests
        (False, "on", None, None, None, True),
        (False, "off", None, None, None, False),
        (False, "on", None, "V", {"V"}, True),
        (False, "off", None, "V", {"V"}, False),
        # Recording mode with hold_scan_on = None (default: hold on all saved wav files)
        (True, "on", "test.wav", "V", None, True),
        (True, "on", "test.wav", "D", None, True),
        (True, "on", "test.wav", "S", None, True),
        (True, "on", "test.wav", None, None, True),
        (True, "on", None, "V", None, False),
        # Recording mode with hold_scan_on = {"V"}
        (True, "on", "test.wav", "V", {"V"}, True),
        (True, "on", "test.wav", "D", {"V"}, False),
        (True, "on", "test.wav", "S", {"V"}, False),
        (True, "on", "test.wav", None, {"V"}, False),
        # Recording mode with hold_scan_on = set() (empty set: never hold on any classification)
        (True, "on", "test.wav", "V", set(), False),
        (True, "on", "test.wav", "D", set(), False),
        (True, "on", "test.wav", "S", set(), False),
        (True, "on", "test.wav", None, set(), False),
        # Recording mode with hold_scan_on = {"V", "D"}
        (True, "on", "test.wav", "V", {"V", "D"}, True),
        (True, "on", "test.wav", "D", {"V", "D"}, True),
        (True, "on", "test.wav", "S", {"V", "D"}, False),
    ],
)
def test_scanner_interesting_matrix(
    tmp_path: Path,
    record: bool,
    state: str,
    msg_file: str | None,
    classification: str | None,
    hold_scan_on: set[str] | None,
    expected_interesting: bool,
) -> None:
    wav_dir = str(tmp_path / "wav")
    os.makedirs(wav_dir, exist_ok=True)
    scanner = make_test_scanner(wav_dir=wav_dir)
    scanner.record = record
    scanner.hold_scan_on = hold_scan_on

    msg = ChannelMessage(
        state=state,
        rf=145.5,
        bb=0,
        channel=0,
        file=msg_file,
        classification=classification,
    )

    assert scanner.interesting(msg) is expected_interesting


@pytest.mark.asyncio
async def test_assign_channels_skips_inactive_banks(tmp_path: Path) -> None:
    """_assign_channels_to_demodulators must skip channels whose resolved banks are not active."""
    wav_dir = str(tmp_path / "wav")
    os.makedirs(wav_dir, exist_ok=True)
    scanner = make_test_scanner(wav_dir=wav_dir)

    # Set active bank filter to OPERATIONS only
    from frequency_manager import FrequencyConfiguration, FrequencyManager
    fm = FrequencyManager(FrequencyConfiguration(file_name=None, disable_lockout=False, disable_priority=False), 5000)
    fm.set_active_banks(["OPERATIONS"])
    await fm.add({'single': 462.5625, 'banks': ['FRS_FAMILY'], 'label': 'FRS Ch 1'})
    await fm.add({'single': 467.7125, 'banks': ['OPERATIONS'], 'label': 'FRS Ch 14 Ops'})
    scanner.frequency_manager = fm
    scanner.mismatched_freqs = {}

    # Create two channels: one in FRS_FAMILY (inactive) and one in OPERATIONS (active)
    ch_inactive = ChannelFrequency(
        rf=462.5625, bb=10000, active=False, hanging=False, locked=False,
        label="FRS Ch 1"
    )
    ch_active = ChannelFrequency(
        rf=467.7125, bb=20000, active=False, hanging=False, locked=False,
        label="FRS Ch 14 Ops"
    )

    # Mock receiver & demodulators (1 free demodulator)
    from unittest.mock import AsyncMock
    demod = MagicMock()
    demod.center_freq = 0
    demod.set_center_freq = AsyncMock()
    scanner.receiver = MagicMock()
    scanner.receiver.demodulators = [demod]
    scanner.receiver.get_demod_freqs = MagicMock(return_value=[])
    scanner.center_freq = 460000000
    scanner._demod_signal_stats = {0: (0.0, 0)}

    await scanner._assign_channels_to_demodulators([ch_inactive, ch_active])

    # The active channel (467.7125 MHz, bb=20000) should have been assigned, while inactive (bb=10000) was skipped
    demod.set_center_freq.assert_called_once()
    assert demod.set_center_freq.call_args[0][0] == 20000


@pytest.mark.asyncio
async def test_assign_channels_promiscuous_skips_bank_scan(tmp_path: Path) -> None:
    """Without --banks (promiscuous), _assign_channels_to_demodulators must not
    call resolve_banks() per channel per cycle — is_bank_active() is always
    active, so the scan is pure waste. Channels must still be assigned."""
    wav_dir = str(tmp_path / "wav")
    os.makedirs(wav_dir, exist_ok=True)
    scanner = make_test_scanner(wav_dir=wav_dir)

    from frequency_manager import FrequencyConfiguration, FrequencyManager
    fm = FrequencyManager(FrequencyConfiguration(file_name=None, disable_lockout=False, disable_priority=False), 5000)
    await fm.add({'single': 462.5625, 'banks': ['FRS_FAMILY'], 'label': 'FRS Ch 1'})
    scanner.frequency_manager = fm
    scanner.mismatched_freqs = {}

    resolve_banks_calls = {"n": 0}
    original_resolve_banks = fm.resolve_banks

    def _spy_resolve_banks(rf: float, ctcss_hz: float | None = None) -> list[str]:
        resolve_banks_calls["n"] += 1
        return original_resolve_banks(rf, ctcss_hz)

    fm.resolve_banks = _spy_resolve_banks  # type: ignore[method-assign]

    ch = ChannelFrequency(
        rf=462.5625, bb=10000, active=False, hanging=False, locked=False,
        label="FRS Ch 1"
    )

    from unittest.mock import AsyncMock
    demod = MagicMock()
    demod.center_freq = 0
    demod.set_center_freq = AsyncMock()
    scanner.receiver = MagicMock()
    scanner.receiver.demodulators = [demod]
    scanner.receiver.get_demod_freqs = MagicMock(return_value=[])
    scanner.center_freq = 460000000
    scanner._demod_signal_stats = {0: (0.0, 0)}

    await scanner._assign_channels_to_demodulators([ch])

    assert resolve_banks_calls["n"] == 0
    demod.set_center_freq.assert_called_once()
    assert demod.set_center_freq.call_args[0][0] == 10000


def test_process_completed_transmission_discards_inactive_banks(tmp_path: Path) -> None:
    """_process_completed_transmission must discard WAV files whose final resolved banks are not active."""
    wav_dir = str(tmp_path / "wav")
    os.makedirs(wav_dir, exist_ok=True)
    tmp_wav = str(tmp_path / "tmp_inactive_bank.wav")
    _write_wav(tmp_wav, num_samples=16_000)

    scanner = make_test_scanner(wav_dir=wav_dir)
    from frequency_manager import FrequencyConfiguration, FrequencyManager
    fm = FrequencyManager(FrequencyConfiguration(file_name=None, disable_lockout=False, disable_priority=False), 5000)
    fm.set_active_banks(["SECURITY"])
    scanner.frequency_manager = fm

    msg = ChannelMessage(
        state='off', rf=467.7125, bb=0, channel=0,
        wav_tmp_path=tmp_wav, banks=["FRS_FAMILY"],
    )

    _msg, record = scanner._process_completed_transmission(msg)

    assert record is None
    assert not os.path.exists(tmp_wav), "Discarded WAV must be deleted"
    assert _msg.detail == "Discarded inactive bank selection"


@pytest.mark.asyncio
async def test_add_metadata_populates_banks_when_filtering(tmp_path: Path) -> None:
    """_add_metadata must populate ChannelFrequency.banks when active_banks is
    set, and leave it [] in promiscuous mode -- mirroring the gated resolve in
    _assign_channels_to_demodulators so non-bank users pay no per-cycle cost."""
    wav_dir = str(tmp_path / "wav")
    os.makedirs(wav_dir, exist_ok=True)
    scanner = make_test_scanner(wav_dir=wav_dir)

    from frequency_manager import FrequencyConfiguration, FrequencyManager
    fm = FrequencyManager(FrequencyConfiguration(file_name=None, disable_lockout=False, disable_priority=False), 5000)
    await fm.add({'single': 462.5625, 'banks': ['FRS_FAMILY'], 'label': 'FRS Ch 1'})
    scanner.frequency_manager = fm

    # Receiver stub: no live demodulators, center frequency 460 MHz. The test
    # baseband 2,562,500 Hz resolves to 462.5625 MHz (the FRS_FAMILY entry).
    scanner.receiver = MagicMock()
    scanner.receiver.get_demod_freq_map = MagicMock(return_value={})
    scanner.receiver.center_freq = 460000000
    bb = 2562500

    # Promiscuous mode (no --banks): no bank tags resolved.
    fm.set_active_banks(None)
    sweep = scanner._add_metadata(np.array([bb]))
    assert sweep[0].banks == []

    # Bank filtering active: tags resolved onto the channel.
    fm.set_active_banks(["FRS_FAMILY"])
    sweep = scanner._add_metadata(np.array([bb]))
    assert sweep[0].banks == ["FRS_FAMILY"]

    # Resolved tags reflect the entry's configured banks even when the active
    # selection differs; discarding by active_banks happens downstream in
    # _process_current_demodulators via is_bank_active().
    fm.set_active_banks(["OPERATIONS"])
    sweep = scanner._add_metadata(np.array([bb]))
    assert sweep[0].banks == ["FRS_FAMILY"]

    # A hit outside every configured entry resolves to the UNTAGGED sentinel
    # when bank filtering is active (fail-closed), mirroring tier 5.
    sweep = scanner._add_metadata(np.array([0]))
    assert sweep[0].banks == ["UNTAGGED"]


@pytest.mark.asyncio
async def test_set_active_banks_takes_effect_next_assignment(tmp_path: Path) -> None:
    """set_active_banks() must flip which channels _assign_channels_to_demodulators
    assigns on the next cycle (immediate-apply semantics)."""
    wav_dir = str(tmp_path / "wav")
    os.makedirs(wav_dir, exist_ok=True)
    scanner = make_test_scanner(wav_dir=wav_dir)

    from frequency_manager import FrequencyConfiguration, FrequencyManager
    fm = FrequencyManager(FrequencyConfiguration(file_name=None, disable_lockout=False, disable_priority=False), 5000)
    await fm.add({'single': 462.5625, 'banks': ['FRS_FAMILY'], 'label': 'FRS Ch 1'})
    await fm.add({'single': 467.7125, 'banks': ['OPERATIONS'], 'label': 'FRS Ch 14 Ops'})
    scanner.frequency_manager = fm
    scanner.mismatched_freqs = {}

    from unittest.mock import AsyncMock
    demod = MagicMock()
    demod.center_freq = 0
    demod.set_center_freq = AsyncMock()
    scanner.receiver = MagicMock()
    scanner.receiver.demodulators = [demod]
    scanner.receiver.get_demod_freqs = MagicMock(return_value=[])
    scanner.center_freq = 460000000
    scanner._demod_signal_stats = {0: (0.0, 0)}

    ch_family = ChannelFrequency(
        rf=462.5625, bb=10000, active=False, hanging=False, locked=False,
        label="FRS Ch 1"
    )
    ch_ops = ChannelFrequency(
        rf=467.7125, bb=20000, active=False, hanging=False, locked=False,
        label="FRS Ch 14 Ops"
    )

    # Cycle 1: OPERATIONS active -> the OPERATIONS channel is assigned.
    scanner.set_active_banks(["OPERATIONS"])
    await scanner._assign_channels_to_demodulators([ch_family, ch_ops])
    assert demod.set_center_freq.call_args[0][0] == 20000

    # Reset the demodulator to a free slot for the next cycle.
    demod.set_center_freq.reset_mock()
    demod.center_freq = 0

    # Cycle 2: switch at runtime to FRS_FAMILY -> the FRS_FAMILY channel is assigned.
    scanner.set_active_banks(["FRS_FAMILY"])
    await scanner._assign_channels_to_demodulators([ch_family, ch_ops])
    demod.set_center_freq.assert_called_once()
    assert demod.set_center_freq.call_args[0][0] == 10000

