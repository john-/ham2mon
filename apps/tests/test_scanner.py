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
from frequency_manager import ChannelMessage, TransmissionRecord
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
        rf=460_125_000.0,
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
        state='off', rf=460_125_000.0, bb=0, channel=0,
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
        state='off', rf=460_125_000.0, bb=0, channel=0,
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
        state='off', rf=460_125_000.0, bb=0, channel=0,
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
