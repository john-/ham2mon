"""Unit tests for Scanner integration with ComponentManager and sidecar JSON writing."""

import json
import os
from pathlib import Path
from typing import override
from unittest.mock import AsyncMock, MagicMock

import pytest
from components.base import ChannelInfo, ComponentResult, WavGatekeeper
from components.manager import ComponentManager
from config import MasterHam2MonConfig
from frequency_manager import (
    FrequencyConfiguration,
    FrequencyManager,
    TransmissionRecord,
)
from scanner import Scanner


class MockGatekeeper(WavGatekeeper):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__(config)
        self.keep_ret: bool = bool(config.get("keep", True))
        meta_val = config.get("metadata")
        self.meta_ret: dict[str, object] = (
            dict(meta_val) if isinstance(meta_val, dict) else {}
        )
        cls_val = config.get("classification")
        self.class_ret: str | None = str(cls_val) if cls_val is not None else None

    @override
    def start(self) -> None:
        pass

    @override
    def process(self, wav_path: str, channel_info: ChannelInfo) -> ComponentResult:
        return ComponentResult(
            keep=self.keep_ret,
            classification=self.class_ret,
            metadata=self.meta_ret,
        )

    @override
    def stop(self) -> None:
        pass


def test_scanner_sidecar_json_written_on_kept_wav(tmp_path: Path):
    wav_dir = str(tmp_path / "wav")
    os.makedirs(wav_dir, exist_ok=True)
    tmp_wav = str(tmp_path / "tmp_460125000_20250101T120001.wav")

    # Generate valid 1-second WAV
    with open(tmp_wav, "wb") as f:
        _ = f.write(
            b"RIFF"
            + (36 + 16000).to_bytes(4, "little")
            + b"WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data"
            + (16000).to_bytes(4, "little")
            + b"\x00" * 16000
        )

    cfg = MasterHam2MonConfig()
    cfg.audio.wav_dir = wav_dir

    fm = FrequencyManager(
        FrequencyConfiguration(disable_lockout=False, disable_priority=False), 5000
    )

    scanner = Scanner.__new__(Scanner)
    scanner.config = cfg
    scanner._wav_dir = wav_dir
    scanner.frequency_manager = fm

    # Spy on resolve_banks: this path (kept WAV + component, empty msg.banks)
    # must resolve banks exactly once, not once per fallback site (scanner.py
    # 544/578/621 pre-dedup).
    resolve_banks_calls = {"n": 0}
    original_resolve_banks = fm.resolve_banks

    def _spy_resolve_banks(rf: float, ctcss_hz: float | None = None) -> list[str]:
        resolve_banks_calls["n"] += 1
        return original_resolve_banks(rf, ctcss_hz)

    fm.resolve_banks = _spy_resolve_banks  # type: ignore[method-assign]

    gk = MockGatekeeper(
        {"keep": True, "classification": "VOICE", "metadata": {"confidence": 0.98}}
    )

    scanner._component_manager = ComponentManager(cfg)
    scanner._component_manager.wav_gatekeeper = gk

    from channel_loggers import ChannelMessage

    msg = ChannelMessage(
        state="off",
        rf=460.125,
        bb=0,
        channel=0,
        wav_tmp_path=tmp_wav,
        started_at=1_700_000_000.0,
    )

    _msg, record = scanner._process_completed_transmission(msg)

    assert resolve_banks_calls["n"] == 1

    assert record is not None
    assert record.classification == "VOICE"
    assert record.metadata == {"confidence": 0.98}
    assert os.path.exists(record.wav_path)

    # Check sidecar JSON file exists and has expected payload
    expected_json_path = os.path.splitext(record.wav_path)[0] + ".json"
    assert os.path.exists(expected_json_path)

    with open(expected_json_path, "r") as f:
        sidecar_data: dict[str, object] = json.load(f)

    assert sidecar_data["rf"] == 460.125
    assert sidecar_data["classification"] == "VOICE"
    assert sidecar_data["metadata"] == {"confidence": 0.98}
    assert sidecar_data["banks"] == []


def test_scanner_no_sidecar_json_on_discard(tmp_path: Path):
    wav_dir = str(tmp_path / "wav")
    os.makedirs(wav_dir, exist_ok=True)
    tmp_wav = str(tmp_path / "tmp_460125000_20250101T120001.wav")

    with open(tmp_wav, "wb") as f:
        _ = f.write(
            b"RIFF"
            + (36 + 16000).to_bytes(4, "little")
            + b"WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data"
            + (16000).to_bytes(4, "little")
            + b"\x00" * 16000
        )

    cfg = MasterHam2MonConfig()
    cfg.audio.wav_dir = wav_dir

    fm = FrequencyManager(
        FrequencyConfiguration(disable_lockout=False, disable_priority=False), 5000
    )

    scanner = Scanner.__new__(Scanner)
    scanner.config = cfg
    scanner._wav_dir = wav_dir
    scanner.frequency_manager = fm

    gk = MockGatekeeper(
        {"keep": False, "classification": "SKIP", "metadata": {"confidence": 0.1}}
    )

    scanner._component_manager = ComponentManager(cfg)
    scanner._component_manager.wav_gatekeeper = gk

    from channel_loggers import ChannelMessage

    msg = ChannelMessage(
        state="off",
        rf=460.125,
        bb=0,
        channel=0,
        wav_tmp_path=tmp_wav,
        started_at=1_700_000_000.0,
    )

    _msg, record = scanner._process_completed_transmission(msg)

    assert record is None
    assert not os.path.exists(tmp_wav)


@pytest.mark.asyncio
async def test_got_channel_activity_dispatches_notifiers_only_when_interesting(
    tmp_path: Path,
) -> None:
    """Verify got_channel_activity() dispatches notifiers when interesting(msg) is True."""
    wav_dir = str(tmp_path / "wav")
    os.makedirs(wav_dir)
    tmp_wav = str(tmp_path / "tmp_460125000_20250101T120002.wav")
    with open(tmp_wav, "wb") as f:
        _ = f.write(
            b"RIFF"
            + (36 + 16000).to_bytes(4, "little")
            + b"WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data"
            + (16000).to_bytes(4, "little")
            + b"\x00" * 16000
        )

    cfg = MasterHam2MonConfig()
    cfg.audio.wav_dir = wav_dir

    fm = MagicMock()
    fm.get_label = MagicMock(return_value="TEST")
    fm.is_priority = MagicMock(return_value=None)

    notifier_mock = MagicMock()
    notifier_mock.on_transmission = AsyncMock()

    cm = ComponentManager(cfg)
    cm.notifiers = [notifier_mock]

    scanner = Scanner.__new__(Scanner)
    scanner.config = cfg
    scanner._wav_dir = wav_dir
    scanner.auto_priority = False
    scanner.frequency_manager = fm
    scanner.activity_logger = AsyncMock()
    scanner.frequency_provider = AsyncMock()
    scanner._component_manager = cm
    scanner.record = True
    scanner.hold_scan_on = None


    from channel_loggers import ChannelMessage

    msg = ChannelMessage(
        state="off",
        rf=460.125,
        bb=0,
        channel=0,
        wav_tmp_path=tmp_wav,
        started_at=1_700_000_000.0,
    )

    # 1. Positive case: kept WAV recording -> interesting(msg) is True, notifier called
    await scanner.got_channel_activity(msg)

    assert notifier_mock.on_transmission.called
    record_passed = notifier_mock.on_transmission.call_args[0][0]
    assert isinstance(record_passed, TransmissionRecord)
    assert record_passed.rf == 460.125

    # 2. Negative case: discarded recording (mismatched CTCSS) -> record is None, interesting(msg) is False, notifier NOT called
    tmp_mismatch_wav = str(tmp_path / "tmp_mismatch.wav")
    with open(tmp_mismatch_wav, "wb") as f:
        _ = f.write(b"RIFF" + b"\x00" * 36 + b"WAVEfmt " + b"\x00" * 16)

    notifier_mock.on_transmission.reset_mock()

    msg_unwanted = ChannelMessage(
        state="off",
        rf=460.125,
        bb=0,
        channel=0,
        wav_tmp_path=tmp_mismatch_wav,
        discard=True,
        started_at=1_700_000_000.0,
    )


    await scanner.got_channel_activity(msg_unwanted)

    assert msg_unwanted.file is None
    assert not scanner.interesting(msg_unwanted)
    notifier_mock.on_transmission.assert_not_called()

