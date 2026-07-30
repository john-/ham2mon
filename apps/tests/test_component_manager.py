"""Unit tests for ComponentManager execution, timeout isolation, and dispatch."""

import time
from typing import override

import pytest
from components.base import (
    ChannelInfo,
    ComponentResult,
    TransmissionNotifier,
    WavGatekeeper,
)
from components.manager import ComponentManager, load_component_class
from config import ComponentEntryConfig, ConfigError, MasterHam2MonConfig
from frequency_manager import TransmissionRecord


class HangingGatekeeper(WavGatekeeper):
    @override
    def start(self) -> None:
        pass

    @override
    def process(self, wav_path: str, channel_info: ChannelInfo) -> ComponentResult:
        time.sleep(2.0)
        return ComponentResult(keep=False)

    @override
    def stop(self) -> None:
        pass


class CrashingGatekeeper(WavGatekeeper):
    @override
    def start(self) -> None:
        pass

    @override
    def process(self, wav_path: str, channel_info: ChannelInfo) -> ComponentResult:
        raise RuntimeError("Model evaluation crashed")

    @override
    def stop(self) -> None:
        pass


class CountingNotifier(TransmissionNotifier):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__(config)
        self.count: int = 0

    @override
    def start(self) -> None:
        pass

    @override
    async def on_transmission(self, record: TransmissionRecord) -> None:
        self.count += 1

    @override
    def stop(self) -> None:
        pass


class CrashingNotifier(TransmissionNotifier):
    @override
    def start(self) -> None:
        pass

    @override
    async def on_transmission(self, record: TransmissionRecord) -> None:
        raise RuntimeError("Notifier output failed")

    @override
    def stop(self) -> None:
        pass


def test_load_component_class_validation():
    with pytest.raises(
        ConfigError,
        match=r"Invalid component class_path: 'InvalidPath'\. Expected format",
    ):
        load_component_class("InvalidPath")

    with pytest.raises(ConfigError, match="Failed to import"):
        load_component_class("nonexistent_module.Foo")


def test_component_manager_timeout_isolation():
    cfg = MasterHam2MonConfig()
    cfg.components.wav_gatekeeper = ComponentEntryConfig(
        class_path="tests.test_component_manager.HangingGatekeeper",
        timeout_sec=0.1,
        config={},
        name="hanging_gk",
    )

    mgr = ComponentManager(cfg)
    mgr.start_all()

    info = ChannelInfo(
        rf=460.125,
        bb_hz=0,
        channel=0,
        label=None,
        priority=None,
        matched_ctcss_hz=None,
        signal_db=None,
        timestamp=1700000000.0,
        wav_tmp_path="/tmp/fake.wav",
    )

    res = mgr.process_wav("/tmp/fake.wav", info)
    assert res.keep is True  # Safe default on timeout
    assert "timeout" in (res.detail or "").lower()

    mgr.stop_all()


def test_component_manager_exception_fault_isolation():
    cfg = MasterHam2MonConfig()
    cfg.components.wav_gatekeeper = ComponentEntryConfig(
        class_path="tests.test_component_manager.CrashingGatekeeper",
        timeout_sec=1.0,
        config={},
        name="crashing_gk",
    )

    mgr = ComponentManager(cfg)
    mgr.start_all()

    info = ChannelInfo(
        rf=460.125,
        bb_hz=0,
        channel=0,
        label=None,
        priority=None,
        matched_ctcss_hz=None,
        signal_db=None,
        timestamp=1700000000.0,
        wav_tmp_path="/tmp/fake.wav",
    )

    res = mgr.process_wav("/tmp/fake.wav", info)
    assert res.keep is True  # Safe default on exception
    assert "error" in (res.detail or "").lower()

    mgr.stop_all()


@pytest.mark.asyncio
async def test_component_manager_dispatch_transmission():
    cfg = MasterHam2MonConfig()
    n1 = CountingNotifier({})
    n2 = CrashingNotifier({})
    mgr = ComponentManager(cfg)
    mgr.notifiers = [n1, n2]

    rec = TransmissionRecord(
        rf=460.125,
        bb_hz=0,
        channel=0,
        label="Test",
        priority=None,
        matched_ctcss_hz=None,
        signal_db=-60,
        classification="V",
        wav_path="/tmp/final.wav",
        started_at=1700000000.0,
        duration_sec=2.5,
        metadata={},
    )

    await mgr.dispatch_transmission(rec)
    assert n1.count == 1
