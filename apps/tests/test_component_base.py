"""
Unit tests for ham2mon component base classes and dataclasses.
"""

from components.base import (
    ChannelInfo,
    ComponentResult,
    TransmissionNotifier,
    WavGatekeeper,
)
from frequency_manager import TransmissionRecord
from typing_extensions import override


class DummyGatekeeper(WavGatekeeper):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__(config)
        self.started: bool = False
        self.stopped: bool = False

    @override
    def start(self) -> None:
        self.started = True

    @override
    def process(self, wav_path: str, channel_info: ChannelInfo) -> ComponentResult:
        return ComponentResult(keep=True, classification="V", metadata={"score": 0.99})

    @override
    def stop(self) -> None:
        self.stopped = True


class DummyNotifier(TransmissionNotifier):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__(config)
        self.started: bool = False
        self.stopped: bool = False
        self.last_record: TransmissionRecord | None = None

    @override
    def start(self) -> None:
        self.started = True

    @override
    async def on_transmission(self, record: TransmissionRecord) -> None:
        self.last_record = record

    @override
    def stop(self) -> None:
        self.stopped = True


def test_component_result_defaults():
    res = ComponentResult(keep=True)
    assert res.keep is True
    assert res.classification is None
    assert res.detail is None
    assert res.metadata == {}


def test_channel_info_immutable():
    info = ChannelInfo(
        rf=460.125,
        bb_hz=0,
        channel=0,
        label="Net A",
        priority=1,
        matched_ctcss_hz=156.7,
        signal_db=-65,
        timestamp=1700000000.0,
        wav_tmp_path="/tmp/test.wav",
    )
    assert info.rf == 460.125
    assert info.label == "Net A"


def test_component_recover():
    gk = DummyGatekeeper({})
    gk.start()
    assert gk.started is True
    gk.recover()
    assert gk.stopped is True
    assert gk.started is True
