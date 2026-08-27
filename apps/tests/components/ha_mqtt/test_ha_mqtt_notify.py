"""Unit tests for HomeAssistantMqttComponent."""

import asyncio
import json
import socket
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from components.ha_mqtt.ha_mqtt_notify import HomeAssistantMqttComponent
from frequency_manager import TransmissionRecord


def make_record(
    classification: str = "V",
    rf: float = 460.125,
    banks: list[str] | None = None,
) -> TransmissionRecord:
    """Build a kept TransmissionRecord for notifier tests."""
    return TransmissionRecord(
        rf=rf,
        bb_hz=0,
        channel=0,
        label="Test",
        priority=1,
        matched_ctcss_hz=131.8,
        signal_db=-42,
        classification=classification,
        wav_path="/tmp/final.wav",
        started_at=1700000000.0,
        duration_sec=4.2,
        metadata={},
        banks=banks or [],
    )


@pytest.fixture
def mqtt_client() -> MagicMock:
    """Mock aiomqtt.Client instance wired to return itself from __aenter__."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.publish = AsyncMock()
    return client


@pytest.fixture
def patched_aiomqtt(mqtt_client: MagicMock) -> MagicMock:
    """Patch aiomqtt.Client/Will in the component module namespace."""
    patcher_client = patch(
        "components.ha_mqtt.ha_mqtt_notify.aiomqtt.Client", return_value=mqtt_client
    )
    patcher_will = patch("components.ha_mqtt.ha_mqtt_notify.aiomqtt.Will")
    patcher_client.start()
    patcher_will.start()
    try:
        yield mqtt_client
    finally:
        patcher_will.stop()
        patcher_client.stop()


def test_defaults_resolve_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("H2M_INSTANCE", "SDR1")
    monkeypatch.delenv("MQTT_BROKER_HOST", raising=False)
    comp = HomeAssistantMqttComponent({})
    assert comp.broker_host == "localhost"
    assert comp.broker_port == 1883
    assert comp.instance == "SDR1"
    assert comp.node_id == "ham2mon_sdr1"
    assert comp.base_topic == "ham2mon/sdr1"
    assert comp.availability_topic == "ham2mon/sdr1/status"
    assert comp.wanted == "V"
    assert comp.off_delay_sec == 5
    assert comp.is_healthy is False


def test_instance_defaults_to_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("H2M_INSTANCE", raising=False)
    comp = HomeAssistantMqttComponent({})
    assert comp.instance == socket.gethostname()
    assert comp.base_topic == f"ham2mon/{socket.gethostname().lower()}"


def test_config_values_are_honored() -> None:
    comp = HomeAssistantMqttComponent(
        {
            "instance": "RAMERICA",
            "broker_host": "10.0.0.5",
            "broker_port": 1884,
            "username": "user",
            "password": "secret",
            "wanted": "D",
            "off_delay_sec": 10,
        }
    )
    assert comp.broker_host == "10.0.0.5"
    assert comp.broker_port == 1884
    assert comp.username == "user"
    assert comp.password == "secret"
    assert comp.instance == "RAMERICA"
    assert comp.wanted == "D"
    assert comp.off_delay_sec == 10


async def test_on_transmission_skips_non_wanted(
    patched_aiomqtt: MagicMock,
) -> None:
    comp = HomeAssistantMqttComponent({"instance": "SDR1"})
    await comp.on_transmission(make_record(classification="D"))
    patched_aiomqtt.__aenter__.assert_not_called()
    patched_aiomqtt.publish.assert_not_called()
    assert comp._connected is False


async def test_on_transmission_publishes_discovery_once_and_state(
    patched_aiomqtt: MagicMock,
) -> None:
    comp = HomeAssistantMqttComponent({"instance": "SDR1"})
    record = make_record(banks=["PUBLIC_SAFETY", "LAW_ENFORCEMENT"])

    await comp.on_transmission(record)
    await comp.on_transmission(record)

    topics = [c.args[0] for c in patched_aiomqtt.publish.call_args_list]

    discovery_topics = [t for t in topics if "/config" in t]
    assert discovery_topics == [
        "homeassistant/sensor/ham2mon_sdr1/last_transmission/config",
        "homeassistant/binary_sensor/ham2mon_sdr1/voice_activity/config",
    ]

    online_calls = [
        c for c in patched_aiomqtt.publish.call_args_list
        if c.args[0] == "ham2mon/sdr1/status"
    ]
    assert len(online_calls) == 1
    assert all(c.kwargs.get("retain") is True for c in online_calls)

    last_tx_calls = [
        c for c in patched_aiomqtt.publish.call_args_list
        if c.args[0] == "ham2mon/sdr1/last_transmission/state"
    ]
    assert len(last_tx_calls) == 2
    payload = json.loads(last_tx_calls[0].args[1])
    assert payload["freq"] == "460.1250"
    assert payload["duration"] == 4.2
    assert payload["banks"] == ["PUBLIC_SAFETY", "LAW_ENFORCEMENT"]
    assert payload["label"] == "Test"
    assert payload["classification"] == "V"
    assert payload["wav_path"] == "/tmp/final.wav"

    voice_calls = [
        c for c in patched_aiomqtt.publish.call_args_list
        if c.args[0] == "ham2mon/sdr1/voice_activity/state"
    ]
    assert len(voice_calls) == 2
    assert voice_calls[0].args[1] == "ON"


async def test_discovery_config_payloads_are_valid_json(
    patched_aiomqtt: MagicMock,
) -> None:
    comp = HomeAssistantMqttComponent({"instance": "SDR1"})
    await comp.on_transmission(make_record())

    for c in patched_aiomqtt.publish.call_args_list:
        if "/config" in c.args[0]:
            payload = json.loads(c.args[1])
            assert payload["device"]["identifiers"] == ["ham2mon_sdr1"]
            assert payload["device"]["name"] == "ham2mon (SDR1)"
            assert payload["availability_topic"] == "ham2mon/sdr1/status"
            if c.args[0].startswith("homeassistant/sensor/"):
                assert payload["unit_of_measurement"] == "MHz"


async def test_connect_failure_recovers_on_next_transmission(
    patched_aiomqtt: MagicMock,
) -> None:
    comp = HomeAssistantMqttComponent({"instance": "SDR1"})
    patched_aiomqtt.__aenter__ = AsyncMock(
        side_effect=[ConnectionError("broker down"), patched_aiomqtt]
    )

    await comp.on_transmission(make_record())
    assert comp._connected is False

    await comp.on_transmission(make_record())
    assert comp._connected is True
    assert comp._discovery_published is True

    discovery_topics = [
        c.args[0] for c in patched_aiomqtt.publish.call_args_list if "/config" in c.args[0]
    ]
    assert len(discovery_topics) == 2


async def test_publish_failure_resets_connection(
    patched_aiomqtt: MagicMock,
) -> None:
    comp = HomeAssistantMqttComponent({"instance": "SDR1"})
    patched_aiomqtt.publish = AsyncMock(
        side_effect=ConnectionError("broker dropped")
    )

    await comp.on_transmission(make_record())
    assert comp._connected is False
    assert comp._client is None
    assert comp._discovery_published is False
    patched_aiomqtt.__aexit__.assert_awaited_once()


async def test_state_publish_failure_closes_session(
    patched_aiomqtt: MagicMock,
) -> None:
    comp = HomeAssistantMqttComponent({"instance": "SDR1"})
    await comp.on_transmission(make_record())
    assert comp._connected is True

    patched_aiomqtt.publish = AsyncMock(
        side_effect=ConnectionError("broker dropped")
    )
    await comp.on_transmission(make_record())

    assert comp._connected is False
    assert comp._client is None
    assert comp._discovery_published is False
    patched_aiomqtt.__aexit__.assert_awaited_once()


async def test_connect_timeout_does_not_hang(
    monkeypatch: pytest.MonkeyPatch,
    patched_aiomqtt: MagicMock,
) -> None:
    monkeypatch.setattr(
        "components.ha_mqtt.ha_mqtt_notify.CONNECT_TIMEOUT_SEC", 0.05
    )
    comp = HomeAssistantMqttComponent({"instance": "SDR1"})

    pending = asyncio.Future()

    async def slow_enter() -> MagicMock:
        await pending
        return patched_aiomqtt

    patched_aiomqtt.__aenter__ = slow_enter

    start = time.monotonic()
    await comp.on_transmission(make_record())
    elapsed = time.monotonic() - start

    assert comp._connected is False
    assert comp._client is None
    assert elapsed < 1.0
    patched_aiomqtt.__aexit__.assert_not_awaited()


async def test_stop_async_publishes_offline_and_closes(
    patched_aiomqtt: MagicMock,
) -> None:
    comp = HomeAssistantMqttComponent({"instance": "SDR1"})
    await comp.on_transmission(make_record())
    assert comp._connected is True

    await comp.stop_async()

    patched_aiomqtt.publish.assert_awaited_with(
        "ham2mon/sdr1/status", "offline", retain=True
    )
    patched_aiomqtt.__aexit__.assert_awaited_once()
    assert comp._connected is False
    assert comp._client is None
