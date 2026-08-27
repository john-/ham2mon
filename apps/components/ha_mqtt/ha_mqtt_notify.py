"""Home Assistant MQTT TransmissionNotifier component for ham2mon.

Publishes MQTT Discovery configs so entities appear automatically in Home
Assistant (no manual ``configuration.yaml`` editing), then pushes per
transmission state updates. Mirrors the lazy-connect / self-healing /
``stop_async`` pattern used across ham2mon TransmissionNotifier components.

Entities created per ham2mon instance (grouped under one HA "device" keyed by
``instance``):

- ``sensor.<instance>_last_transmission`` (freq, with duration/priority/etc attrs)
- ``binary_sensor.<instance>_voice_activity`` (on during Tx, auto-off via off_delay)

Bank tags are read per-transmission from ``TransmissionRecord.banks`` and
published in the state payload. They are not part of the component config.

Requires: aiomqtt
"""

import asyncio
import json
import os
import socket

import aiomqtt
from frequency_manager import TransmissionRecord
from typing_extensions import override

from components.base import TransmissionNotifier

CONNECT_TIMEOUT_SEC: float = 10.0
CLOSE_TIMEOUT_SEC: float = 5.0


class HomeAssistantMqttComponent(TransmissionNotifier):
    """TransmissionNotifier component publishing HA-discoverable MQTT entities."""

    def __init__(self, config: dict[str, object]) -> None:
        super().__init__(config)

        self.broker_host: str = str(
            config.get("broker_host", os.getenv("MQTT_BROKER_HOST", "localhost"))
        )
        self.broker_port: int = int(str(config.get("broker_port", 1883)))
        username_val = config.get("username")
        password_val = config.get("password")
        self.username: str | None = str(username_val) if username_val else os.getenv("MQTT_USERNAME")
        self.password: str | None = str(password_val) if password_val else os.getenv("MQTT_PASSWORD")

        self.instance: str = str(
            config.get("instance") or os.getenv("H2M_INSTANCE") or socket.gethostname()
        )
        self.wanted: str = str(config.get("wanted", "V"))
        self.off_delay_sec: int = int(str(config.get("off_delay_sec", 5)))

        self.discovery_prefix: str = str(config.get("discovery_prefix", "homeassistant"))
        self.node_id: str = f"ham2mon_{self.instance.lower()}"
        self.base_topic: str = f"ham2mon/{self.instance.lower()}"
        self.availability_topic: str = f"{self.base_topic}/status"

        self._client: aiomqtt.Client | None = None
        self._connect_lock: asyncio.Lock = asyncio.Lock()
        self._connected: bool = False
        self._discovery_published: bool = False

    @property
    def is_healthy(self) -> bool:
        """Return True if the MQTT client believes it is connected."""
        return self._connected

    # ---- Component lifecycle -------------------------------------------------

    @override
    def start(self) -> None:
        """Record config only. Actual connection happens lazily from async context."""
        self.logger.info(
            "Starting HomeAssistantMqttComponent (broker=%s:%s, instance=%s, wanted=%s)",
            self.broker_host,
            self.broker_port,
            self.instance,
            self.wanted,
        )

    @override
    def stop(self) -> None:
        """Synchronous stop fallback; cannot close an async session.

        Real teardown happens in ``stop_async()``. Note that
        ``Component.recover()`` only invokes this synchronous ``stop()``, so it
        cannot restore the MQTT session; no recovery path currently calls it.
        """

    @override
    async def stop_async(self) -> None:
        """Publish ``offline`` status and close the MQTT session cleanly."""
        if self._client is not None:
            if self._connected:
                try:
                    await self._client.publish(
                        self.availability_topic, "offline", retain=True
                    )
                except Exception as e:  # noqa: BLE001
                    self.logger.debug("Failed to publish offline status: %s", e)
            await self._close_client(self._client)
        self._client = None
        self._connected = False
        self.logger.info("Stopped HomeAssistantMqttComponent")

    # ---- Connection management -------------------------------------------------

    async def _ensure_connected(self) -> bool:
        """Lazily connect (and publish discovery once). Returns success."""
        if self._connected and self._client is not None:
            return True

        async with self._connect_lock:
            if self._connected and self._client is not None:
                return True

            try:
                will = aiomqtt.Will(
                    topic=self.availability_topic,
                    payload="offline",
                    retain=True,
                )
                client = aiomqtt.Client(
                    hostname=self.broker_host,
                    port=self.broker_port,
                    username=self.username,
                    password=self.password,
                    will=will,
                )
                client = await asyncio.wait_for(
                    client.__aenter__(), timeout=CONNECT_TIMEOUT_SEC
                )
                self._client = client
                self._connected = True

                await client.publish(self.availability_topic, "online", retain=True)

                if not self._discovery_published:
                    await self._publish_discovery()
                    self._discovery_published = True

                return True
            except Exception as e:  # noqa: BLE001
                self.logger.error(
                    "Failed to connect to MQTT broker %s:%s: %s",
                    self.broker_host,
                    self.broker_port,
                    e,
                )
                if self._client is not None:
                    await self._close_client(self._client)
                self._client = None
                self._connected = False
                # Broker may have restarted without persistence; republish
                # discovery configs on the next successful reconnect.
                self._discovery_published = False
                return False

    async def _close_client(self, client: aiomqtt.Client | None) -> None:
        """Best-effort close of an MQTT session, bounded in time. No-op if None."""
        if client is None:
            return
        try:
            await asyncio.wait_for(
                client.__aexit__(None, None, None), timeout=CLOSE_TIMEOUT_SEC
            )
        except Exception as e:  # noqa: BLE001
            self.logger.debug("Error closing MQTT session: %s", e)

    async def _publish_discovery(self) -> None:
        """Publish retained MQTT Discovery config payloads for each entity."""
        if self._client is None:
            return

        device = {
            "identifiers": [self.node_id],
            "name": f"ham2mon ({self.instance})",
            "manufacturer": "ham2mon",
            "model": "SDR Scanner",
        }

        last_tx_topic = f"{self.base_topic}/last_transmission/state"
        voice_activity_topic = f"{self.base_topic}/voice_activity/state"

        entities = [
            (
                "sensor",
                "last_transmission",
                {
                    "name": "Last Transmission",
                    "unique_id": f"{self.node_id}_last_transmission",
                    "state_topic": last_tx_topic,
                    "unit_of_measurement": "MHz",
                    "value_template": "{{ value_json.freq }}",
                    "json_attributes_topic": last_tx_topic,
                    "icon": "mdi:radio-handheld",
                    "availability_topic": self.availability_topic,
                    "device": device,
                },
            ),
            (
                "binary_sensor",
                "voice_activity",
                {
                    "name": "Voice Activity",
                    "unique_id": f"{self.node_id}_voice_activity",
                    "state_topic": voice_activity_topic,
                    "payload_on": "ON",
                    "payload_off": "OFF",
                    "off_delay": self.off_delay_sec,
                    "device_class": "sound",
                    "availability_topic": self.availability_topic,
                    "device": device,
                },
            ),
        ]

        for component, object_id, payload in entities:
            topic = (
                f"{self.discovery_prefix}/{component}/{self.node_id}/{object_id}/config"
            )
            await self._client.publish(topic, json.dumps(payload), retain=True)

        self.logger.info("Published MQTT discovery configs for instance=%s", self.instance)

    # ---- Transmission handling -------------------------------------------------

    @override
    async def on_transmission(self, record: TransmissionRecord) -> None:
        """Publish state updates for a completed transmission."""
        if record.classification != self.wanted:
            self.logger.debug(
                "Skipping HA MQTT publish for non-wanted classification: %s",
                record.classification,
            )
            return

        if not await self._ensure_connected():
            return

        if self._client is None:
            return

        ctcss_str = (
            f"{record.matched_ctcss_hz:.1f}Hz"
            if record.matched_ctcss_hz is not None
            else None
        )

        state = {
            "freq": f"{record.rf:.4f}",
            "duration": record.duration_sec,
            "priority": record.priority,
            "strength": record.signal_db,
            "ctcss": ctcss_str,
            "label": record.label,
            "classification": record.classification,
            "wav_path": record.wav_path,
            "created": record.started_at,
            "banks": record.banks,
        }

        try:
            await self._client.publish(
                f"{self.base_topic}/last_transmission/state", json.dumps(state)
            )
            await self._client.publish(f"{self.base_topic}/voice_activity/state", "ON")
        except Exception as e:  # noqa: BLE001
            self.logger.error(
                "Failed to publish MQTT state: %s (will reconnect on next transmission)",
                e,
            )
            await self._close_client(self._client)
            self._client = None
            self._connected = False
            self._discovery_published = False
