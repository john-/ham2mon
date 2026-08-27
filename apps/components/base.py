"""
Base data contracts and abstract protocols for the ham2mon component architecture.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from frequency_manager import TransmissionRecord

logger = logging.getLogger(f"ham2mon.{__name__}")


@dataclass
class ComponentResult:
    """Return value from WavGatekeeper.process()."""

    keep: bool
    """True -> keep and promote to final storage. False -> delete tmp file."""

    classification: str | None = None
    """Short label appended to the final filename. 'V', 'D', 'S', or any
    user-defined string. None means no classification label in filename."""

    detail: str | None = None
    """Human-readable reason logged when keep=False
    (e.g. 'Discarded: data burst'). Appears in ChannelMessage.detail
    and the application log."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Arbitrary key-value pairs written to sidecar JSON and passed to TransmissionNotifiers.
    Example: {'confidence': 0.94}"""


@dataclass(frozen=True)
class ChannelInfo:
    """Read-only snapshot passed to WavGatekeeper.process()."""

    rf: float
    """RF frequency in MHz (e.g. 460.125)."""

    bb_hz: int
    """Baseband offset frequency in Hz."""

    channel: int
    """Demodulator slot index (0-based)."""

    label: str | None
    """Human-readable label from the frequencies YAML file, or None."""

    priority: int | None
    """Priority level if marked priority, else None."""

    matched_ctcss_hz: float | None
    """The CTCSS tone (Hz) that opened squelch, or None if not applicable."""

    signal_db: int | None
    """Average signal strength in dB over the transmission, or None."""

    timestamp: float
    """Unix timestamp when the transmission started (ChannelMessage.started_at)."""

    wav_tmp_path: str
    """Absolute path to the tmp WAV file being evaluated."""

    banks: list[str] = field(default_factory=list)
    """Resolved scanner bank tags for this channel."""


class Component(ABC):
    """Base abstract class for all ham2mon components."""

    config: dict[str, object]
    name: str
    logger: logging.Logger

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self.name = str(config.get("name") or self.__class__.__name__)
        self.logger = logging.getLogger(f"ham2mon.component.{self.name}")

    @abstractmethod
    def start(self) -> None:
        """Called once at ham2mon startup. Fail fast on error."""

    @abstractmethod
    def stop(self) -> None:
        """Called on orderly shutdown."""

    async def stop_async(self) -> None:
        """Async teardown hook; defaults to the synchronous ``stop()``.

        Components with async-only teardown (e.g. an MQTT session close)
        override this. Invoked by ``ComponentManager.stop_all_async()`` on
        orderly shutdown.
        """
        self.stop()

    def recover(self) -> None:
        """Attempt orderly recovery by restarting the component."""
        try:
            self.stop()
        except Exception:
            self.logger.warning(
                "Error stopping %s during recovery", self.name, exc_info=True
            )
        self.start()


class WavGatekeeper(Component, ABC):
    """Abstract base protocol for decision-making WAV evaluation components."""

    @abstractmethod
    def process(self, wav_path: str, channel_info: ChannelInfo) -> ComponentResult:
        """Evaluate a completed tmp WAV recording."""


class TransmissionNotifier(Component, ABC):
    """Abstract base protocol for notification-only transmission subscriber components."""

    @abstractmethod
    async def on_transmission(self, record: TransmissionRecord) -> None:
        """Handle a completed, saved transmission (TransmissionRecord)."""
