"""
TransmissionNotifier component wrapping ham2mon activity loggers.
"""

from channel_loggers import ActivityLogger
from frequency_manager import TransmissionRecord
from typing_extensions import override

from components.base import TransmissionNotifier


class ActivityLoggerComponent(TransmissionNotifier):
    """TransmissionNotifier component wrapping an ActivityLogger instance."""

    _logger_instance: ActivityLogger | None

    def __init__(
        self, config: dict[str, object], activity_logger: ActivityLogger | None = None
    ) -> None:
        super().__init__(config)
        self._logger_instance = activity_logger

    def set_logger(self, activity_logger: ActivityLogger) -> None:
        """Inject activity logger instance."""
        self._logger_instance = activity_logger

    @override
    def start(self) -> None:
        """Start activity logger component."""

    @override
    async def on_transmission(self, record: TransmissionRecord) -> None:
        """Forward TransmissionRecord to activity logger."""
        if self._logger_instance:
            await self._logger_instance.log(None, record=record)

    @override
    def stop(self) -> None:
        """Stop activity logger component."""
