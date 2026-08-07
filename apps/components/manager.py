"""
Component lifecycle management, dynamic loading, and dispatch execution.
"""

import asyncio
import importlib
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import cast

from config import ConfigError, MasterHam2MonConfig
from frequency_manager import TransmissionRecord

from components.base import (
    ChannelInfo,
    Component,
    ComponentResult,
    TransmissionNotifier,
    WavGatekeeper,
)

logger = logging.getLogger(f"ham2mon.{__name__}")


def load_component_class(class_path: str) -> type[Component]:
    """Dynamically import and return a Component class from a module dot-path."""
    if not class_path or "." not in class_path:
        raise ConfigError(
            f"Invalid component class_path: '{class_path}'. Expected format 'module.ClassName'."
        )

    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        raise ConfigError(
            f"Failed to import component module '{module_path}': {e}"
        ) from e

    try:
        cls_obj = cast(object, getattr(module, class_name))
    except AttributeError as e:
        raise ConfigError(
            f"Component class '{class_name}' not found in module '{module_path}'."
        ) from e

    if not isinstance(cls_obj, type) or not issubclass(cls_obj, Component):
        raise ConfigError(f"Class '{class_path}' must subclass Component.")

    return cls_obj


class ComponentManager:
    """Owns lifecycle (start/stop/restart) and execution dispatching for components."""

    config: MasterHam2MonConfig

    def __init__(self, master_config: MasterHam2MonConfig) -> None:
        self.config = master_config
        self.wav_gatekeeper: WavGatekeeper | None = None
        self.notifiers: list[TransmissionNotifier] = []

        self._wav_timeout_sec: float = 10.0
        self._executor: ThreadPoolExecutor | None = None

        self._init_components()

    def _init_components(self) -> None:
        """Instantiate configured components."""
        cfg = self.config.components

        if cfg.wav_gatekeeper and cfg.wav_gatekeeper.class_path:
            gk_cfg = cfg.wav_gatekeeper
            cls = load_component_class(gk_cfg.class_path)
            instance = cls(gk_cfg.config)
            if not isinstance(instance, WavGatekeeper):
                raise ConfigError(
                    f"Component '{gk_cfg.class_path}' must subclass WavGatekeeper."
                )
            self.wav_gatekeeper = instance
            self._wav_timeout_sec = gk_cfg.timeout_sec

        for notif_cfg in cfg.notifiers:
            if not notif_cfg.class_path:
                continue
            cls = load_component_class(notif_cfg.class_path)
            instance = cls(notif_cfg.config)
            if not isinstance(instance, TransmissionNotifier):
                raise ConfigError(
                    f"Component '{notif_cfg.class_path}' must subclass TransmissionNotifier."
                )
            self.notifiers.append(instance)

    def start_all(self) -> None:
        """Synchronously start all components. Raise ConfigError on failure."""
        if self.wav_gatekeeper:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="ham2mon-wav-gatekeeper",
            )
            try:
                self.wav_gatekeeper.start()
            except Exception as e:
                logger.error(
                    "Failed to start WavGatekeeper component %s: %s",
                    self.wav_gatekeeper.name,
                    e,
                )
                raise ConfigError(
                    f"Failed to start WavGatekeeper component {self.wav_gatekeeper.name}: {e}"
                ) from e

        for notif in self.notifiers:
            try:
                notif.start()
            except Exception as e:
                logger.error(
                    "Failed to start TransmissionNotifier component %s: %s",
                    notif.name,
                    e,
                )
                raise ConfigError(
                    f"Failed to start TransmissionNotifier component {notif.name}: {e}"
                ) from e

    def has_wav_component(self) -> bool:
        """Return True if an active WavGatekeeper component is loaded."""
        return self.wav_gatekeeper is not None

    def process_wav(self, wav_path: str, channel_info: ChannelInfo) -> ComponentResult:
        """Execute WavGatekeeper.process() with timeout and fault isolation."""
        if not self.wav_gatekeeper:
            return ComponentResult(keep=True)

        gk = self.wav_gatekeeper

        if not self._executor:
            try:
                return gk.process(wav_path, channel_info)
            except Exception as e:
                logger.warning(
                    "WavGatekeeper component %s raised exception on %s: %s",
                    gk.name,
                    wav_path,
                    e,
                    exc_info=True,
                )
                return ComponentResult(keep=True, detail="WavGatekeeper error")

        try:
            future = self._executor.submit(gk.process, wav_path, channel_info)
            result = future.result(timeout=self._wav_timeout_sec)
            return result

        except FuturesTimeoutError:
            logger.warning(
                "WavGatekeeper component %s timed out after %.1f s on %s. Keeping file.",
                gk.name,
                self._wav_timeout_sec,
                wav_path,
            )
            gk.recover()
            return ComponentResult(keep=True, detail="WavGatekeeper timeout")
        except Exception as e:
            logger.warning(
                "WavGatekeeper component %s raised exception on %s: %s. Keeping file.",
                gk.name,
                wav_path,
                e,
                exc_info=True,
            )
            gk.recover()
            return ComponentResult(keep=True, detail="WavGatekeeper error")

    async def dispatch_transmission(self, record: TransmissionRecord) -> None:
        """Concurrently dispatch TransmissionRecord to all configured TransmissionNotifiers."""
        if not self.notifiers:
            return

        async def _safe_dispatch(notifier: TransmissionNotifier) -> None:
            try:
                await notifier.on_transmission(record)
            except Exception as e:
                logger.warning(
                    "TransmissionNotifier component %s raised exception on transmission: %s",
                    notifier.name,
                    e,
                    exc_info=True,
                )

        tasks = [_safe_dispatch(n) for n in self.notifiers]
        _ = await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all_async(self) -> None:
        """Asynchronously stop all components and clean up thread pools.

        Calls each component's ``stop_async()`` (which defaults to the
        synchronous ``stop()`` for components that do not override it), so
        components requiring async teardown (e.g. an MQTT session close) get
        their graceful shutdown path on an orderly exit.
        """
        components: list[Component] = list(self.notifiers)
        if self.wav_gatekeeper:
            components.append(self.wav_gatekeeper)

        results = await asyncio.gather(
            *(component.stop_async() for component in components),
            return_exceptions=True,
        )
        for component, result in zip(components, results, strict=False):
            if isinstance(result, BaseException):
                logger.warning(
                    "Error stopping component %s asynchronously: %s",
                    component.name,
                    result,
                )

        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
