"""
TFLite Audio Classification Subprocess Controller.
"""

import logging
import os
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal

from typing_extensions import Self

logger = logging.getLogger(f"ham2mon.{__name__}")


@dataclass(kw_only=True)
class ClassifierParams:
    """Holds classifier command line options provided by the user."""

    wanted: dict[Literal["V", "D", "S"], bool]
    model_file_name: Path


class ClassificationNotWanted(Exception):
    """Raised when no categories (V, D, S) are requested for classification."""


class Classifier:
    """Orchestrates background subprocess for TFLite audio classification."""

    params: ClassifierParams
    audio_rate: int
    model_path: Path
    _proc: subprocess.Popen[str] | None
    _loaded: bool

    def __init__(self, params: ClassifierParams, audio_rate: int) -> None:
        self.params = params
        self.audio_rate = audio_rate
        self._proc = None
        self._loaded = False

        if all(value is False for value in self.params.wanted.values()):
            raise ClassificationNotWanted()

        # Resolve model path
        self.model_path = Path(self.params.model_file_name)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")
        self.model_path = self.model_path.resolve(strict=True)

        # Eagerly start the subprocess and load TensorFlow at construction time
        # to ensure classification is ready before transmissions begin.
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        # Start classification service as a subprocess
        script_dir = Path(__file__).parent
        service_script = script_dir / "classification_service.py"

        logger.info(
            "Starting background TensorFlow classification service subprocess..."
        )

        # bufsize=1 ensures line-buffering so that readline and write lines flush immediately.
        proc = subprocess.Popen(
            [
                sys.executable,
                service_script.as_posix(),
                "--model",
                self.model_path.absolute().as_posix(),
                "--audio-rate",
                str(self.audio_rate),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._proc = proc

        # Read the ready signal to ensure the subprocess initialized correctly
        if proc.stdout is None or proc.stderr is None:
            raise RuntimeError("Classification service failed to initialize.")

        proc_stdout: IO[str] = proc.stdout
        proc_stderr: IO[str] = proc.stderr

        ready: str = ""
        timeout = 120.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            r, _, _ = select.select([proc_stdout], [], [], 0.5)
            if r:
                ready = proc_stdout.readline().strip()
                break
            # Check if the process exited early
            if proc.poll() is not None:
                break

        if ready != "READY":
            # Retrieve available startup error traceback details from stderr
            err_msg = ""
            try:
                er, _, _ = select.select([proc_stderr], [], [], 5.0)
                if er:
                    fd = proc_stderr.fileno()
                    os.set_blocking(fd, False)
                    try:
                        err_bytes = os.read(fd, 4096)
                        err_msg = err_bytes.decode("utf-8", errors="replace")
                    except BlockingIOError:
                        pass
            except (OSError, RuntimeError):
                pass
            raise RuntimeError(
                "Classification service failed to initialize.\n"
                + f"Handshake status: {ready!r}\n"
                + f"Subprocess stderr:\n{err_msg}"
            )

        # Start a daemon background thread to consume service's stderr
        # and forward messages to the logs, avoiding pipe-buffer block deadlocks.
        try:
            os.set_blocking(proc_stderr.fileno(), True)
        except OSError:
            pass

        def log_stderr_stream(pipe: IO[str]) -> None:
            try:
                for line in pipe:
                    stripped = line.strip()
                    if stripped:
                        if stripped.startswith("INFO:"):
                            logger.info("TensorFlow Service: %s", stripped)
                        else:
                            logger.warning("TensorFlow Service: %s", stripped)
            except (OSError, ValueError) as e:
                try:
                    logger.debug("TensorFlow Service stderr drain thread exception: %s", e)
                except Exception:  # noqa: BLE001, S110
                    pass
            finally:
                try:
                    logger.debug("TensorFlow Service stderr drain thread exiting.")
                except Exception:  # noqa: BLE001, S110
                    pass

        t = threading.Thread(
            target=log_stderr_stream, args=(proc_stderr,), daemon=True
        )
        t.start()

        logger.info("TensorFlow classification service started successfully!")
        self._loaded = True

    def is_wanted(self, file: str) -> tuple[bool, Literal["V", "D", "S"] | None]:
        """Classify given audio file and return (is_wanted, detected_label)."""
        proc = self._proc
        # Self-healing: restart the service if it crashed or is not running
        if (
            proc is None
            or proc.poll() is not None
            or proc.stdin is None
            or proc.stdout is None
        ):
            logger.warning(
                "TensorFlow classification service is not running. Attempting restart..."
            )
            self.clean_up()
            try:
                self._ensure_loaded()
            except (RuntimeError, OSError) as e:
                logger.error("Failed to restart TensorFlow classification service: %s", e)
                return False, None
            proc = self._proc

        if (
            proc is None
            or proc.poll() is not None
            or proc.stdin is None
            or proc.stdout is None
        ):
            return False, None

        proc_stdin: IO[str] = proc.stdin
        proc_stdout: IO[str] = proc.stdout

        # Write filename to process stdin
        _ = proc_stdin.write(file + "\n")
        proc_stdin.flush()

        # Read prediction response from process stdout with a timeout to avoid hangs
        ready_r, _, _ = select.select([proc_stdout], [], [], 5.0)
        if not ready_r:
            logger.error("Timeout waiting for classification response for file: %s", file)
            logger.warning("Classification result lost for %s due to subprocess hang.", file)
            self.clean_up()  # Terminate hung subprocess
            return False, None

        raw_line: str = proc_stdout.readline()
        line = raw_line.strip()
        if not line:
            return False, None

        detected_as: Literal["V", "D", "S"] | None = (
            line if line in ("V", "D", "S") else None  # type: ignore[assignment]
        )
        wanted = bool(detected_as and self.params.wanted[detected_as])
        return wanted, detected_as

    def clean_up(self) -> None:
        """Terminate and clean up classification subprocess."""
        proc = self._proc
        if not proc:
            return

        try:
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
            try:
                _ = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Classification service did not terminate in time. Killing it..."
                )
                proc.kill()
                _ = proc.wait()
        except (OSError, RuntimeError) as e:
            logger.debug("Error during classification service cleanup: %s", e)
        finally:
            self._proc = None
            self._loaded = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        self.clean_up()

    def __del__(self) -> None:
        self.clean_up()
