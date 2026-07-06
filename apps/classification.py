#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 2023

@author: john
"""

import os
import sys
import logging
import subprocess
import select
import threading
import time
from pathlib import Path
from typing import Dict, Literal, Optional
from dataclasses import dataclass

@dataclass(kw_only=True)
class ClassifierParams:
    '''
    Holds classifier command line options provided by the user
    '''
    wanted: Dict[Literal['V', 'D', 'S'], bool]
    model_file_name: Path

class ClassificationNotWanted(Exception):
    pass

class Classifier(object):

    def __init__(self, params: ClassifierParams, audio_rate: int):
        self.params = params
        self.audio_rate = audio_rate
        self._proc: Optional[subprocess.Popen[str]] = None
        self._loaded = False

        if all(value is False for value in self.params.wanted.values()):
            raise ClassificationNotWanted()

        # Resolve model path
        if self.params.model_file_name is None:
            raise FileNotFoundError("Model file not specified.")

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

        logging.info("Starting background TensorFlow classification service subprocess...")

        # bufsize=1 ensures line-buffering so that readline and write lines flush immediately.
        self._proc = subprocess.Popen(
            [sys.executable, service_script.as_posix(),
             "--model", self.model_path.absolute().as_posix(),
             "--audio_rate", str(self.audio_rate)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Read the ready signal to ensure the subprocess initialized correctly
        if self._proc is None or self._proc.stdout is None or self._proc.stderr is None:
            raise RuntimeError("Classification service failed to initialize.")

        ready = ""
        # Check if stdout has data to avoid blocking if the service crashed instantly.
        # We wait up to 120 seconds for the service to start, checking in intervals
        # to see if the process exited early.
        timeout = 120.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            r, _, _ = select.select([self._proc.stdout], [], [], 0.5)
            if r:
                ready = self._proc.stdout.readline().strip()
                break
            # Check if the process exited early
            if self._proc.poll() is not None:
                break

        if ready != "READY":
            # Retrieve available startup error traceback details from stderr
            err_msg = ""
            try:
                er, _, _ = select.select([self._proc.stderr], [], [], 5.0)
                if er:
                    fd = self._proc.stderr.fileno()
                    # Set non-blocking so read() returns immediately if select() was a spurious wakeup.
                    # Note: this leaves the fd non-blocking, which is safe since the subprocess
                    # is about to be cleaned up/terminated due to startup failure.
                    os.set_blocking(fd, False)
                    try:
                        err_bytes = os.read(fd, 4096)
                        err_msg = err_bytes.decode('utf-8', errors='replace')
                    except BlockingIOError:
                        pass
            except Exception:
                pass
            raise RuntimeError(
                f"Classification service failed to initialize.\n"
                f"Handshake status: {ready!r}\n"
                f"Subprocess stderr:\n{err_msg}"
            )

        # Start a daemon background thread to consume service's stderr
        # and forward messages to the logs, avoiding pipe-buffer block deadlocks.
        def log_stderr_stream(pipe):
            try:
                for line in pipe:
                    stripped = line.strip()
                    if stripped:
                        if stripped.startswith("INFO:"):
                            logging.info(f"TensorFlow Service: {stripped}")
                        else:
                            logging.warning(f"TensorFlow Service: {stripped}")
            except Exception as e:
                try:
                    logging.debug(f"TensorFlow Service stderr drain thread exception: {e}")
                except Exception:
                    pass
            finally:
                try:
                    logging.debug("TensorFlow Service stderr drain thread exiting.")
                except Exception:
                    pass

        t = threading.Thread(target=log_stderr_stream, args=(self._proc.stderr,), daemon=True)
        t.start()

        logging.info("TensorFlow classification service started successfully!")
        self._loaded = True

    def is_wanted(self, file: str) -> tuple[bool, Optional[Literal['V', 'D', 'S']]]:
        # Self-healing: restart the service if it crashed or is not running
        if (self._proc is None
                or self._proc.poll() is not None
                or self._proc.stdin is None
                or self._proc.stdout is None):
            logging.warning("TensorFlow classification service is not running. Attempting restart...")
            self.clean_up()
            try:
                self._ensure_loaded()
            except Exception as e:
                logging.error(f"Failed to restart TensorFlow classification service: {e}")
                return False, None

        if (self._proc is None
                or self._proc.poll() is not None
                or self._proc.stdin is None
                or self._proc.stdout is None):
            return False, None

        # Write filename to process stdin
        self._proc.stdin.write(file + "\n")
        self._proc.stdin.flush()

        # Read prediction response from process stdout with a timeout to avoid hangs
        ready_r, _, _ = select.select([self._proc.stdout], [], [], 5.0)
        if not ready_r:
            logging.error(f"Timeout waiting for classification response for file: {file}")
            logging.warning(f"Classification result lost for {file} due to subprocess hang.")
            self.clean_up()  # Terminate hung subprocess
            return False, None

        line = self._proc.stdout.readline().strip()
        if not line:
            return False, None

        detected_as = line if line in ('V', 'D', 'S') else None
        wanted = bool(detected_as and self.params.wanted[detected_as])
        return wanted, detected_as

    def clean_up(self) -> None:
        if not self._proc:
            return

        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logging.warning("Classification service did not terminate in time. Killing it...")
                self._proc.kill()
                self._proc.wait()
        except Exception as e:
            logging.debug(f"Error during classification service cleanup: {e}")
        finally:
            self._proc = None
            self._loaded = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clean_up()

    def __del__(self):
        self.clean_up()

def main():
    """Test the classifier

    Sets up the classifier
    Classifies a couple of audio files for testing purposes
    """

    audio_rate = 8000
    model_file_name = Path(__file__).parent / 'model' / 'model_1.tflite'
    classifier_params = ClassifierParams(
        wanted={'V': True,
                'D': True,
                'S': True
        },
        model_file_name=model_file_name
    )

    try:
        classifier = Classifier(classifier_params, audio_rate)
    except Exception as error:
        raise Exception(f'Could not create classifier ({error})')

    print(f'should be voice (V) got {classifier.is_wanted("test/voice.wav")[1]}')
    print(f'should be data (D) got {classifier.is_wanted("test/data.wav")[1]}')
    print(f'should be skip (S) got {classifier.is_wanted("test/skip.wav")[1]}')
    classifier.clean_up()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
