import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Generator, Tuple
from unittest.mock import MagicMock

import pytest
from demodulators.BaseTuner import BaseTuner
from gnuradio import gr  # type: ignore

# Mock tensorflow to allow importing classification.py/BaseTuner without tensorflow installed
sys.modules["tensorflow"] = MagicMock()


class MockTuner(BaseTuner):
    def __init__(
        self,
        classify: None,
        notify_scanner: Callable,
        file_metadata: list[str] | None = None,
        get_priority_info: Callable[[int], Tuple[int | None, bool]] | None = None,
    ) -> None:
        gr.hier_block2.__init__(
            self,
            "MockTuner",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(1, 1, gr.sizeof_float),
        )
        super().__init__(classify, notify_scanner, file_metadata, get_priority_info)
        self.record = True
        self.audio_bps = 8
        self.min_recording = 0.0
        self.center_freq = 1000

        class MockWavSink:
            def close(self) -> None:
                pass

        self.blocks_wavfile_sink = MockWavSink()

    def set_temp_file(self, temp_file: str) -> None:
        """Simulate the state set_file_name would establish for a given tmp path."""
        basename = os.path.basename(temp_file).removesuffix(".wav")
        self.freq_str, self.tstamp_str = basename.split("_", 1)
        self.file_name = temp_file


@pytest.fixture(autouse=True)
def setup_teardown_wav_dirs() -> Generator[None, None, None]:
    # Helper to clean up only test wav files created by tests in this module (using the test timestamp)
    def cleanup_test_files():
        for f in Path("wav").glob("*_20260531_160622.345.wav"):
            if f.is_file():
                f.unlink()
        if os.path.exists("wav/tmp"):
            shutil.rmtree("wav/tmp")

    # Clean up before run to ensure clean state
    cleanup_test_files()
    os.makedirs("wav/tmp", exist_ok=True)

    yield

    # Clean up after run
    cleanup_test_files()


def test_persist_filename_no_metadata() -> None:
    # Test default behaviour without any metadata requested
    tuner = MockTuner(None, lambda msg: None, file_metadata=[])

    # Simulate a recording file
    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)  # write dummy bytes

    tuner.set_temp_file(temp_file)

    msg = tuner._persist_wavfile(rf_center_freq=460000000)
    assert msg is not None
    assert msg.file == "wav/460.1250_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_20260531_160622.345.wav")


def test_persist_filename_with_priority() -> None:
    # Mock priority callback
    def get_priority_info(bb: int) -> Tuple[int | None, bool]:
        return 2, False  # Priority 2, not auto

    tuner = MockTuner(
        None,
        lambda msg: None,
        file_metadata=["priority"],
        get_priority_info=get_priority_info,
    )

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = tuner._persist_wavfile(rf_center_freq=460000000)
    assert msg is not None
    assert msg.file == "wav/460.1250_P2_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_P2_20260531_160622.345.wav")


def test_persist_filename_with_auto_priority() -> None:
    def get_priority_info(bb: int) -> Tuple[int | None, bool]:
        return 1, True  # Priority 1, auto

    tuner = MockTuner(
        None,
        lambda msg: None,
        file_metadata=["priority"],
        get_priority_info=get_priority_info,
    )

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = tuner._persist_wavfile(rf_center_freq=460000000)
    assert msg is not None
    assert msg.file == "wav/460.1250_PA_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_PA_20260531_160622.345.wav")


def test_persist_filename_with_strength() -> None:
    tuner = MockTuner(None, lambda msg: None, file_metadata=["strength"])

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = tuner._persist_wavfile(rf_center_freq=460000000, avg_signal=-50)
    assert msg is not None
    assert msg.file == "wav/460.1250_-50dB_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_-50dB_20260531_160622.345.wav")


def test_persist_filename_with_priority_and_strength() -> None:
    def get_priority_info(bb: int) -> Tuple[int | None, bool]:
        return 1, False

    tuner = MockTuner(
        None,
        lambda msg: None,
        file_metadata=["priority", "strength"],
        get_priority_info=get_priority_info,
    )

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = tuner._persist_wavfile(rf_center_freq=460000000, avg_signal=-51)
    assert msg is not None
    assert msg.file == "wav/460.1250_P1_-51dB_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_P1_-51dB_20260531_160622.345.wav")


def test_persist_filename_no_metadata_with_classification() -> None:
    # Test new breaking change naming format when classification is enabled and no metadata flags are specified
    class MockClassifier:
        def is_wanted(self, filename: str) -> Tuple[bool, str]:
            return True, "V"

    tuner = MockTuner(MockClassifier(), lambda msg: None, file_metadata=[])

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = tuner._persist_wavfile(rf_center_freq=460000000)
    assert msg is not None
    assert msg.file == "wav/460.1250_V_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_V_20260531_160622.345.wav")


def test_persist_filename_no_active_file() -> None:
    tuner = MockTuner(None, lambda msg: None, file_metadata=[])
    tuner.file_name = None
    msg = tuner._persist_wavfile(rf_center_freq=460000000)
    assert msg is None


def test_persist_filename_short_recording() -> None:
    tuner = MockTuner(None, lambda msg: None, file_metadata=[])
    tuner.min_recording = 5.0  # min_size will be 44 + 8 * 1000 * 5.0 = 40044 bytes

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)  # 100 bytes is shorter than min_size

    tuner.set_temp_file(temp_file)

    msg = tuner._persist_wavfile(rf_center_freq=460000000)
    assert msg is not None
    assert msg.file is None
    assert msg.detail == "Discarded short recording"
    assert not os.path.exists(temp_file)


def test_persist_filename_unwanted_classification() -> None:
    # Test discarding when classification indicates it is unwanted (using 'S' for static/noise)
    class MockClassifier:
        def is_wanted(self, filename: str) -> Tuple[bool, str]:
            return False, "S"

    tuner = MockTuner(MockClassifier(), lambda msg: None, file_metadata=[])

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = tuner._persist_wavfile(rf_center_freq=460000000)
    assert msg is not None
    assert msg.file is None
    assert msg.classification == "S"
    assert msg.detail == "Discarded unwanted classification"
    assert not os.path.exists(temp_file)


def test_persist_filename_priority_none() -> None:
    # Test that when 'priority' metadata is requested, but get_priority_info returns priority=None
    def get_priority_info(bb: int) -> Tuple[int | None, bool]:
        return None, False

    tuner = MockTuner(
        None,
        lambda msg: None,
        file_metadata=["priority"],
        get_priority_info=get_priority_info,
    )

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = tuner._persist_wavfile(rf_center_freq=460000000)
    assert msg is not None
    assert msg.file == "wav/460.1250_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_20260531_160622.345.wav")


def test_persist_filename_strength_metadata_none_signal() -> None:
    # Test that when 'strength' metadata is requested, but avg_signal is None
    tuner = MockTuner(None, lambda msg: None, file_metadata=["strength"])

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = tuner._persist_wavfile(rf_center_freq=460000000, avg_signal=None)
    assert msg is not None
    assert msg.file == "wav/460.1250_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_20260531_160622.345.wav")


def test_persist_filename_with_ctcss() -> None:
    # Test that when 'ctcss' metadata is requested and a tone is matched, it is appended to the filename
    tuner = MockTuner(None, lambda msg: None, file_metadata=["ctcss"])
    tuner.matched_ctcss_tone = 103.5

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = tuner._persist_wavfile(rf_center_freq=460000000)
    assert msg is not None
    assert msg.file == "wav/460.1250_103.5Hz_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_103.5Hz_20260531_160622.345.wav")


def test_persist_filename_with_ctcss_none() -> None:
    # Test that when 'ctcss' metadata is requested, but matched_ctcss_tone is None, the ctcss segment is omitted
    tuner = MockTuner(None, lambda msg: None, file_metadata=["ctcss"])
    tuner.matched_ctcss_tone = None

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = tuner._persist_wavfile(rf_center_freq=460000000)
    assert msg is not None
    assert msg.file == "wav/460.1250_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_20260531_160622.345.wav")

