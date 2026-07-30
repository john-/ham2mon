import os
import shutil
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from config import MasterHam2MonConfig
from demodulators.BaseTuner import BaseTuner
from gnuradio import gr  # type: ignore
from scanner import Scanner

# Mock tensorflow to allow importing classification.py/BaseTuner without tensorflow installed
sys.modules["tensorflow"] = MagicMock()


import time


class MockTuner(BaseTuner):
    def __init__(
        self,
        notify_scanner: Callable = lambda msg: None,
    ) -> None:
        gr.hier_block2.__init__(
            self,
            "MockTuner",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(1, 1, gr.sizeof_float),
        )
        super().__init__(notify_scanner)
        self.record = True
        self.center_freq = 125000

        class MockWavSink:
            def close(self) -> None:
                pass

        self.blocks_wavfile_sink = MockWavSink()

    def set_temp_file(self, temp_file: str) -> None:
        """Simulate the state set_file_name would establish for a given tmp path."""
        basename = os.path.basename(temp_file).removesuffix(".wav")
        parts = basename.split("_")
        self.freq_str = parts[0]
        self.tstamp_str = f"{parts[1]}_{parts[2]}"
        try:
            t_struct = time.strptime(f"{parts[1]}_{parts[2].split('.')[0]}", "%Y%m%d_%H%M%S")
            millis = float("0." + parts[2].split(".")[1])
            self.time_stamp = time.mktime(t_struct) + millis
        except (ValueError, IndexError):
            self.time_stamp = time.time()
        self.file_name = temp_file



from conftest import make_test_scanner


def create_test_scanner(
    file_metadata: list[str] | None = None,
    min_recording_sec: float = 0.0,
    classifier: MagicMock | None = None,
    get_priority_info: Callable[[int], tuple[int | None, bool]] | None = None,
) -> Scanner:
    cfg = MasterHam2MonConfig()
    cfg.audio.file_metadata = file_metadata if file_metadata is not None else []
    cfg.audio.min_recording_sec = min_recording_sec
    cfg.audio.bit_depth = 8
    return make_test_scanner(
        config=cfg,
        wav_dir="wav",
        classifier=classifier,
        get_priority_info=get_priority_info,
    )




def process_recording(
    tuner: MockTuner,
    scanner: Scanner,
    rf_center_freq: int = 460000000,
    avg_signal: int | None = None,
):
    msg = tuner._close_recording(rf_center_freq=rf_center_freq, avg_signal=avg_signal)
    if msg is not None and msg.wav_tmp_path is not None:
        msg, _ = scanner._process_completed_transmission(msg)
    return msg


@pytest.fixture(autouse=True)
def setup_teardown_wav_dirs() -> Generator[None, None, None]:
    # Helper to clean up only test wav files created by tests in this module (using the test timestamp)
    def cleanup_test_files():
        for f in Path("wav").glob("*_20260531_160622.345.wav"):
            if f.is_file():
                f.unlink()
        for f in Path("wav").glob("*_20260719_163000.000.wav"):
            if f.is_file():
                f.unlink()
        if os.path.exists("wav/tmp"):
            shutil.rmtree("wav/tmp")

    cleanup_test_files()
    os.makedirs("wav/tmp", exist_ok=True)

    yield

    cleanup_test_files()


def test_persist_filename_no_metadata() -> None:
    scanner = create_test_scanner(file_metadata=[])
    tuner = MockTuner()

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = process_recording(tuner, scanner, rf_center_freq=460000000)
    assert msg is not None
    assert msg.file == "wav/460.1250_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_20260531_160622.345.wav")


def test_persist_filename_with_priority() -> None:
    def get_priority_info(bb: int) -> tuple[int | None, bool]:
        return 2, False  # Priority 2, not auto

    scanner = create_test_scanner(file_metadata=["priority"], get_priority_info=get_priority_info)
    tuner = MockTuner()

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = process_recording(tuner, scanner, rf_center_freq=460000000)
    assert msg is not None
    assert msg.file == "wav/460.1250_P2_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_P2_20260531_160622.345.wav")


def test_persist_filename_with_auto_priority() -> None:
    def get_priority_info(bb: int) -> tuple[int | None, bool]:
        return 1, True  # Priority 1, auto

    scanner = create_test_scanner(file_metadata=["priority"], get_priority_info=get_priority_info)
    tuner = MockTuner()

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = process_recording(tuner, scanner, rf_center_freq=460000000)
    assert msg is not None
    assert msg.file == "wav/460.1250_PA_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_PA_20260531_160622.345.wav")


def test_persist_filename_with_strength() -> None:
    scanner = create_test_scanner(file_metadata=["strength"])
    tuner = MockTuner()

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = process_recording(tuner, scanner, rf_center_freq=460000000, avg_signal=-50)
    assert msg is not None
    assert msg.file == "wav/460.1250_-50dB_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_-50dB_20260531_160622.345.wav")


def test_persist_filename_with_priority_and_strength() -> None:
    def get_priority_info(bb: int) -> tuple[int | None, bool]:
        return 1, False

    scanner = create_test_scanner(file_metadata=["priority", "strength"], get_priority_info=get_priority_info)
    tuner = MockTuner()

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = process_recording(tuner, scanner, rf_center_freq=460000000, avg_signal=-51)
    assert msg is not None
    assert msg.file == "wav/460.1250_P1_-51dB_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_P1_-51dB_20260531_160622.345.wav")


def test_persist_filename_no_metadata_with_classification() -> None:
    # Test new breaking change naming format when classification is enabled and no metadata flags are specified
    class MockClassifier:
        def is_wanted(self, filename: str) -> tuple[bool, str]:
            return True, "V"

    scanner = create_test_scanner(file_metadata=[], classifier=MockClassifier())
    tuner = MockTuner()

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = process_recording(tuner, scanner, rf_center_freq=460000000)
    assert msg is not None
    assert msg.file == "wav/460.1250_V_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_V_20260531_160622.345.wav")


def test_persist_filename_no_active_file() -> None:
    scanner = create_test_scanner(file_metadata=[])
    tuner = MockTuner()
    tuner.file_name = None

    msg = process_recording(tuner, scanner, rf_center_freq=460000000)
    assert msg is None


def test_persist_filename_short_recording() -> None:
    scanner = create_test_scanner(file_metadata=[], min_recording_sec=5.0)
    tuner = MockTuner()

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = process_recording(tuner, scanner, rf_center_freq=460000000)
    assert msg is not None
    assert msg.file is None
    assert msg.detail == "Discarded short recording"
    assert not os.path.exists(temp_file)


def test_persist_filename_zero_recording_header_only_discard() -> None:
    scanner = create_test_scanner(file_metadata=[], min_recording_sec=0.0)
    tuner = MockTuner()

    temp_file = "wav/tmp/460.1250_20260719_163000.000.wav"
    with open(temp_file, "wb") as f:
        f.write(b"RIFF" + b"\x00" * 40)

    tuner.set_temp_file(temp_file)

    msg = process_recording(tuner, scanner, rf_center_freq=460000000)
    assert msg is not None
    assert msg.file is None
    assert msg.detail == "Discarded short recording"
    assert not os.path.exists(temp_file)


def test_persist_filename_zero_recording_with_audio_retained() -> None:
    scanner = create_test_scanner(file_metadata=[], min_recording_sec=0.0)
    tuner = MockTuner()

    temp_file = "wav/tmp/460.1250_20260719_163000.000.wav"
    with open(temp_file, "wb") as f:
        f.write(b"RIFF" + b"\x00" * 100)

    tuner.set_temp_file(temp_file)

    msg = process_recording(tuner, scanner, rf_center_freq=460000000)
    assert msg is not None
    assert msg.file is not None
    assert msg.detail is None


def test_persist_filename_unwanted_classification() -> None:
    # Test discarding when classification indicates it is unwanted (using 'S' for static/noise)
    class MockClassifier:
        def is_wanted(self, filename: str) -> tuple[bool, str]:
            return False, "S"

    scanner = create_test_scanner(file_metadata=[], classifier=MockClassifier())
    tuner = MockTuner()

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = process_recording(tuner, scanner, rf_center_freq=460000000)
    assert msg is not None
    assert msg.file is None
    assert msg.classification == "S"
    assert msg.detail == "Discarded unwanted classification"
    assert not os.path.exists(temp_file)


def test_persist_filename_priority_none() -> None:
    # Test that when 'priority' metadata is requested, but get_priority_info returns priority=None
    def get_priority_info(bb: int) -> tuple[int | None, bool]:
        return None, False

    scanner = create_test_scanner(file_metadata=["priority"], get_priority_info=get_priority_info)
    tuner = MockTuner()

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = process_recording(tuner, scanner, rf_center_freq=460000000)
    assert msg is not None
    assert msg.file == "wav/460.1250_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_20260531_160622.345.wav")


def test_persist_filename_strength_metadata_none_signal() -> None:
    # Test that when 'strength' metadata is requested, but avg_signal is None
    scanner = create_test_scanner(file_metadata=["strength"])
    tuner = MockTuner()

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = process_recording(tuner, scanner, rf_center_freq=460000000, avg_signal=None)
    assert msg is not None
    assert msg.file == "wav/460.1250_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_20260531_160622.345.wav")


def test_persist_filename_with_ctcss() -> None:
    # Test that when 'ctcss' metadata is requested and a tone is matched, it is appended to the filename
    scanner = create_test_scanner(file_metadata=["ctcss"])
    tuner = MockTuner()
    tuner.matched_ctcss_tone = 103.5

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = process_recording(tuner, scanner, rf_center_freq=460000000)
    assert msg is not None
    assert msg.file == "wav/460.1250_103.5Hz_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_103.5Hz_20260531_160622.345.wav")


def test_persist_filename_with_ctcss_none() -> None:
    # Test that when 'ctcss' metadata is requested, but matched_ctcss_tone is None, the ctcss segment is omitted
    scanner = create_test_scanner(file_metadata=["ctcss"])
    tuner = MockTuner()
    tuner.matched_ctcss_tone = None

    temp_file = "wav/tmp/460.1250_20260531_160622.345.wav"
    with open(temp_file, "wb") as f:
        f.write(b"0" * 100)

    tuner.set_temp_file(temp_file)

    msg = process_recording(tuner, scanner, rf_center_freq=460000000)
    assert msg is not None
    assert msg.file == "wav/460.1250_20260531_160622.345.wav"
    assert os.path.exists("wav/460.1250_20260531_160622.345.wav")
