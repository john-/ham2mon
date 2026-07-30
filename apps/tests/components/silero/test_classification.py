"""Tests for Silero VAD WavGatekeeper component (ONNX)."""

import wave
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from components.base import ChannelInfo
from components.silero.component import SileroVadComponent


@pytest.fixture
def dummy_wav(tmp_path: Path) -> Path:
    """Generate a dummy WAV audio file for testing."""
    wav_file = tmp_path / "test_voice.wav"
    sample_rate = 16000
    duration_sec = 0.5
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(str(wav_file), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio_data.tobytes())

    return wav_file


@pytest.fixture
def channel_info(dummy_wav: Path) -> ChannelInfo:
    """Create test ChannelInfo instance."""
    return ChannelInfo(
        rf=462.6375,
        bb_hz=5000,
        channel=0,
        label="FRS 4",
        priority=None,
        matched_ctcss_hz=74.4,
        signal_db=-45,
        timestamp=1000000.0,
        wav_tmp_path=str(dummy_wav),
    )


def test_silero_component_start_missing_model(tmp_path: Path) -> None:
    """Verify FileNotFoundError raised when model path does not exist."""
    comp = SileroVadComponent(
        {"model_path": str(tmp_path / "nonexistent.onnx"), "threshold": 0.5}
    )
    with pytest.raises(FileNotFoundError):
        comp.start()


def test_silero_component_threshold_bool_fallback() -> None:
    """Verify boolean threshold in config falls back to default 0.5 instead of 1.0."""
    comp = SileroVadComponent({"threshold": True, "min_voice_chunks": "3", "max_eval_sec": 2.5})
    assert comp.threshold == 0.5
    assert comp.min_voice_chunks == 3
    assert comp.max_eval_sec == 2.5


def test_silero_component_package_reexport() -> None:
    """Verify SileroVadComponent is re-exported from components.silero subpackage."""
    from components.silero import SileroVadComponent as ReexportedComponent

    assert ReexportedComponent is SileroVadComponent


@patch("components.silero.component.OnnxSessionClass")
def test_silero_component_process_voice(
    mock_session_cls: MagicMock, dummy_wav: Path, channel_info: ChannelInfo, tmp_path: Path
) -> None:
    """Verify SileroVadComponent returns keep=True (Voice) when probability exceeds threshold."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_model = tmp_path / "mock_silero.onnx"
    _ = mock_model.write_bytes(b"dummy_onnx_model")

    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    mock_session.run.return_value = [np.array([[0.85]], dtype=np.float32), h, c]  # pyright: ignore[reportAny]

    comp = SileroVadComponent(
        {
            "model_path": str(mock_model),
            "threshold": 0.5,
            "min_voice_chunks": 1,
        }
    )
    comp.start()

    res = comp.process(str(dummy_wav), channel_info)

    assert res.keep is True
    assert res.classification == "V"
    assert res.metadata.get("vad_prob") == pytest.approx(0.85)  # pyright: ignore[reportUnknownMemberType]


@patch("components.silero.component.OnnxSessionClass")
def test_silero_component_process_noise(
    mock_session_cls: MagicMock, dummy_wav: Path, channel_info: ChannelInfo, tmp_path: Path
) -> None:
    """Verify SileroVadComponent returns keep=False (Skip) when probability is below threshold."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_model = tmp_path / "mock_silero.onnx"
    _ = mock_model.write_bytes(b"dummy_onnx_model")

    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    mock_session.run.return_value = [np.array([[0.10]], dtype=np.float32), h, c]  # pyright: ignore[reportAny]

    comp = SileroVadComponent(
        {
            "model_path": str(mock_model),
            "threshold": 0.5,
        }
    )
    comp.start()

    res = comp.process(str(dummy_wav), channel_info)

    assert res.keep is False
    assert res.classification == "S"
    assert "No voice detected" in (res.detail or "")


@patch("components.silero.component.OnnxSessionClass")
def test_silero_component_process_resampling(
    mock_session_cls: MagicMock, tmp_path: Path, channel_info: ChannelInfo
) -> None:
    """Verify SileroVadComponent resamples audio with non-standard framerate (e.g. 22050 Hz)."""
    wav_file = tmp_path / "non_standard_rate.wav"
    sample_rate = 22050
    duration_sec = 0.5
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(str(wav_file), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio_data.tobytes())

    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_model = tmp_path / "mock_silero.onnx"
    _ = mock_model.write_bytes(b"dummy_onnx_model")

    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    mock_session.run.return_value = [np.array([[0.90]], dtype=np.float32), h, c]  # pyright: ignore[reportAny]

    comp = SileroVadComponent(
        {
            "model_path": str(mock_model),
            "threshold": 0.5,
            "min_voice_chunks": 1,
        }
    )
    comp.start()

    res = comp.process(str(wav_file), channel_info)

    assert res.keep is True
    assert res.classification == "V"
    # Verify model was called with 16000 Hz sr input tensor
    called_inputs = cast(dict[str, np.ndarray], mock_session.run.call_args[0][1])  # pyright: ignore[reportAny]
    assert called_inputs["sr"][0] == 16000


@patch("components.silero.component.OnnxSessionClass")
def test_silero_component_process_partial_chunk(
    mock_session_cls: MagicMock, tmp_path: Path, channel_info: ChannelInfo
) -> None:
    """Verify SileroVadComponent evaluates partial trailing chunk by zero-padding."""
    wav_file = tmp_path / "partial_chunk.wav"
    sample_rate = 16000
    # 600 samples = 512 samples (1 full chunk) + 88 samples (partial chunk)
    t = np.linspace(0, 600 / sample_rate, 600, endpoint=False)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(str(wav_file), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio_data.tobytes())

    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_model = tmp_path / "mock_silero.onnx"
    _ = mock_model.write_bytes(b"dummy_onnx_model")

    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    mock_session.run.return_value = [np.array([[0.80]], dtype=np.float32), h, c]  # pyright: ignore[reportAny]

    comp = SileroVadComponent(
        {"model_path": str(mock_model), "threshold": 0.5, "min_voice_chunks": 1}
    )
    comp.start()

    res = comp.process(str(wav_file), channel_info)

    assert res.keep is True
    # 600 samples zero-padded to 1024 samples = exactly 2 chunks evaluated
    call_cnt = cast(int, mock_session.run.call_count)  # pyright: ignore[reportAny]
    assert call_cnt == 2


@patch("components.silero.component.OnnxSessionClass")
def test_silero_component_process_corrupt_wav_fail_open(
    mock_session_cls: MagicMock, tmp_path: Path, channel_info: ChannelInfo
) -> None:
    """Verify corrupted/unreadable WAV causes fail-open behavior (keep=True)."""
    corrupt_wav = tmp_path / "corrupt.wav"
    _ = corrupt_wav.write_bytes(b"NOT_A_REAL_WAV_FILE_HEADER")

    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_model = tmp_path / "mock_silero.onnx"
    _ = mock_model.write_bytes(b"dummy_onnx_model")

    comp = SileroVadComponent(
        {"model_path": str(mock_model), "threshold": 0.5}
    )
    comp.start()

    res = comp.process(str(corrupt_wav), channel_info)

    # Fail-open guard must keep the transmission
    assert res.keep is True
    assert res.detail == "VAD evaluation error"


@patch("components.silero.component.OnnxSessionClass")
def test_silero_component_process_stereo_downmix(
    mock_session_cls: MagicMock, tmp_path: Path, channel_info: ChannelInfo
) -> None:
    """Verify stereo (2 channel) WAV file is downmixed to mono before inference."""
    wav_file = tmp_path / "stereo.wav"
    sample_rate = 16000
    t = np.linspace(0, 0.1, int(sample_rate * 0.1), endpoint=False)
    left = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right = (np.cos(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    stereo_data = np.column_stack((left, right)).flatten()

    with wave.open(str(wav_file), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(stereo_data.tobytes())

    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_model = tmp_path / "mock_silero.onnx"
    _ = mock_model.write_bytes(b"dummy_onnx_model")

    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    mock_session.run.return_value = [np.array([[0.80]], dtype=np.float32), h, c]  # pyright: ignore[reportAny]

    comp = SileroVadComponent(
        {"model_path": str(mock_model), "threshold": 0.5, "min_voice_chunks": 1}
    )
    comp.start()

    res = comp.process(str(wav_file), channel_info)

    assert res.keep is True
    assert res.classification == "V"


@patch("components.silero.component.OnnxSessionClass")
def test_silero_component_process_unsupported_sample_width(
    mock_session_cls: MagicMock, tmp_path: Path, channel_info: ChannelInfo
) -> None:
    """Verify unsupported sample width (e.g. 24-bit PCM / sampwidth=3) fails open."""
    wav_file = tmp_path / "24bit.wav"
    sample_rate = 16000
    dummy_24bit_pcm = bytes([0] * (sample_rate * 3 // 10))

    with wave.open(str(wav_file), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(sample_rate)
        w.writeframes(dummy_24bit_pcm)

    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_model = tmp_path / "mock_silero.onnx"
    _ = mock_model.write_bytes(b"dummy_onnx_model")

    comp = SileroVadComponent(
        {"model_path": str(mock_model), "threshold": 0.5}
    )
    comp.start()

    res = comp.process(str(wav_file), channel_info)

    assert res.keep is True
    assert res.detail == "VAD evaluation error"


@patch("components.silero.component.OnnxSessionClass")
def test_silero_component_process_8bit_and_32bit_pcm(
    mock_session_cls: MagicMock, tmp_path: Path, channel_info: ChannelInfo
) -> None:
    """Verify 8-bit uint8 and 32-bit int32 PCM audio are decoded correctly."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_model = tmp_path / "mock_silero.onnx"
    _ = mock_model.write_bytes(b"dummy_onnx_model")

    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    mock_session.run.return_value = [np.array([[0.80]], dtype=np.float32), h, c]  # pyright: ignore[reportAny]

    comp = SileroVadComponent(
        {"model_path": str(mock_model), "threshold": 0.5}
    )
    comp.start()

    # Test 8-bit uint8
    wav_8bit = tmp_path / "8bit.wav"
    data_8bit = bytes([128] * 1600)
    with wave.open(str(wav_8bit), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(16000)
        w.writeframes(data_8bit)
    res_8bit = comp.process(str(wav_8bit), channel_info)
    assert res_8bit.keep is True

    # Test 32-bit int32
    wav_32bit = tmp_path / "32bit.wav"
    data_32bit = np.zeros(1600, dtype=np.int32).tobytes()
    with wave.open(str(wav_32bit), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(4)
        w.setframerate(16000)
        w.writeframes(data_32bit)
    res_32bit = comp.process(str(wav_32bit), channel_info)
    assert res_32bit.keep is True


@patch("components.silero.component.OnnxSessionClass")
def test_silero_component_process_empty_audio(
    mock_session_cls: MagicMock, tmp_path: Path, channel_info: ChannelInfo
) -> None:
    """Verify empty WAV file (0 frames) returns keep=False with 'Empty audio file' detail."""
    empty_wav = tmp_path / "empty.wav"
    with wave.open(str(empty_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"")

    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_model = tmp_path / "mock_silero.onnx"
    _ = mock_model.write_bytes(b"dummy_onnx_model")

    comp = SileroVadComponent(
        {"model_path": str(mock_model), "threshold": 0.5}
    )
    comp.start()

    res = comp.process(str(empty_wav), channel_info)

    assert res.keep is False
    assert res.classification == "S"
    assert res.detail == "Empty audio file"
