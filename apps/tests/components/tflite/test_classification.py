import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from components.tflite import classification as classification_module
from components.tflite.classification import (
    ClassificationNotWanted,
    Classifier,
    ClassifierParams,
)

MODEL_PATH = Path(classification_module.__file__).parent / "model" / "model_1.tflite"
WAV_DIR = Path(__file__).resolve().parents[3] / "test"


@pytest.fixture
def classifier_params() -> ClassifierParams:
    return ClassifierParams(
        wanted={"V": True, "D": True, "S": True}, model_file_name=MODEL_PATH
    )


# --- Happy-Path & Live Model Tests ---


def test_classifier_initialization(classifier_params: ClassifierParams) -> None:
    """Test that the classifier initializes and loads the TFLite model successfully."""
    with Classifier(classifier_params, audio_rate=8000) as classifier:
        assert classifier._loaded is True
        assert classifier.audio_rate == 8000
        assert classifier._proc is not None


def test_classification_not_wanted() -> None:
    """Test that Classifier raises ClassificationNotWanted if no categories are requested."""
    params = ClassifierParams(
        wanted={"V": False, "D": False, "S": False}, model_file_name=MODEL_PATH
    )
    with pytest.raises(ClassificationNotWanted):
        _ = Classifier(params, audio_rate=8000)


def test_classify_voice(classifier_params: ClassifierParams) -> None:
    """Test classification of a voice recording."""
    with Classifier(classifier_params, audio_rate=8000) as classifier:
        wanted, detected = classifier.is_wanted(str(WAV_DIR / "voice.wav"))
        assert detected == "V"
        assert wanted is True


def test_classify_data(classifier_params: ClassifierParams) -> None:
    """Test classification of a data recording."""
    with Classifier(classifier_params, audio_rate=8000) as classifier:
        wanted, detected = classifier.is_wanted(str(WAV_DIR / "data.wav"))
        assert detected == "D"
        assert wanted is True


def test_classify_skip(classifier_params: ClassifierParams) -> None:
    """Test classification of a skip/noise recording."""
    with Classifier(classifier_params, audio_rate=8000) as classifier:
        wanted, detected = classifier.is_wanted(str(WAV_DIR / "skip.wav"))
        assert detected == "S"
        assert wanted is True


def test_classify_filtering() -> None:
    """Test that files detected as unwanted are filtered out (wanted is False)."""
    params = ClassifierParams(
        wanted={"V": True, "D": False, "S": False}, model_file_name=MODEL_PATH
    )
    with Classifier(params, audio_rate=8000) as classifier:
        wanted_v, detected_v = classifier.is_wanted(str(WAV_DIR / "voice.wav"))
        assert detected_v == "V"
        assert wanted_v is True

        wanted_d, detected_d = classifier.is_wanted(str(WAV_DIR / "data.wav"))
        assert detected_d == "D"
        assert wanted_d is False


# --- Mocked Subprocess & Failure-Path Tests ---


def test_classifier_nonexistent_model_file_raises() -> None:
    """Test that a nonexistent model path raises FileNotFoundError."""
    params = ClassifierParams(
        wanted={"V": True, "D": True, "S": True},
        model_file_name=Path("/no/such/model.tflite"),
    )
    with pytest.raises(FileNotFoundError):
        _ = Classifier(params, audio_rate=8000)


def test_classifier_startup_handshake_failure_raises(tmp_path: Path) -> None:
    """Test that subprocess failing handshake raises RuntimeError."""
    fake_model = tmp_path / "model.tflite"
    fake_model.touch()
    params = ClassifierParams(
        wanted={"V": True, "D": True, "S": True}, model_file_name=fake_model
    )

    mock_proc = MagicMock()
    mock_proc.stdout.readline.return_value = "ERROR: Failed to load model\n"
    mock_proc.poll.return_value = 1
    mock_proc.stderr.fileno.return_value = 2

    with (
        patch("subprocess.Popen", return_value=mock_proc),
        patch(
            "select.select",
            return_value=([mock_proc.stdout], [], []),
        ),
        pytest.raises(RuntimeError, match="Handshake status"),
    ):
        _ = Classifier(params, audio_rate=8000)


def test_clean_up_no_process_is_noop() -> None:
    """Test that clean_up() is safe when no process exists."""
    clf = Classifier.__new__(Classifier)
    clf._proc = None
    clf._loaded = False
    clf.clean_up()  # Should not raise exception
    assert clf._proc is None


def test_clean_up_idempotent(classifier_params: ClassifierParams) -> None:
    """Test that clean_up() can be safely called multiple times."""
    with Classifier(classifier_params, audio_rate=8000) as classifier:
        classifier.clean_up()
        assert classifier._proc is None
        classifier.clean_up()  # Second call must be no-op


def test_clean_up_kills_on_terminate_timeout(
    classifier_params: ClassifierParams,
) -> None:
    """Test clean_up escalates to kill() if terminate() times out."""
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = "READY\n"
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=5),
            None,
        ]
        mock_popen.return_value = mock_proc

        with patch("select.select", return_value=([mock_proc.stdout], [], [])):
            clf = Classifier(classifier_params, audio_rate=8000)
            clf.clean_up()
            mock_proc.kill.assert_called_once()


def test_is_wanted_restarts_crashed_process(
    classifier_params: ClassifierParams,
) -> None:
    """Test that is_wanted self-heals by restarting if the process crashed."""
    with Classifier(classifier_params, audio_rate=8000) as classifier:
        proc_orig = classifier._proc
        assert proc_orig is not None
        # Simulate process crash
        proc_orig.kill()
        _ = proc_orig.wait()
        classifier._proc = None
        classifier._loaded = False

        # Next call to is_wanted should restart process and succeed
        wanted, detected = classifier.is_wanted(str(WAV_DIR / "voice.wav"))
        assert detected == "V"
        assert wanted is True
        assert classifier._proc is not None
        assert classifier._proc != proc_orig


def test_is_wanted_times_out_gracefully(
    classifier_params: ClassifierParams,
) -> None:
    """Test that is_wanted handles subprocess timeout gracefully returning (False, None)."""
    with (
        Classifier(classifier_params, audio_rate=8000) as classifier,
        patch("select.select", return_value=([], [], [])),
    ):
        wanted, detected = classifier.is_wanted(str(WAV_DIR / "voice.wav"))
        assert wanted is False
        assert detected is None


def test_context_manager_cleans_up_on_exception(
    classifier_params: ClassifierParams,
) -> None:
    """Test context manager cleans up process on exception."""
    proc: subprocess.Popen[str] | None = None
    try:
        with Classifier(classifier_params, audio_rate=8000) as classifier:
            proc = classifier._proc
            raise ValueError("Test error")
    except ValueError:
        pass

    assert proc is not None
    assert proc.poll() is not None
