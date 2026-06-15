import pytest
from pathlib import Path
from classification import Classifier, ClassifierParams, ClassificationNotWanted

TEST_DIR = Path(__file__).parent
MODEL_PATH = TEST_DIR.parent / "model" / "model_1.tflite"
WAV_DIR = TEST_DIR.parent / "test"

@pytest.fixture
def classifier_params():
    return ClassifierParams(
        wanted={'V': True, 'D': True, 'S': True},
        model_file_name=MODEL_PATH
    )

def test_classifier_initialization(classifier_params):
    """Test that the classifier initializes and loads the TFLite model successfully."""
    with Classifier(classifier_params, audio_rate=8000) as classifier:
        assert classifier._loaded is True
        assert classifier.audio_rate == 8000
        assert classifier._proc is not None

def test_classification_not_wanted():
    """Test that Classifier raises ClassificationNotWanted if no categories are requested."""
    params = ClassifierParams(
        wanted={'V': False, 'D': False, 'S': False},
        model_file_name=MODEL_PATH
    )
    with pytest.raises(ClassificationNotWanted):
        Classifier(params, audio_rate=8000)

def test_classify_voice(classifier_params):
    """Test classification of a voice recording."""
    with Classifier(classifier_params, audio_rate=8000) as classifier:
        wanted, detected = classifier.is_wanted(str(WAV_DIR / "voice.wav"))
        assert detected == 'V'
        assert wanted is True

def test_classify_data(classifier_params):
    """Test classification of a data recording."""
    with Classifier(classifier_params, audio_rate=8000) as classifier:
        wanted, detected = classifier.is_wanted(str(WAV_DIR / "data.wav"))
        assert detected == 'D'
        assert wanted is True

def test_classify_skip(classifier_params):
    """Test classification of a skip/noise recording."""
    with Classifier(classifier_params, audio_rate=8000) as classifier:
        wanted, detected = classifier.is_wanted(str(WAV_DIR / "skip.wav"))
        assert detected == 'S'
        assert wanted is True

def test_classify_filtering():
    """Test that files detected as unwanted are filtered out (wanted is False)."""
    # Configure parameter to only want Voice (V)
    params = ClassifierParams(
        wanted={'V': True, 'D': False, 'S': False},
        model_file_name=MODEL_PATH
    )
    with Classifier(params, audio_rate=8000) as classifier:
        # Voice should return wanted=True
        wanted_v, detected_v = classifier.is_wanted(str(WAV_DIR / "voice.wav"))
        assert detected_v == 'V'
        assert wanted_v is True
        
        # Data should return wanted=False (filtered out)
        wanted_d, detected_d = classifier.is_wanted(str(WAV_DIR / "data.wav"))
        assert detected_d == 'D'
        assert wanted_d is False
