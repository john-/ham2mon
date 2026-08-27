"""
Reference WavGatekeeper implementation wrapping TFLite audio classification.
"""

from pathlib import Path
from typing import cast

from config import resolve_app_relative_path
from typing_extensions import override

from components.base import ChannelInfo, ComponentResult, WavGatekeeper

from .classification import ClassificationNotWanted, Classifier, ClassifierParams


class TfliteClassifierComponent(WavGatekeeper):
    """WavGatekeeper component wrapping the TFLite ML classifier."""

    def __init__(self, config: dict[str, object]) -> None:
        super().__init__(config)
        self._classifier: Classifier | None = None

    @override
    def start(self) -> None:
        """Instantiate Classifier model on startup."""
        model_path_val = self.config.get("model_path")
        model_path = (
            resolve_app_relative_path(str(model_path_val))
            if model_path_val
            else Path("")
        )
        wanted_val = self.config.get("wanted")
        wanted_cfg = (
            cast(dict[object, object], wanted_val)
            if isinstance(wanted_val, dict)
            else {}
        )

        params = ClassifierParams(
            wanted={
                "V": bool(wanted_cfg.get("voice", False)),
                "D": bool(wanted_cfg.get("data", False)),
                "S": bool(wanted_cfg.get("skip", False)),
            },
            model_file_name=model_path,
        )

        try:
            self._classifier = Classifier(params, audio_rate=8000)
            self.logger.info(
                "TfliteClassifierComponent initialized with model %s", model_path
            )
        except ClassificationNotWanted:
            self.logger.info(
                "No classification targets wanted; TfliteClassifierComponent disabled."
            )
            self._classifier = None

    @override
    def process(self, wav_path: str, channel_info: ChannelInfo) -> ComponentResult:
        """Classify tmp WAV file."""
        if not self._classifier:
            return ComponentResult(keep=True)

        is_wanted, classification = self._classifier.is_wanted(wav_path)
        if not is_wanted:
            return ComponentResult(
                keep=False,
                classification=classification,
                detail="Discarded unwanted classification",
            )

        metadata: dict[str, object] = {}
        if classification:
            metadata["classification"] = classification

        return ComponentResult(
            keep=True,
            classification=classification,
            metadata=metadata,
        )

    @override
    def stop(self) -> None:
        """Clean up classifier resources."""
        if self._classifier:
            self._classifier.clean_up()
            self._classifier = None
