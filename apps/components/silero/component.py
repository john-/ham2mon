"""Silero VAD WavGatekeeper component for ham2mon.

Evaluates WAV recordings directly using Silero VAD (ONNX Runtime).
"""

import wave
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from config import resolve_app_relative_path
from typing_extensions import override

from components.base import ChannelInfo, ComponentResult, WavGatekeeper


class OnnxInferenceSession(Protocol):
    """Protocol defining structural interface of an ONNX Runtime InferenceSession."""

    def get_inputs(self) -> list[object]: ...
    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, np.ndarray],
        run_options: object = None,
    ) -> list[np.ndarray]: ...


class OnnxSessionFactory(Protocol):
    """Protocol for the dynamic ONNX InferenceSession constructor."""

    def __call__(
        self, path_or_bytes: str, *args: object, **kwargs: object
    ) -> OnnxInferenceSession: ...


OnnxSessionClass: OnnxSessionFactory | None = None

try:
    import onnxruntime as ort

    OnnxSessionClass = cast(OnnxSessionFactory, cast(object, ort.InferenceSession))
except ImportError:
    OnnxSessionClass = None


class SileroVadComponent(WavGatekeeper):
    """WavGatekeeper component wrapping Silero VAD (ONNX model)."""

    threshold: float
    min_voice_chunks: int
    max_eval_sec: float
    model_path: Path
    _session: OnnxInferenceSession | None

    def __init__(self, config: dict[str, object]) -> None:
        super().__init__(config)
        raw_thresh = config.get("threshold", 0.5)
        if isinstance(raw_thresh, bool) or not isinstance(
            raw_thresh, (int, float, str)
        ):
            self.threshold = 0.5
        else:
            try:
                self.threshold = float(raw_thresh)
            except (ValueError, TypeError):
                self.threshold = 0.5
        raw_chunks = config.get("min_voice_chunks", 3)
        if isinstance(raw_chunks, bool) or not isinstance(raw_chunks, (int, float, str)):
            self.min_voice_chunks = 3
        else:
            try:
                self.min_voice_chunks = max(1, int(raw_chunks))
            except (ValueError, TypeError):
                self.min_voice_chunks = 3

        raw_max_eval = config.get("max_eval_sec", 3.0)
        if isinstance(raw_max_eval, bool) or not isinstance(raw_max_eval, (int, float, str)):
            self.max_eval_sec = 3.0
        else:
            try:
                self.max_eval_sec = max(0.0, float(raw_max_eval))
            except (ValueError, TypeError):
                self.max_eval_sec = 3.0

        model_str = str(
            config.get("model_path", "apps/components/silero/model/silero_vad.onnx")
        )
        self.model_path = Path(model_str)
        self._session = None

    @override
    def start(self) -> None:
        """Load Silero VAD ONNX model on startup."""
        if not callable(OnnxSessionClass):
            raise TypeError(
                "Could not load ONNX Runtime. "
                + "Please install 'onnxruntime' (e.g. `uv sync --extra onnxruntime`)."
            )

        resolved_path = resolve_app_relative_path(self.model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Silero VAD model not found at {self.model_path}")

        session_factory = OnnxSessionClass
        self._session = session_factory(resolved_path.as_posix())
        self.logger.info(
            "Silero VAD (ONNX) loaded successfully (threshold=%.2f, min_voice_chunks=%d, max_eval_sec=%.1f)",
            self.threshold,
            self.min_voice_chunks,
            self.max_eval_sec,
        )

    @override
    def process(self, wav_path: str, channel_info: ChannelInfo) -> ComponentResult:
        """Evaluate a WAV file for voice activity using Silero VAD ONNX."""
        if self._session is None:
            return ComponentResult(keep=True)

        try:
            with wave.open(wav_path, "rb") as w:
                nchannels, sampwidth, framerate, nframes = w.getparams()[:4]
                data = w.readframes(nframes)

            if sampwidth == 2:
                audio = (
                    np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                )
            elif sampwidth == 1:
                audio = (
                    np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0
                ) / 128.0
            elif sampwidth == 4:
                audio = (
                    np.frombuffer(data, dtype=np.int32).astype(np.float32)
                    / 2147483648.0
                )
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth}")

            if nchannels > 1:
                audio = audio.reshape(-1, nchannels).mean(axis=1)

            if len(audio) == 0:
                return ComponentResult(
                    keep=False, classification="S", detail="Empty audio file"
                )

            # Cap evaluation duration if max_eval_sec is set (> 0)
            if self.max_eval_sec > 0.0:
                target_samples = int(framerate * self.max_eval_sec)
                if len(audio) > target_samples:
                    middle = len(audio) / 2.0
                    start = max(int(middle - target_samples / 2.0), 0)
                    end = int(middle + target_samples / 2.0)
                    audio = audio[start:end]

            # Silero VAD requires exact 8000 Hz or 16000 Hz audio.
            # Resample non-standard framerates (e.g. 11025, 22050, 44100) to 16000 Hz.
            target_sr = 8000 if framerate == 8000 else 16000
            if framerate != target_sr:
                num_target_samples = int(len(audio) * target_sr / framerate)
                if num_target_samples > 0:
                    audio = np.interp(
                        np.linspace(0, len(audio), num_target_samples, endpoint=False),
                        np.arange(len(audio)),
                        audio,
                    ).astype(np.float32)
                framerate = target_sr

            chunk_size = 512 if framerate == 16000 else 256
            sr = framerate

            # Zero-pad trailing partial chunk so no audio tail is dropped
            remainder = len(audio) % chunk_size
            if remainder > 0:
                pad_size = chunk_size - remainder
                audio = np.pad(audio, (0, pad_size), mode="constant")

            speech_probs: list[float] = []

            # Silero VAD v4 ONNX model inputs:
            # - input: audio chunk array shape (1, chunk_size)
            # - sr: sample rate scalar int64 array shape (1,)
            # - h: lstm hidden state shape (2, 1, 64)
            # - c: lstm cell state shape (2, 1, 64)
            h = np.zeros((2, 1, 64), dtype=np.float32)
            c = np.zeros((2, 1, 64), dtype=np.float32)
            sr_arr = np.array([sr], dtype=np.int64)

            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size].reshape(1, -1)
                inputs = {
                    "input": chunk,
                    "sr": sr_arr,
                    "h": h,
                    "c": c,
                }
                out = self._session.run(None, inputs)
                out_prob = out[0]
                h = out[1]
                c = out[2]
                out_arr = np.asarray(out_prob)
                speech_probs.append(float(out_arr.flat[0]))  # pyright: ignore[reportAny]

            avg_prob = float(np.mean(speech_probs)) if speech_probs else 0.0
            max_prob = float(np.max(speech_probs)) if speech_probs else 0.0  # pyright: ignore[reportAny]

            # Require min_voice_chunks to meet/exceed threshold for positive voice decision
            chunks_above_thresh = sum(1 for p in speech_probs if p >= self.threshold)
            is_voice = chunks_above_thresh >= self.min_voice_chunks

            if is_voice:
                return ComponentResult(
                    keep=True,
                    classification="V",
                    metadata={
                        "vad_prob": max_prob,
                        "vad_avg_prob": avg_prob,
                        "voice_chunks": chunks_above_thresh,
                    },
                )

            msg_detail = (
                f"No voice detected (max prob: {max_prob:.2f} < {self.threshold})"
            )
            return ComponentResult(
                keep=False,
                classification="S",
                detail=msg_detail,
            )

        except Exception as e:
            self.logger.warning(
                "Error processing VAD on %s: %s", wav_path, e, exc_info=True
            )
            return ComponentResult(keep=True, detail="VAD evaluation error")

    @override
    def stop(self) -> None:
        """Clean up session resources on stop."""
        self._session = None
