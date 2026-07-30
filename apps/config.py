"""
Typed configuration models and centralized validation for ham2mon.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, List, Optional

import yaml

logger = logging.getLogger(f"ham2mon.{__name__}")


class ConfigError(Exception):
    """Raised when ham2mon configuration invariants are violated."""

    pass


@dataclass(kw_only=True)
class HardwareConfig:
    """Hardware and SDR receiver settings."""

    args: str = "uhd"
    sample_rate: float = 4.0e6
    freq_correction: int = 0
    center_frequencies: List[str] = field(default_factory=lambda: ["146"])

    def __post_init__(self):
        if self.sample_rate < 1.0e6:
            raise ConfigError(
                f"Hardware sample_rate ({self.sample_rate} Hz) must be at least 1,000,000 Hz (1 MHz)."
            )

        for entry in self.center_frequencies:
            if "-" in str(entry):
                parts = str(entry).split("-")
                if len(parts) != 2:
                    raise ConfigError(f"Invalid frequency range format: '{entry}'. Expected format: 'lower-upper'.")
                try:
                    lo, hi = float(parts[0]), float(parts[1])
                    if lo >= hi:
                        raise ConfigError(
                            f"Invalid frequency range '{entry}': lower frequency ({lo}) must be < upper ({hi})."
                        )
                except ValueError:
                    raise ConfigError(f"Invalid frequency range format: '{entry}'.")


@dataclass(kw_only=True)
class GainConfig:
    """SDR hardware channel gains in dB.

    Fields default to None, meaning the user did not explicitly supply a value.
    Use get_value() to retrieve a usable float (user value or application default).
    Use is_explicit() to test whether the user supplied a value for a given gain.
    """

    agc: bool = False
    rf: Optional[float] = None
    lna: Optional[float] = None
    mix: Optional[float] = None
    if_gain: Optional[float] = None
    bb: Optional[float] = None
    att: Optional[float] = None
    lna_mix_bb: Optional[float] = None
    tia: Optional[float] = None
    pga: Optional[float] = None
    lb: Optional[float] = None

    DEFAULTS: ClassVar[dict[str, float]] = {
        "rf":        0.0,
        "lna":       8.0,
        "mix":       5.0,
        "if_gain":  16.0,
        "bb":       16.0,
        "att":       8.0,
        "lna_mix_bb": 8.0,
        "tia":       8.0,
        "pga":       8.0,
        "lb":        8.0,
    }

    def get_value(self, name: str) -> float:
        """Return the user-supplied value if present, else the application default."""
        user_val = getattr(self, name, None)
        if user_val is not None:
            return float(user_val)
        return self.DEFAULTS.get(name, 0.0)

    def is_explicit(self, name: str) -> bool:
        """Return True if the user explicitly supplied a value for this gain field."""
        return getattr(self, name, None) is not None

    def set_value(self, name: str, value: float) -> None:
        """Set a gain field in-place (used by the TUI for runtime adjustment)."""
        setattr(self, name, value)


# Ordered mapping of GainConfig field names to osmosdr hardware gain element names.
# This is the single canonical mapping used by Receiver to validate and apply gains.
GAIN_FIELDS: list[tuple[str, str]] = [
    ("rf",         "RF"),
    ("lna",        "LNA"),
    ("mix",        "MIX"),
    ("if_gain",    "IF"),
    ("bb",         "BB"),
    ("att",        "ATT"),
    ("lna_mix_bb", "LNA_MIX_BB"),
    ("tia",        "TIA"),
    ("pga",        "PGA"),
    ("lb",         "LB"),
]


@dataclass(kw_only=True)
class ScannerConfig:
    """Scanner loop timing and priority-promotion behavior."""

    quiet_timeout: int = 12
    active_timeout: int = 20
    auto_priority: bool = False


@dataclass(kw_only=True)
class ReceiverConfig:
    """Receiver DSP, demodulator, squelch, and channel timeout settings."""

    demodulators: int = 4
    mode: int = 0
    squelch_db: int = -60
    threshold_db: int = 10
    channel_spacing: int = 5000
    max_ctcss_tones: int = 0

    def __post_init__(self):
        if self.mode not in (0, 1, 2):
            raise ConfigError(
                f"Invalid demodulator mode: {self.mode}. Must be 0 (NBFM), 1 (AM), or 2 (WBFM)."
            )


@dataclass(kw_only=True)
class AudioConfig:
    """Audio playback, WAV file recording limits, and metadata settings."""

    play: bool = True
    volume_db: int = 0
    record: bool = False
    bit_depth: int = 16
    min_recording_sec: float = 0.0
    max_recording_sec: float = 0.0
    file_metadata: List[str] = field(default_factory=list)
    wav_dir: str = "wav"


    def __post_init__(self):
        valid_fields = {"priority", "strength", "ctcss"}
        for field_name in self.file_metadata:
            if field_name.lower() not in valid_fields:
                raise ConfigError(
                    f"Unsupported metadata field: '{field_name}'. Allowed: {valid_fields}"
                )
        if (self.min_recording_sec > 0.0 or self.max_recording_sec > 0.0) and not self.record:
            raise ConfigError(
                "Recording limits (min_recording_sec / max_recording_sec) require "
                "recording to be enabled (record: true / -w / --write)."
            )


@dataclass(kw_only=True)
class WantedFlags:
    """ML signal wanted flags (voice, data, skip classification targets)."""

    voice: bool = False
    data: bool = False
    skip: bool = False


@dataclass(kw_only=True)
class ClassificationConfig:
    """Signal classification model path and wanted flag mappings."""

    model_path: Optional[Path] = None
    wanted: WantedFlags = field(default_factory=WantedFlags)

    def __post_init__(self):
        # Classification requires a valid model file on disk
        if self.wanted.voice or self.wanted.data or self.wanted.skip:
            if not self.model_path:
                raise ConfigError(
                    "Classification enabled (voice/data/skip/auto_priority), but 'model_path' is missing."
                )
            model_p = Path(self.model_path)
            if not model_p.exists():
                raise ConfigError(
                    f"Classification model file not found at: {self.model_path}"
                )


@dataclass(kw_only=True)
class FrequencyPoliciesConfig:
    """Frequencies file path, lockout settings, and priority overrides."""

    file: Optional[Path] = None
    disable_lockout: bool = False
    disable_priority: bool = False


@dataclass(kw_only=True)
class DisplayConfig:
    """Spectrum visualizer display window bounds and theme file selection."""

    min_db: float = -10.0
    max_db: float = 50.0
    theme_file: str = "default.theme.yaml"

    def __post_init__(self):
        if self.min_db >= (self.max_db - 10.0):
            raise ConfigError(
                f"min_db ({self.min_db}) must be at least 10dB lower than max_db ({self.max_db})."
            )


@dataclass(kw_only=True)
class ActivityConfig:
    """Telemetry activity type, destination path, and log interval settings."""

    type: str = "none"
    dest: str = "channel-log"
    interval_sec: int = 15

    def __post_init__(self):
        valid_types = {"none", "fixed-field", "json-server"}
        if self.type not in valid_types:
            raise ConfigError(
                f"Invalid activity.type '{self.type}'. Must be one of {valid_types}."
            )


@dataclass(kw_only=True)
class LoggingConfig:
    """Diagnostic system logger destination, level, and file configuration."""

    dest: str = "none"
    level: str = "warn"
    file: str = ""

    def __post_init__(self):
        valid_dests = {"none", "file", "syslog", "stderr"}
        if self.dest not in valid_dests:
            raise ConfigError(
                f"Invalid logging.dest '{self.dest}'. Must be one of {valid_dests}."
            )


@dataclass(kw_only=True)
class MasterHam2MonConfig:
    """Master configuration container wrapping all individual sub-configurations."""

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    gains: GainConfig = field(default_factory=GainConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    receiver: ReceiverConfig = field(default_factory=ReceiverConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    frequency_policies: FrequencyPoliciesConfig = field(default_factory=FrequencyPoliciesConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    channel_activity: ActivityConfig = field(default_factory=ActivityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        if self.scanner.auto_priority:
            self.classification.wanted.voice = True

        if self.scanner.auto_priority and self.frequency_policies.disable_priority:
            logger.warning(
                "scanner.auto_priority is enabled, but frequency_policies.disable_priority is True. "
                "Auto-promoted priority channels will be ignored by frequency manager."
            )

        # Classification forces audio recording
        if (
            self.classification.wanted.voice
            or self.classification.wanted.data
            or self.classification.wanted.skip
        ):
            self.audio.record = True
            self.classification.__post_init__()


def load_raw_yaml(yaml_path: Path | str) -> dict[str, object]:
    """Load raw dictionary from a YAML file."""
    path = Path(yaml_path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_config_path(given_path: str | Path, domain_ext: str) -> Path:
    """Resolve a config path for one domain (e.g. ".config.yaml" or ".freqs.yaml").

    Resolution order:
    1. If *given_path* exists exactly as specified, return it.
    2. Else append *domain_ext* and check again
       (e.g. ``"site-ra"`` → ``"site-ra.config.yaml"``).
    3. If neither exists, raise :exc:`ConfigError` naming both paths tried.

    A relative *given_path* is resolved against the current working directory
    (standard CLI-argument behaviour).  An absolute path is used as-is.
    There is no search-directory list — if the path is not found by step 2,
    resolution fails immediately with a clear error message.

    Args:
        given_path: The path string or :class:`~pathlib.Path` supplied by the
            user (e.g. from a ``-C`` / ``-F`` argument).
        domain_ext: The double-extension suffix to append when the bare name
            does not resolve on its own (e.g. ``".config.yaml"`` or
            ``".freqs.yaml"``).

    Returns:
        The resolved, existing :class:`~pathlib.Path`.

    Raises:
        :exc:`ConfigError`: If neither *given_path* nor
            *given_path* + *domain_ext* exists on disk.
    """
    candidate = Path(given_path)
    if candidate.exists():
        return candidate
    with_ext = Path(str(given_path) + domain_ext)
    if with_ext.exists():
        return with_ext
    raise ConfigError(
        f"Configuration file not found. Tried:\n"
        f"  {candidate}\n"
        f"  {with_ext}"
    )


def build_config_from_dict(raw: dict[str, object]) -> MasterHam2MonConfig:
    """Build and validate MasterHam2MonConfig from a dictionary (supports nested YAML structure)."""
    hw_data = raw.get("hardware", {})
    hardware = (
        HardwareConfig(**hw_data) if isinstance(hw_data, dict) else HardwareConfig()
    )

    gains_val = raw.get("gains")
    gains_data = dict(gains_val) if isinstance(gains_val, dict) else {}
    if "if" in gains_data:
        val = gains_data.pop("if")
        if "if_gain" not in gains_data:
            gains_data["if_gain"] = val
    gains = GainConfig(**gains_data)

    scanner_data = raw.get("scanner", {})
    scanner = (
        ScannerConfig(**scanner_data) if isinstance(scanner_data, dict) else ScannerConfig()
    )

    recv_data = raw.get("receiver", {})
    receiver = (
        ReceiverConfig(**recv_data) if isinstance(recv_data, dict) else ReceiverConfig()
    )

    audio_data = raw.get("audio", {})
    audio = AudioConfig(**audio_data) if isinstance(audio_data, dict) else AudioConfig()

    class_val = raw.get("classification")
    class_data = dict(class_val) if isinstance(class_val, dict) else {}
    wanted_data = class_data.pop("wanted", {}) if "wanted" in class_data else {}
    wanted = (
        WantedFlags(**wanted_data) if isinstance(wanted_data, dict) else WantedFlags()
    )
    model_val = class_data.get("model_path")
    if model_val is not None and str(model_val).strip().lower() not in (
        "none",
        "null",
        "",
    ):
        class_data["model_path"] = Path(model_val)
    else:
        class_data["model_path"] = None
    classification = ClassificationConfig(wanted=wanted, **class_data)

    freq_val = raw.get("frequency_policies")
    freq_data = dict(freq_val) if isinstance(freq_val, dict) else {}
    # frequency_policies.file is only ever written here by the CLI merge layer
    # (already a resolved Path).  A stray string value from a hand-edited
    # general config is coerced to Path so it fails predictably downstream
    # (FileNotFoundError / AttributeError) rather than silently doing nothing.
    file_val = freq_data.get("file")
    if file_val is not None:
        freq_data["file"] = Path(file_val)
    frequency_policies = FrequencyPoliciesConfig(**freq_data)

    disp_val = raw.get("display")
    disp_data = dict(disp_val) if isinstance(disp_val, dict) else {}
    theme_val = disp_data.get("theme_file")
    if theme_val is not None and str(theme_val).strip().lower() in ("none", "null", ""):
        disp_data["theme_file"] = ""
    display = DisplayConfig(**disp_data)

    act_data = raw.get("channel_activity", {})
    channel_activity = (
        ActivityConfig(**act_data) if isinstance(act_data, dict) else ActivityConfig()
    )

    log_val = raw.get("logging")
    log_data = dict(log_val) if isinstance(log_val, dict) else {}
    log_file = log_data.get("file")
    if log_file is not None and str(log_file).strip().lower() in ("none", "null", ""):
        log_data["file"] = ""
    logging = LoggingConfig(**log_data)

    return MasterHam2MonConfig(
        hardware=hardware,
        gains=gains,
        scanner=scanner,
        receiver=receiver,
        audio=audio,
        classification=classification,
        frequency_policies=frequency_policies,
        display=display,
        channel_activity=channel_activity,
        logging=logging,
    )
