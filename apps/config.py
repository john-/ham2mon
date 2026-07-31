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
    hold_scan_on: set[str] | None = None  # None = hold on all recorded types


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
class ComponentEntryConfig:
    """Config entry for one component in the YAML configuration file."""

    class_path: str
    timeout_sec: float = 10.0
    config: dict[str, object] = field(default_factory=dict)
    name: str = ""


@dataclass(kw_only=True)
class ComponentsConfig:
    """Container for active component configurations."""

    wav_gatekeeper: ComponentEntryConfig | None = None
    notifiers: list[ComponentEntryConfig] = field(default_factory=list)


@dataclass(kw_only=True)
class MasterHam2MonConfig:
    """Root configuration tree combining all domain sub-configs."""

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    gains: GainConfig = field(default_factory=GainConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    receiver: ReceiverConfig = field(default_factory=ReceiverConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    frequency_policies: FrequencyPoliciesConfig = field(default_factory=FrequencyPoliciesConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    channel_activity: ActivityConfig = field(default_factory=ActivityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    components: ComponentsConfig = field(default_factory=ComponentsConfig)

    def __post_init__(self):
        if self.scanner.auto_priority and self.frequency_policies.disable_priority:
            logger.warning(
                "scanner.auto_priority is enabled, but frequency_policies.disable_priority is True. "
                "Auto-promoted priority channels will be ignored by frequency manager."
            )

        if self.scanner.hold_scan_on is not None and self.components.wav_gatekeeper is None:
            logger.warning(
                "scanner.hold_scan_on is configured (%s), but no components.wav_gatekeeper "
                "classifier is configured. Unclassified transmissions (classification=None) "
                "will not match hold_scan_on, so range scanning will not hold.",
                self.scanner.hold_scan_on,
            )

        # WavGatekeeper forces audio recording so temporary files can be evaluated
        if self.components.wav_gatekeeper is not None:
            self.audio.record = True


def load_raw_yaml(yaml_path: Path | str) -> dict[str, object]:
    """Load raw dictionary from a YAML file."""
    path = Path(yaml_path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_app_relative_path(given_path: str | Path) -> Path:
    """Resolve a path against current working directory and project root (`apps/`).

    Resolution order:
    1. If `given_path` is absolute or exists as specified, return it.
    2. If `given_path` starts with 'apps/' and doesn't exist, try stripping 'apps/'
       (for execution inside apps/).
    3. If `given_path` doesn't start with 'apps/' and doesn't exist, try prefixing 'apps/'
       (for execution from repo root).
    4. Return `given_path` as Path object.
    """
    path = Path(given_path)
    if path.is_absolute() or path.exists():
        return path

    path_str = str(given_path)
    if path_str.startswith("apps/"):
        stripped = Path(path_str[5:])
        if stripped.exists():
            return stripped
    else:
        prefixed = Path("apps") / path
        if prefixed.exists():
            return prefixed

    return path


def resolve_config_path(given_path: str | Path, domain_ext: str) -> Path:
    """Resolve a config path for one domain (e.g. ".config.yaml" or ".freqs.yaml").

    Resolution order:
    1. If *given_path* exists exactly as specified (or via app root fallback), return it.
    2. Else append *domain_ext* and check again
       (e.g. ``"site-ra"`` → ``"site-ra.config.yaml"``).
    3. If neither exists, raise :exc:`ConfigError` naming both paths tried.
    """
    candidate = resolve_app_relative_path(given_path)
    if candidate.exists():
        return candidate

    with_ext = resolve_app_relative_path(str(given_path) + domain_ext)
    if with_ext.exists():
        return with_ext

    raw_cand = Path(given_path)
    raw_ext = Path(str(given_path) + domain_ext)
    raise ConfigError(
        f"Configuration file not found. Tried:\n"
        f"  {raw_cand}\n"
        f"  {raw_ext}"
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

    scanner_val = raw.get("scanner")
    scanner_data = dict(scanner_val) if isinstance(scanner_val, dict) else {}
    if "hold_scan_on" in scanner_data and scanner_data["hold_scan_on"] is not None:
        hso = scanner_data["hold_scan_on"]
        if isinstance(hso, str):
            scanner_data["hold_scan_on"] = {x.strip() for x in hso.split(",") if x.strip()}
        elif isinstance(hso, (list, tuple, set)):
            scanner_data["hold_scan_on"] = {str(item).strip() for item in hso if str(item).strip()}
        else:
            raise ConfigError(
                f"Invalid scanner.hold_scan_on type: {type(hso).__name__}. Expected string or list of classification codes."
            )
    scanner = ScannerConfig(**scanner_data)

    recv_data = raw.get("receiver", {})
    receiver = (
        ReceiverConfig(**recv_data) if isinstance(recv_data, dict) else ReceiverConfig()
    )

    audio_data = raw.get("audio", {})
    audio = AudioConfig(**audio_data) if isinstance(audio_data, dict) else AudioConfig()



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

    comp_val = raw.get("components")
    comp_data = dict(comp_val) if isinstance(comp_val, dict) else {}
    gk_val = comp_data.get("wav_gatekeeper")
    wav_gatekeeper = ComponentEntryConfig(**gk_val) if isinstance(gk_val, dict) else None

    notif_list = comp_data.get("notifiers", [])
    notifiers = []
    if isinstance(notif_list, list):
        for n_item in notif_list:
            if isinstance(n_item, dict):
                notifiers.append(ComponentEntryConfig(**n_item))

    components = ComponentsConfig(
        wav_gatekeeper=wav_gatekeeper, notifiers=notifiers
    )

    return MasterHam2MonConfig(
        hardware=hardware,
        gains=gains,
        scanner=scanner,
        receiver=receiver,
        audio=audio,
        frequency_policies=frequency_policies,
        display=display,
        channel_activity=channel_activity,
        logging=logging,
        components=components,
    )

