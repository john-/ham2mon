import dataclasses
from pathlib import Path

import pytest
from config import (
    ConfigError,
    MasterHam2MonConfig,
    build_config_from_dict,
    resolve_config_path,
)
from h2m_parser import CLI_OPTION_MAP, CLParser


def test_default_config_build():
    config = build_config_from_dict({})
    assert config.hardware.sample_rate == 4.0e6
    assert config.receiver.demodulators == 4
    assert config.audio.bit_depth == 16


def test_auto_priority_cascades_voice(tmp_path: Path):
    model_file = tmp_path / "test.tflite"
    model_file.touch()

    config = build_config_from_dict({
        "classification": {
            "model_path": model_file,
            "auto_priority": True
        }
    })
    assert config.classification.wanted.voice is True
    assert config.audio.record is True


def test_classification_requires_model():
    with pytest.raises(ConfigError, match="model_path"):
        _ = build_config_from_dict({
            "classification": {
                "wanted": {"voice": True}
            }
        })


def test_classification_model_must_exist(tmp_path: Path):
    missing_model = tmp_path / "nonexistent.tflite"
    with pytest.raises(ConfigError, match="not found"):
        _ = build_config_from_dict({
            "classification": {
                "model_path": missing_model,
                "wanted": {"voice": True}
            }
        })


def test_sample_rate_floor():
    with pytest.raises(ConfigError, match="sample_rate"):
        _ = build_config_from_dict({
            "hardware": {"sample_rate": 500000}
        })


def test_frequency_range_validation():
    with pytest.raises(ConfigError, match="lower frequency"):
        _ = build_config_from_dict({
            "hardware": {"center_frequencies": ["468-460"]}
        })


def test_frequency_range_format_validation():
    with pytest.raises(ConfigError, match="Expected format"):
        _ = build_config_from_dict({
            "hardware": {"center_frequencies": ["146-155-160"]}
        })


def test_recording_limits_require_recording_enabled():
    with pytest.raises(ConfigError, match="Recording limits"):
        _ = build_config_from_dict({
            "audio": {
                "record": False,
                "min_recording_sec": 1.0
            }
        })


def test_demod_mode_validation():
    with pytest.raises(ConfigError, match="Invalid demodulator mode"):
        _ = build_config_from_dict({
            "receiver": {"mode": 5}
        })


def test_display_db_window_validation():
    with pytest.raises(ConfigError, match="min_db"):
        _ = build_config_from_dict({
            "display": {"min_db": 50.0, "max_db": 50.0}
        })


def test_metadata_validation():
    with pytest.raises(ConfigError, match="Unsupported metadata field"):
        _ = build_config_from_dict({
            "audio": {"file_metadata": ["invalid_tag"]}
        })


def test_clparser_integration():
    parser = CLParser(args=[])
    assert hasattr(parser, "master_config")
    assert isinstance(parser.master_config, MasterHam2MonConfig)


def test_theme_file_none_normalization():
    """display.theme_file and logging.file string-normalization is unchanged."""
    for val in ("None", "none", "null", ""):
        config = build_config_from_dict({
            "classification": {"model_path": val},
            "display": {"theme_file": val},
            "logging": {"file": val}
        })
        assert config.classification.model_path is None
        assert config.display.theme_file == ""
        assert config.logging.file == ""


def test_frequencies_file_in_general_config_is_ignored_by_clparser(tmp_path: Path):
    """A frequency_policies: file: key embedded in a general config YAML must be
    stripped by CLParser before merge — it must not populate
    MasterHam2MonConfig.frequency_policies.file unless -F is explicitly supplied."""
    cfg_file = tmp_path / "site.config.yaml"
    _ = cfg_file.write_text(
        "hardware:\n  center_frequencies:\n    - '146'\n"
        + "frequency_policies:\n  file: chicago\n"
    )
    parser = CLParser(args=["-C", str(cfg_file)])
    assert parser.master_config.frequency_policies.file is None


def test_frequencies_file_set_via_cli(tmp_path: Path):
    """frequency_policies.file IS populated when -F is explicitly supplied."""
    freqs_file = tmp_path / "chicago.freqs.yaml"
    _ = freqs_file.write_text("frequencies: []\n")
    parser = CLParser(args=["-F", str(freqs_file)])
    assert parser.master_config.frequency_policies.file == freqs_file


# ---------------------------------------------------------------------------
# resolve_config_path() unit tests
# ---------------------------------------------------------------------------

def test_resolve_config_path_exact_hit(tmp_path: Path):
    """Exact path that exists is returned unchanged."""
    f = tmp_path / "my.config.yaml"
    f.touch()
    assert resolve_config_path(f, ".config.yaml") == f


def test_resolve_config_path_suffix_hit(tmp_path: Path):
    """Bare name + domain suffix resolves when the suffixed file exists."""
    f = tmp_path / "site-ra.config.yaml"
    f.touch()
    bare = tmp_path / "site-ra"
    result = resolve_config_path(bare, ".config.yaml")
    assert result == f


def test_resolve_config_path_freqs_suffix_hit(tmp_path: Path):
    """Same suffix-append logic works for the .freqs.yaml domain."""
    f = tmp_path / "chicago.freqs.yaml"
    f.touch()
    bare = tmp_path / "chicago"
    assert resolve_config_path(bare, ".freqs.yaml") == f


def test_resolve_config_path_not_found_raises(tmp_path: Path):
    """ConfigError is raised when neither candidate exists."""
    missing = tmp_path / "nonexistent"
    with pytest.raises(ConfigError, match="nonexistent"):
        _ = resolve_config_path(missing, ".config.yaml")


def test_resolve_config_path_not_found_names_both_candidates(tmp_path: Path):
    """The error message names both the bare path and the suffixed path."""
    missing = tmp_path / "ghost"
    with pytest.raises(ConfigError) as exc_info:
        _ = resolve_config_path(missing, ".freqs.yaml")
    msg = str(exc_info.value)
    assert "ghost" in msg
    assert ".freqs.yaml" in msg


def test_resolve_config_path_absolute(tmp_path: Path):
    """Absolute paths work the same as relative ones."""
    f = tmp_path / "prod.freqs.yaml"
    f.touch()
    assert resolve_config_path(f.resolve(), ".freqs.yaml") == f.resolve()


# ---------------------------------------------------------------------------
# CLParser resolution-failure → clean SystemExit (via parser.error())
# ---------------------------------------------------------------------------

def test_clparser_bad_config_path_exits_cleanly(tmp_path: Path):
    """-C pointing at a nonexistent file must exit via SystemExit (not raise
    a raw ConfigError traceback)."""
    with pytest.raises(SystemExit):
        _ = CLParser(args=["-C", str(tmp_path / "no-such.config.yaml")])


def test_clparser_bad_frequencies_path_exits_cleanly(tmp_path: Path):
    """-F pointing at a nonexistent file must exit via SystemExit."""
    with pytest.raises(SystemExit):
        _ = CLParser(args=["-F", str(tmp_path / "no-such.freqs.yaml")])

def test_gains_if_collision():
    # Verify that when YAML has 'if' and CLI has 'if_gain', the CLI value wins and 'if' is safely popped
    config = build_config_from_dict({
        "gains": {
            "if": 9.0,
            "if_gain": 12.0
        }
    })
    assert config.gains.if_gain == 12.0


def test_gain_config_defaults_and_explicit_tracking():
    config = build_config_from_dict({
        "gains": {
            "lna": 14.0
        }
    })
    # lna was explicitly provided
    assert config.gains.is_explicit("lna") is True
    assert config.gains.get_value("lna") == 14.0

    # mix was not explicitly provided -> returns default
    assert config.gains.is_explicit("mix") is False
    assert config.gains.get_value("mix") == 5.0


def test_gain_config_set_value():
    config = build_config_from_dict({})
    config.gains.set_value("rf", 10.0)
    assert config.gains.get_value("rf") == 10.0
    assert config.gains.is_explicit("rf") is True


def test_all_config_fields_have_cli_mapping_or_are_allowlisted():
    intentionally_yaml_only = {("display", "theme_file"), ("audio", "wav_dir")}
    mapped = {(m.section, m.key) for m in CLI_OPTION_MAP} | intentionally_yaml_only

    for section_field in dataclasses.fields(MasterHam2MonConfig):
        section_name = section_field.name
        section_cls = section_field.type
        if dataclasses.is_dataclass(section_cls):
            for f in dataclasses.fields(section_cls):
                if f.name == "wanted":  # nested WantedFlags check
                    wanted_cls = f.type
                    if dataclasses.is_dataclass(wanted_cls):
                        for sub_f in dataclasses.fields(wanted_cls):
                            assert (section_name, sub_f.name) in mapped, f"{section_name}.{sub_f.name} (nested) has no CLI mapping"
                    continue
                assert (section_name, f.name) in mapped, f"{section_name}.{f.name} has no CLI mapping"
