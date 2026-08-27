#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command line parser for ham2mon with YAML configuration file support.

Created on Sat Jul 18 15:21:33 2015
@author: madengr
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional
from channel_loggers import ActivityParams
from center_frequency_provider import FrequencyRangeParams, FrequencySingleParams, FrequencyGroup
from frequency_manager import FrequencyConfiguration
from config import build_config_from_dict, load_raw_yaml, resolve_config_path, MasterHam2MonConfig, ConfigError, GainConfig


@dataclass(frozen=True)
class CliMapping:
    option_attr: str            # attribute on argparse Namespace
    section: str                # top-level YAML section, e.g. "hardware"
    key: str                    # key within that section, e.g. "sample_rate"
    caster: Callable[[Any], Any] = lambda v: v  # e.g. float, int, str, Path, list
    subsection: str | None = None  # for classification.wanted.*


CLI_OPTION_MAP: list[CliMapping] = [
    # Hardware
    CliMapping("hw_args",         "hardware", "args",               str),
    CliMapping("ask_samp_rate",   "hardware", "sample_rate",        float),
    CliMapping("freq_correction", "hardware", "freq_correction",    int),
    CliMapping("freq_spec",       "hardware", "center_frequencies", list),

    # Gains
    CliMapping("agc",                 "gains", "agc",        bool),
    CliMapping("rf_gain_db",          "gains", "rf",         float),
    CliMapping("lna_gain_db",         "gains", "lna",        float),
    CliMapping("mix_gain_db",         "gains", "mix",        float),
    CliMapping("if_gain_db",          "gains", "if_gain",    float),
    CliMapping("bb_gain_db",          "gains", "bb",         float),
    CliMapping("att_gain_db",         "gains", "att",        float),
    CliMapping("lna_mix_bb_gain_db",  "gains", "lna_mix_bb", float),
    CliMapping("tia_gain_db",         "gains", "tia",        float),
    CliMapping("pga_gain_db",         "gains", "pga",        float),
    CliMapping("lb_gain_db",          "gains", "lb",         float),

    # Receiver
    CliMapping("num_demod",       "receiver", "demodulators",   int),
    CliMapping("type_demod",      "receiver", "mode",           int),
    CliMapping("squelch_db",      "receiver", "squelch_db",     int),
    CliMapping("threshold_db",    "receiver", "threshold_db",   int),
    CliMapping("channel_spacing", "receiver", "channel_spacing",int),
    CliMapping("quiet_timeout",   "scanner",  "quiet_timeout",  int),
    CliMapping("active_timeout",  "scanner",  "active_timeout", int),
    CliMapping("max_ctcss_tones", "receiver", "max_ctcss_tones",int),

    # Audio
    CliMapping("play",            "audio", "play",              bool),
    CliMapping("volume_db",       "audio", "volume_db",         int),
    CliMapping("record",          "audio", "record",            bool),
    CliMapping("audio_bps",       "audio", "bit_depth",         int),
    CliMapping("min_recording",   "audio", "min_recording_sec", float),
    CliMapping("max_recording",   "audio", "max_recording_sec", float),
    CliMapping("wav_dir",         "audio", "wav_dir",           str),
    CliMapping("file_metadata",   "audio", "file_metadata",
               lambda v: [f.strip().lower() for f in v.split(",") if f.strip()]
                         if isinstance(v, str) else list(v)),

    # Scanner Options
    CliMapping("auto_priority",    "scanner",        "auto_priority", bool),
    CliMapping("hold_scan_on",     "scanner",        "hold_scan_on",
               lambda v: {x.strip() for x in v.split(",") if x.strip()} if isinstance(v, str) else v),

    # Frequency Policies
    CliMapping("frequency_file_name", "frequency_policies", "file",             Path),
    CliMapping("disable_lockout",     "frequency_policies", "disable_lockout",  bool),
    CliMapping("disable_priority",    "frequency_policies", "disable_priority", bool),
    CliMapping("active_banks",        "frequency_policies", "active_banks",     list),

    # Display
    CliMapping("max_db",     "display", "max_db",     float),
    CliMapping("min_db",     "display", "min_db",     float),
    CliMapping("theme_file", "display", "theme_file", str),

    # Channel Activity
    CliMapping("activity_type",     "channel_activity", "type",         str),
    CliMapping("activity_dest",     "channel_activity", "dest",         str),
    CliMapping("activity_interval", "channel_activity", "interval_sec", int),

    # Logging
    CliMapping("log_level", "logging", "level", str),
    CliMapping("log_dest",  "logging", "dest",  str),
    CliMapping("log_file",  "logging", "file",  str),
]


class CLParser(object):
    """Command line parser supporting YAML config files and CLI flag overrides.

    Attributes:
        master_config (MasterHam2MonConfig): Validated typed master configuration object
        hw_args (string): Argument string to pass to hardware
        num_demod (int): Number of parallel demodulators
        frequency_params (FrequencyParams): Requested RF center frequency or range in Hz
        ask_samp_rate (int): Asking sample rate of hardware in sps (1E6 min)
        gain_config (GainConfig): SDR hardware gain configuration (user values and defaults)
        squelch_db (int): Squelch in dB
        volume_db (int): Volume in dB
        threshold_db (int): Threshold for channel detection in dB
        record (bool): Record audio to file if True
        play (bool): Play audio through speaker if True
        frequency_file_name (Path): Name of file with frequencies
        disable_lockout (bool): Disable locking out of channels
        disable_priority (bool): Disable prioritization out of channels
        auto_priority (bool): Automatically set priority channels
        activity_dest (string): Name of file or endpoint for channel activity logging
        activity_type (string): Log file type for channel activity detection
        activity_interval (int): Timeout delay between active channel activity log entries
        freq_correction (int): Frequency correction in ppm
        audio_bps (int): Audio bit depth in bps
        max_db (float): Spectrum max dB for display
        min_db (float): Spectrum min dB for display
        channel_spacing (int): Channel spacing (spectrum bin size) for identification of channels
        min_recording (float): Minimum length of a recording in seconds
        max_recording (float): Maximum length of a recording in seconds
        log_level (str): Log verbosity level
        log_dest (str): Log destination
        log_file (str): Log file path
        file_metadata (list[str]): Output filename metadata fields
    """

    def __init__(self, args: Optional[List[str]] = None) -> None:
        # Pass 1: Extract -C / --config parameter
        pre_parser = ArgumentParser(add_help=False)
        pre_parser.add_argument("-C", "--config", type=Path, dest="config_file", default=None,
                                help="YAML configuration file path")
        if args is not None:
            pre_args, _ = pre_parser.parse_known_args(args)
        else:
            pre_args, _ = pre_parser.parse_known_args()

        raw_yaml: dict[str, object] = {}

        if pre_args.config_file is not None:
            try:
                resolved_config = resolve_config_path(pre_args.config_file, ".config.yaml")
                raw_yaml = load_raw_yaml(resolved_config)
            except ConfigError as err:
                pre_parser.error(str(err))

        # Strip any frequency_policies.file that was embedded in the general config:
        # the frequencies file must always be supplied explicitly via -F.
        freq_section = raw_yaml.get("frequency_policies")
        if isinstance(freq_section, dict) and "file" in freq_section:
            del freq_section["file"]

        # Pass 2: Main ArgumentParser
        parser = ArgumentParser(parents=[pre_parser])

        parser.add_argument("-a", "--args", type=str, dest="hw_args",
                          default=None, help="Hardware args")

        parser.add_argument("-n", "--demod", type=int, dest="num_demod",
                          default=None, help="Number of demodulators")

        parser.add_argument("-d", "--demodulator", type=int, dest="type_demod",
                          default=None, help="Type of demodulator (0=NBFM, 1=AM and 2=WBFM)")

        parser.add_argument("-f", "--freq", type=str, dest="freq_spec",
                          nargs='+', default=None,
                          help="Hardware RF center frequency or range in Mhz")

        parser.add_argument("--quiet-timeout", type=int,
                          dest="quiet_timeout", default=None,
                          help="Timeout when there is no activity")

        parser.add_argument("--active-timeout", type=int,
                          dest="active_timeout", default=None,
                          help="Timeout when there is activity")

        parser.add_argument("-r", "--rate", type=float, dest="ask_samp_rate",
                          default=None, help="Hardware ask sample rate in sps (1E6 minimum)")

        parser.add_argument("-g", "--gain", "--rf-gain", type=float, dest="rf_gain_db",
                          default=None, help="Hardware RF gain in dB")

        parser.add_argument("-i", "--if-gain", type=float, dest="if_gain_db", metavar="IF_GAIN",
                          default=None, help="Hardware IF gain in dB")

        parser.add_argument("-o", "--bb-gain", type=float, dest="bb_gain_db", metavar="BB_GAIN",
                          default=None, help="Hardware BB gain in dB")

        parser.add_argument("--lna-gain", type=float, dest="lna_gain_db",
                          default=None, help="Hardware LNA gain in dB")

        parser.add_argument("--att-gain", type=float, dest="att_gain_db",
                          default=None, help="Hardware ATT gain in dB")

        parser.add_argument("--lna-mix-bb-gain", type=float, dest="lna_mix_bb_gain_db",
                          default=None, help="Hardware LNA_MIX_BB gain in dB")

        parser.add_argument("--tia-gain", type=float, dest="tia_gain_db",
                          default=None, help="Hardware TIA gain in dB")

        parser.add_argument("--pga-gain", type=float, dest="pga_gain_db",
                          default=None, help="Hardware PGA gain in dB")

        parser.add_argument("--lb-gain", type=float, dest="lb_gain_db",
                          default=None, help="Hardware LB gain in dB")

        parser.add_argument("-x", "--mix-gain", type=float, dest="mix_gain_db",
                          default=None, help="Hardware MIX gain index")

        parser.add_argument("--agc", dest="agc", action="store_true", default=None,
                          help="Enable automatic gain control")

        parser.add_argument("-s", "--squelch", type=int,
                          dest="squelch_db", default=None,
                          help="Squelch in dB")

        parser.add_argument("-v", "--volume", type=int,
                          dest="volume_db", default=None,
                          help="Volume in dB")

        parser.add_argument("-t", "--threshold", type=int,
                          dest="threshold_db", default=None,
                          help="Threshold in dB")

        parser.add_argument("--banks", nargs="+", dest="active_banks", default=None,
                          help="Active scanner banks to monitor (e.g. --banks FRS_FAMILY SECURITY)")

        parser.add_argument("-w", "--write",
                          dest="record", action="store_true", default=None,
                          help="Record (write) channels to disk")

        parser.add_argument("-F", "--frequencies", type=Path,
                          dest="frequency_file_name", default=None,
                          help="YAML file containing frequencies and ranges in Mhz")

        parser.add_argument("--disable-lockout", action="store_true", default=None,
                          dest="disable_lockout", help="Disable locking out of channels")

        parser.add_argument("--disable-priority", action="store_true", default=None,
                          dest="disable_priority", help="Disable prioritization of channels")

        parser.add_argument("-P", "--auto-priority", action="store_true", default=None,
                          dest="auto_priority", help="Automatically add voice channels as priority channels")

        parser.add_argument("--hold-scan-on", type=str, dest="hold_scan_on", default=None,
                          help="Comma-separated transmission classifications to hold range scanning on (e.g. V,D)")

        parser.add_argument("-T", "--activity-type", type=str,
                          dest="activity_type", default=None,
                          help="Log file type for channel activity detection")

        parser.add_argument("-L", "--activity-dest", type=str,
                          dest="activity_dest", default=None,
                          help="Log file or endpoint for channel activity detection")

        parser.add_argument("-A", "--activity-interval", type=int,
                          dest="activity_interval", default=None,
                          help="Timeout delay for active channel activity log entries")

        parser.add_argument("-c", "--correction", type=int, dest="freq_correction",
                          default=None, help="Frequency correction in ppm")

        parser.add_argument("-m", "--mute-audio", dest="play",
                          action="store_false", default=None,
                          help="Mute audio from speaker (still allows recording)")

        parser.add_argument("-b", "--bps", type=int, dest="audio_bps",
                          default=None, help="Audio bit depth (bps)")

        parser.add_argument("-M", "--max-db", type=float, dest="max_db",
                          default=None, help="Spectrum window max dB for display")

        parser.add_argument("-N", "--min-db", type=float, dest="min_db",
                          default=None, help="Spectrum window min dB for display")

        parser.add_argument("-B", "--channel-spacing", type=int, dest="channel_spacing",
                          default=None, help="Channel spacing (spectrum bin size)")

        parser.add_argument("--min-recording", type=float, dest="min_recording",
                          default=None, help="Minimum length of a recording in seconds")

        parser.add_argument("--max-recording", type=float, dest="max_recording",
                          default=None, help="Maximum length of a recording in seconds")

        parser.add_argument("--wav-dir", type=str, dest="wav_dir",
                          default=None, help="Directory where recorded audio WAV files are saved")

        parser.add_argument("--theme-file", type=str, dest="theme_file",
                          default=None, help="Curses UI theme configuration file name")



        parser.add_argument("--log-level", dest="log_level",
                          choices=["debug", "info", "warn", "error"], default=None,
                          help="Log verbosity level")

        parser.add_argument("--log-dest", dest="log_dest",
                          choices=["none", "file", "syslog", "stderr"], default=None,
                          help="Log destination")

        parser.add_argument("--log-file", dest="log_file",
                          type=str, default=None, help="Log file path")

        parser.add_argument("--file-metadata", type=str,
                          dest="file_metadata", default=None,
                          help="Comma-separated list of metadata fields")

        parser.add_argument("--max-ctcss-tones", type=int,
                          dest="max_ctcss_tones", default=None,
                          help="Maximum number of CTCSS tones configured per frequency")

        parser.add_argument("--list-banks", action="store_true", default=False,
                          dest="list_banks",
                          help="Print each configured bank with its channel members, then exit without scanning")

        if args is not None:
            options = parser.parse_args(args)
        else:
            options = parser.parse_args()
        self.print_help = parser.print_help
        self.parser_args = parser.parse_args

        # Layer explicit CLI overrides over raw YAML
        # Resolve -F / --frequencies through the domain-aware resolver so that
        # bare basenames (e.g. "chicago") expand to "chicago.freqs.yaml".
        if options.frequency_file_name is not None:
            try:
                options.frequency_file_name = resolve_config_path(
                    options.frequency_file_name, ".freqs.yaml"
                )
            except ConfigError as err:
                parser.error(str(err))
        merged_dict = self._merge_cli_options(raw_yaml, options)

        # Build and validate typed MasterHam2MonConfig
        try:
            self.master_config: MasterHam2MonConfig = build_config_from_dict(merged_dict)
        except ConfigError as err:
            parser.error(str(err))

        self.frequency_params = self._build_frequency_params()
        self.list_banks = bool(options.list_banks)

    def _merge_cli_options(self, raw: dict[str, Any], options) -> dict[str, Any]:
        """Layer non-None CLI options onto the raw YAML dict, keyed by CLI_OPTION_MAP."""
        merged = dict(raw)
        for m in CLI_OPTION_MAP:
            value = getattr(options, m.option_attr)
            if value is None:
                continue
            section = merged.setdefault(m.section, {})
            if m.subsection is not None:
                section = section.setdefault(m.subsection, {})
            section[m.key] = m.caster(value)
        return merged

    def _build_frequency_params(self) -> FrequencyGroup:
        """Construct FrequencyGroup from master_config hardware center_frequencies and receiver timeouts."""
        cfg = self.master_config
        single_params: list[FrequencySingleParams] = []
        range_params: list[FrequencyRangeParams] = []
        for freq_entry in cfg.hardware.center_frequencies:
            if "-" in str(freq_entry):
                (lower_freq_str, upper_freq_str) = str(freq_entry).split('-')
                lower_freq = int(float(lower_freq_str) * 1E6)
                upper_freq = int(float(upper_freq_str) * 1E6)
                range_params.append(FrequencyRangeParams(lower_freq=lower_freq, upper_freq=upper_freq))
            else:
                single_freq = int(float(freq_entry) * 1E6)
                single_params.append(FrequencySingleParams(freq=single_freq))

        return FrequencyGroup(
            ranges=range_params,
            singles=single_params,
            sample_rate=int(cfg.hardware.sample_rate),
            quiet_timeout=cfg.scanner.quiet_timeout,
            active_timeout=cfg.scanner.active_timeout
        )


def main():
    """Test the parser"""
    parser = CLParser()
    cfg = parser.master_config

    print("hw_args:             " + cfg.hardware.args)
    print("num_demod:           " + str(cfg.receiver.demodulators))
    print("type_demod:          " + str(cfg.receiver.mode))
    single_freqs = [f'{single.freq}' for single in parser.frequency_params.singles]
    range_freqs = [f'{range.lower_freq}-{range.upper_freq}' for range in parser.frequency_params.ranges]
    print("single frequencies:  " + str(single_freqs))
    print("range frequencies:   " + str(range_freqs))
    print("quiet timeout:       " + str(cfg.scanner.quiet_timeout))
    print("active timeout:      " + str(cfg.scanner.active_timeout))
    print("ask_samp_rate:       " + str(cfg.hardware.sample_rate))
    from config import GAIN_FIELDS
    for field_name, hw_name in GAIN_FIELDS:
        val = cfg.gains.get_value(field_name)
        print('{0: <21}'.format(f"{hw_name} gain:") + str(val))
    print("agc:                 " + str(cfg.gains.agc))
    print("squelch_db:          " + str(cfg.receiver.squelch_db))
    print("volume_db:           " + str(cfg.audio.volume_db))
    print("threshold_db:        " + str(cfg.receiver.threshold_db))
    print("record:              " + str(cfg.audio.record))
    print("play:                " + str(cfg.audio.play))
    print("frequency_file_name: " + str(cfg.frequency_policies.file))
    print("activity_dest:       " + str(cfg.channel_activity.dest))
    print("activity_interval:   " + str(cfg.channel_activity.interval_sec))
    print("activity_type:       " + str(cfg.channel_activity.type))
    print("freq_correction:     " + str(cfg.hardware.freq_correction))
    print("audio_bps:           " + str(cfg.audio.bit_depth))
    print("max_db:              " + str(cfg.display.max_db))
    print("min_db:              " + str(cfg.display.min_db))
    print("channel_spacing:     " + str(cfg.receiver.channel_spacing))
    print("min_recording:       " + str(cfg.audio.min_recording_sec))
    print("max_recording:       " + str(cfg.audio.max_recording_sec))
    print("auto_priority:       " + str(cfg.scanner.auto_priority))
    print("disable_lockout:     " + str(cfg.frequency_policies.disable_lockout))
    print("disable_priority:    " + str(cfg.frequency_policies.disable_priority))
    print("log_level:           " + str(cfg.logging.level))
    print("log_dest:            " + str(cfg.logging.dest))
    print("log_file:            " + str(cfg.logging.file))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
