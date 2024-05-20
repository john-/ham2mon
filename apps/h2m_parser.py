#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 15:21:33 2015

@author: madengr
"""

# from optparse import OptionParser
from argparse import ArgumentParser
from pathlib import Path
from channel_loggers import ChannelLogParams
from frequency_provider import FrequencyRangeParams, FrequencySingleParams, FrequencyGroup

class CLParser(object):
    """Command line parser

    Attributes:
        hw_args (string): Argument string to pass to hardware
        num_demod (int): Number of parallel demodulators
        frequency_params (FrequencyParams): Requested RF center frequency or range in Hz
        ask_samp_rate (int): Asking sample rate of hardware in sps (1E6 min)
        gains : Enumerated gain types and values
        squelch_db (int): Squelch in dB
        volume_dB (int): Volume in dB
        threshold_dB (int): Threshold for channel detection in dB
        record (bool): Record audio to file if True
        play (bool): Play audio through speaker if True
        lockout_file_name (string): Name of file with channels to lockout
        priority_file_name (string): Name of file with channels to for priority
        auto_priority (bool): Automatically set priority channels
        channel_log_target (string): Name of file or endpoint for channel logging
        channel_log_type (string): Log file type for channel detection
        channel_log_timeout (int): Timeout delay between active channel log entries
        freq_correction (int): Frequency correction in ppm
        audio_bps (int): Audio bit depth in bps
        max_db (float): Spectrum max dB for display
        min_db (float): Spectrum min dB for display
        channel_spacing (int): Channel spacing (spectrum bin size) for identification of channels
        freq_low (int): Low frequency for channels
        freq_high (int): High frequency for channels
        min_duration (float): Minimum length of a recording in seconds
    """
    # pylint: disable=too-few-public-methods
    # pylint: disable=too-many-instance-attributes

    def __init__(self) -> None:

        # Setup the parser for command line arguments
        parser = ArgumentParser()

        parser.add_argument("-a", "--args", type=str, dest="hw_args",
                          default='uhd',
                          help="Hardware args")

        parser.add_argument("-n", "--demod", type=int, dest="num_demod",
                          default=4,
                          help="Number of demodulators")

        parser.add_argument("-d", "--demodulator", type=int, dest="type_demod",
                          default=0,
                          help="Type of demodulator (0=NBFM, 1=AM and 2=WBFM)")

        parser.add_argument("-f", "--freq", type=str, dest="freq_spec",
                          nargs='+', default=["146000000"],
                          help="Hardware RF center frequency or range in Hz")
        
        parser.add_argument("--quiet_timeout", type=int,
                          dest="quiet_timeout", default=12,
                          help="Timeout when there is no activity")

        parser.add_argument("--active_timeout", type=int,
                          dest="active_timeout", default=20,
                          help="Timeout when there is activity")


        parser.add_argument("-r", "--rate", type=float, dest="ask_samp_rate",
                          default=4E6,
                          help="Hardware ask sample rate in sps (1E6 minimum)")

        parser.add_argument("-g", "--gain", "--rf_gain",  type=float, dest="rf_gain_db",
                          default=0, help="Hardware RF gain in dB")

        parser.add_argument("-i", "--if_gain", type=float, dest="if_gain_db",
                          default=16, help="Hardware IF gain in dB")

        parser.add_argument("-o", "--bb_gain", type=float, dest="bb_gain_db",
                          default=16, help="Hardware BB gain in dB")

        parser.add_argument("--lna_gain", type=float, dest="lna_gain_db",
                          default=8, help="Hardware LNA gain in dB")

        parser.add_argument("--att_gain", type=float, dest="att_gain_db",
                          default=8, help="Hardware ATT gain in dB")

        parser.add_argument("--lna_mix_bb_gain", type=float, dest="lna_mix_bb_gain_db",
                          default=8, help="Hardware LNA_MIX_BB gain in dB")

        parser.add_argument("--tia_gain", type=float, dest="tia_gain_db",
                          default=8, help="Hardware TIA gain in dB")

        parser.add_argument("--pga_gain", type=float, dest="pga_gain_db",
                          default=8, help="Hardware PGA gain in dB")

        parser.add_argument("--lb_gain", type=float, dest="lb_gain_db",
                          default=8, help="Hardware LB gain in dB")

        parser.add_argument("-x", "--mix_gain", type=float, dest="mix_gain_db",
                          default=5, help="Hardware MIX gain index")

        parser.add_argument("-s", "--squelch", type=int,
                          dest="squelch_db", default=-60,
                          help="Squelch in dB")

        parser.add_argument("-v", "--volume", type=int,
                          dest="volume_db", default=0,
                          help="Volume in dB")

        parser.add_argument("-t", "--threshold", type=int,
                          dest="threshold_db", default=10,
                          help="Threshold in dB")

        parser.add_argument("-w", "--write",
                          dest="record", default=False, action="store_true",
                          help="Record (write) channels to disk")

        parser.add_argument("-l", "--lockout", type=Path,
                          dest="lockout_file_name",
                          default="",
                          help="YAML lockout file containing frequencies and ranges in Mhz")

        parser.add_argument("-p", "--priority", type=str,
                          dest="priority_file_name",
                          default="",
                          help="File of EOL delimited priority channels in Hz (descending priority order)")

        parser.add_argument("-P", "--auto-priority", action="store_true",
                          dest="auto_priority",
                          help="Automatically add tuned channel as priority channel if it contains voice transmissions")

        parser.add_argument("-T", "--log_type", type=str,
                          dest="channel_log_type",
                          default="none",
                          help="Log file type for channel detection")

        parser.add_argument("-L", "--log_target", type=str,
                          dest="channel_log_target",
                          default="channel-log",
                          help="Log file or endpoint for channel detection")

        parser.add_argument("-A", "--log_active_timeout", type=int,
                          dest="channel_log_timeout",
                          default=15,
                          help="Timeout delay for active channel log entries")

        parser.add_argument("-c", "--correction", type=int, dest="freq_correction",
                          default=0,
                          help="Frequency correction in ppm")

        parser.add_argument("-m", "--mute-audio", dest="play",
                          action="store_false", default=True,
                          help="Mute audio from speaker (still allows recording)")

        parser.add_argument("-b", "--bps", type=int, dest="audio_bps",
                          default=16,
                          help="Audio bit depth (bps)")
        
        parser.add_argument("-M", "--max_db", type=float, dest="max_db",
                          default=50,
                          help="Spectrum window max dB for display")

        parser.add_argument("-N", "--min_db", type=float, dest="min_db",
                          default=-10,
                          help="Spectrum window min dB for display (no greater than -10dB from max")

        parser.add_argument("-B", "--channel-spacing", type=int, dest="channel_spacing",
                          default=5000,
                          help="Channel spacing (spectrum bin size)")

        parser.add_argument("--min_recording", type=float, dest="min_recording",
                          default=0,
                          help="Minimum length of a recording in seconds")

        parser.add_argument("--max_recording", type=float, dest="max_recording",
                          default=0,
                          help="Maximum length of a recording in seconds")

        parser.add_argument("--voice", dest="voice", action="store_true",
                          help="Record voice")
        
        parser.add_argument("--data", dest="data", action="store_true",
                          help="Record voice")

        parser.add_argument("--skip", dest="skip", action="store_true",
                          help="Record voice")  

        parser.add_argument("--debug", dest="debug", action="store_true",
                          help="Enable debug file with additional information (ham2mon.log)")              

        options = parser.parse_args()
        self.print_help = parser.print_help
        self.parser_args = parser.parse_args

        self.hw_args = str(options.hw_args)
        self.num_demod = int(options.num_demod)
        self.type_demod = int(options.type_demod)

        self.ask_samp_rate = int(options.ask_samp_rate)

        # this handles multiple -f option (frequency or frequency range in Hz)
        single_freq: int
        lower_freq: int
        upper_freq: int
        single_params: list[FrequencySingleParams] = []
        range_params: list[FrequencyRangeParams] = []
        self.frequency_params: FrequencyGroup
        for freq_entry in options.freq_spec:
            try:
                (lower_freq, upper_freq) = freq_entry.split('-')
                # there are 2 values provided
                try:
                    if lower_freq:
                        lower_freq = int(float(lower_freq))  # float first to handle scientific notation
                    if upper_freq:
                        upper_freq = int(float(upper_freq))
                except ValueError as err:
                    raise Exception(f'Frequencies must be integers: {err}')
                range_params.append(FrequencyRangeParams(lower_freq=lower_freq, upper_freq=upper_freq))
            except ValueError:
                # there is a single value provided
                try:
                    single_freq = int(float(freq_entry))
                    single_params.append(FrequencySingleParams(freq=single_freq))
                except ValueError as err:
                    raise Exception(f'Frequency must be integers: {err}')

        self.frequency_params =  FrequencyGroup(ranges=range_params, singles=single_params,
                                                sample_rate=self.ask_samp_rate,
                                                quiet_timeout=int(options.quiet_timeout),
                                                active_timeout=int(options.active_timeout))
        
        self.gains = [
            { "name": "RF", "value": float(options.rf_gain_db) },
            { "name": "LNA","value": float(options.lna_gain_db) },
            { "name": "MIX","value": float(options.mix_gain_db) },
            { "name": "IF", "value": float(options.if_gain_db) },
            { "name": "BB", "value": float(options.bb_gain_db) },
            { "name": "ATT", "value": float(options.att_gain_db) },
            { "name": "LNA_MIX_BB", "value": float(options.lna_mix_bb_gain_db) },
            { "name": "TIA", "value": float(options.tia_gain_db) },
            { "name": "PGA", "value": float(options.pga_gain_db) },
            { "name": "LB", "value": float(options.lb_gain_db) }
        ]
        self.squelch_db = int(options.squelch_db)
        self.volume_db = int(options.volume_db)
        self.threshold_db = int(options.threshold_db)
        self.record = bool(options.record)
        self.play = bool(options.play)
        self.lockout_file_name = Path(options.lockout_file_name)
        self.priority_file_name = str(options.priority_file_name)
        self.channel_log_params = ChannelLogParams(
            target=str(options.channel_log_target),
            type=str(options.channel_log_type),
            timeout=int(options.channel_log_timeout)
        )
        self.freq_correction = int(options.freq_correction)
        self.audio_bps = int(options.audio_bps)
        self.max_db = float(options.max_db)
        self.min_db = float(options.min_db)
        self.channel_spacing = int(options.channel_spacing)
        self.min_recording = float(options.min_recording)
        self.max_recording = float(options.max_recording)

        voice = bool(options.voice)        
        data = bool(options.data)
        skip = bool(options.skip)

        self.auto_priority = bool(options.auto_priority)
        if self.auto_priority:
            voice = True

        self.classifier_params = {'V': voice,
                                  'D': data,
                                  'S': skip
                                  }

        if voice or data or skip:
            self.record = True

        self.debug = bool(options.debug)

def main():
    """Test the parser"""

    parser = CLParser()

    print("hw_args:             " + parser.hw_args)
    print("num_demod:           " + str(parser.num_demod))
    print("type_demod:          " + str(parser.type_demod))
    single_freqs = [f'{single.freq}' for single in parser.frequency_params.singles]
    range_freqs = [f'{range.lower_freq}-{range.upper_freq}' for range in parser.frequency_params.ranges]
    print("single frequencies:  " + str(single_freqs))
    print("range frequencies:   " + str(range_freqs))
    print("quiet timeout:       " + str(parser.frequency_params.quiet_timeout))
    print("active timeout:      " + str(parser.frequency_params.active_timeout))
    print("ask_samp_rate:       " + str(parser.ask_samp_rate))
    for gain in parser.gains:
        #print(str(gain["value"]))
        print('{0: <21}'.format(gain["name"] + " gain:") + str(gain["value"]))
    print("squelch_db:          " + str(parser.squelch_db))
    print("volume_db:           " + str(parser.volume_db))
    print("threshold_db:        " + str(parser.threshold_db))
    print("record:              " + str(parser.record))
    print("play:                " + str(parser.play))
    print("lockout_file_name:   " + str(parser.lockout_file_name.name))
    print("priority_file_name:  " + str(parser.priority_file_name))
    print("channel_log target:  " + str(parser.channel_log_params.target))
    print("channel_log timeout: " + str(parser.channel_log_params.timeout))
    print("channel_log type:    " + str(parser.channel_log_params.type))
    print("freq_correction:     " + str(parser.freq_correction))
    print("audio_bps:           " + str(parser.audio_bps))
    print("max_db:              " + str(parser.max_db))
    print("min_db:              " + str(parser.min_db))
    print("channel_spacing:     " + str(parser.channel_spacing))
    print("min_recording:       " + str(parser.min_recording))
    print("max_recording:       " + str(parser.max_recording))
    print("voice:               " + str(parser.classifier_params['V']))
    print("data:                " + str(parser.classifier_params['D']))
    print("skip:                " + str(parser.classifier_params['S']))
    print("auto_priority:       " + str(parser.auto_priority))
    print("debug:               " + str(parser.debug))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

