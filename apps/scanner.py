#!/usr/bin/env python
"""
Created on Fri Jul  3 13:38:36 2015

@author: madengr
"""
# import yaml
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import estimate
import h2m_parser as prsr
import numpy as np
import receiver as recvr
from components.base import ChannelInfo
from components.manager import ComponentManager
from center_frequency_provider import FrequencyGroup, FrequencyProvider
from channel_loggers import ActivityLogger, ActivityParams, ChannelMessage
from config import GainConfig, MasterHam2MonConfig

from frequency_manager import (
    ChannelFrequency,
    ChannelList,
    FrequencyConfiguration,
    FrequencyList,
    FrequencyManager,
    TransmissionRecord,
)
from numpy.typing import NDArray
from utilities import (
    baseband_to_bin,
    baseband_to_frequency,
    bin_to_baseband,
    format_freq_mhz,
    format_timestamp,
    frequency_to_baseband,
    wav_bytes_per_sec,
    wav_duration_sec,
)


logger = logging.getLogger(f"ham2mon.{__name__}")

@dataclass(kw_only=True)
class ClassificationCount:
    V: int = field(default=0)
    D: int = field(default=0)
    S: int = field(default=0)

def _delete_file(path: str | None, context: str = "") -> None:
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError as e:
            msg_prefix = f"Failed to delete {context} file" if context else "Failed to delete file"
            logger.warning("%s %s: %s", msg_prefix, path, e)


class Scanner:
    """Scanner that controls receiver

    Estimates channels from FFT power spectrum that are above threshold
    Rounds channels to nearest 5 kHz
    Removes channels that are locked out
    Tunes demodulators to new channels  # gets overwritten by receiver value
    Holds demodulators on channels between scan cycles

    Args:
        ask_samp_rate (int): Asking sample rate of hardware in sps (1E6 min)
        num_demod (int): Number of parallel demodulators
        type_demod (int): Type of demodulator (0=NBFM, 1=AM)
        hw_args (string): Argument string to pass to hardware
        freq_correction (int): Frequency correction in ppm
        record (bool): Record audio to file if True
        auto_priority (bool): Automatically set priority channels
        frequency_configuration (FrequencyConfiguration): File name and other config information
        activity_params (ActivityParams): Parameters for logging channel activity
        audio_bps (int): Audio bit depth in bps (bits/samples)
        frequency_params (FrequencyGroup): Parameters for frequency provider
        spacing (int): granularity of frequency quantization
        min_recording (float): Minimum length of a recording in seconds
        max_recording (float): Maximum length of a recording in seconds
        auto_priority (bool): Automatically set priority channels
        agc (bool): Automatic gain control

    Attributes:
        center_freq (int): Hardware RF center frequency in Hz
        samp_rate (int): Hardware sample rate in sps (1E6 min)
        gains : Enumerated gain types and values
        squelch_db (int): Squelch in dB
        volume_dB (int): Volume in dB
        threshold_dB (int): Threshold for channel detection in dB
        spectrum (numpy.ndarray): FFT power spectrum data in linear, not dB
        frequencies (FrequencyList): List of frequencies including baseband values
        channel_spacing (float):  Spacing that channels will be rounded
        lockout_file_name (string): Name of file with channels to lockout
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments

    def __init__(self,
                 config: MasterHam2MonConfig | None = None,
                 frequency_params: FrequencyGroup | None = None):

        if config is None:
            config = MasterHam2MonConfig()
        if frequency_params is None:
            frequency_params = FrequencyGroup(sample_rate=int(config.hardware.sample_rate))

        self.config = config

        # Default values
        self.squelch_db = config.receiver.squelch_db
        self.volume_db = config.audio.volume_db
        self.threshold_db = config.receiver.threshold_db
        self.record = config.audio.record
        self.play = config.audio.play
        self.audio_bps = config.audio.bit_depth
        self.samp_rate: int
        self.frequency_params = frequency_params
        self.spectrum: NDArray[np.float64] = np.empty(0)
        self.frequencies: FrequencyList = []    # needed for the UI
        self.channels: ChannelList = []
        self._channels: ChannelList = []
        self.activity_params = ActivityParams(
            dest=config.channel_activity.dest,
            type=config.channel_activity.type,
            interval=config.channel_activity.interval_sec,
        )
        self.channel_spacing = config.receiver.channel_spacing
        self.frequency_file_name = config.frequency_policies.file  # used by the gui
        self.log_timeout_last = int(time.time())
        self.log_mode = ""
        self.hang_time: float = 1.0
        self.max_recording = config.audio.max_recording_sec
        self.xmit_stats: dict[float, ClassificationCount] = {}
        self.auto_priority = config.scanner.auto_priority
        self.file_metadata = list(config.audio.file_metadata)
        self._demod_signal_stats: dict[int, tuple[float, int]] = {i: (0.0, 0) for i in range(config.receiver.demodulators)}
        self.mismatched_freqs: dict[float, float] = {}

        self.activity_logger = ActivityLogger.get_logger(self.activity_params,
                                                        get_ctcss=self.get_matched_ctcss)

        frequency_configuration = FrequencyConfiguration(
            file_name=config.frequency_policies.file,
            disable_lockout=config.frequency_policies.disable_lockout,
            disable_priority=config.frequency_policies.disable_priority,
            max_ctcss_tones=config.receiver.max_ctcss_tones,
        )
        self.frequency_manager = FrequencyManager(frequency_configuration, self.channel_spacing)

        self._wav_dir = config.audio.wav_dir

        self._component_manager = ComponentManager(config)
        self._component_manager.start_all()


        # Create receiver object using Option B: pass specific sub-config objects
        self.receiver = recvr.Receiver(
            hardware_config=config.hardware,
            receiver_config=config.receiver,
            audio_config=config.audio,
            gain_config=config.gains,
            notify_scanner=self.got_channel_activity,
            get_ctcss_info=self.frequency_manager.get_ctcss_tones,
            wav_dir=self._wav_dir,
        )

        # Get the hardware sample rate
        self.samp_rate = self.receiver.samp_rate

        self.frequency_params.notify_scanner = self.center_freq_changed
        self.frequency_params.sample_rate = self.samp_rate  # update with hardware sample rate

        self.frequency_provider = FrequencyProvider(self.frequency_params)

        # Set the initial center frequency here to allow setting min/max
        self.center_freq = self.frequency_provider.center_freq
        self.receiver.set_center_freq(self.center_freq)
        # Get the hardware center frequency in Hz
        self.center_freq = self.receiver.center_freq
        # Frequency provider info is used by the interface
        self.step = self.frequency_provider.step
        self.steps = self.frequency_provider.steps

        # self.frequencies = self.frequency_manager.frequencies

        # Start the receiver and wait for samples to accumulate
        self.receiver.start()
        time.sleep(1)

        self.eof = False
        self._watch_thread = threading.Thread(target=self._wait_for_receiver_thread, daemon=True)
        self._watch_thread.start()

    def _wait_for_receiver_thread(self) -> None:
        try:
            self.receiver.wait()
            logger.info("Receiver flowgraph terminated (EOF reached)")
        except Exception as error:
            logger.error(f"Error waiting for receiver: {error}")
        finally:
            self.eof = True

    def center_freq_changed(self):
        '''
        Callback used to propagate provider value with self.  Also,
        notify the interface that the center frequency changed.
        '''
        self.set_center_freq(self.frequency_provider.center_freq)

        self.frequency_params.notify_interface()

    async def scan_cycle(self) -> None:
        """Execute one scan cycle

        Should be called no more than 10 Hz rate
        Estimates channels from FFT power spectrum that are above threshold
        Rounds channels to nearest 5 kHz
        Moves priority channels in front
        Tunes demodulators to new channels
        Holds demodulators on channels between scan cycles
        Add metadata to channels for GUI and further processing
        Log recent active channels
        """

        raw_channels = self._get_raw_channels()

        # Remove suppressed frequencies whose carrier is gone and minimum hold time has elapsed.
        # Note: If a carrier stays active continuously (e.g. repeater with incorrect tone), it will
        # remain suppressed forever. This is desired. If tones are reloaded or lockouts cleared,
        # mismatched_freqs is cleared entirely.
        raw_rf_channels = [baseband_to_frequency(bb, self.center_freq) for bb in raw_channels]
        the_now = time.time()
        for rf_freq in list(self.mismatched_freqs.keys()):
            if rf_freq not in raw_rf_channels and (the_now - self.mismatched_freqs[rf_freq] >= 3.0):
                del self.mismatched_freqs[rf_freq]

        # Accumulate signal levels for active demodulators if requested
        if 'strength' in self.file_metadata:
            for idx, demodulator in enumerate(self.receiver.demodulators):
                if demodulator.center_freq != 0:
                    strength = self._get_signal_strength(demodulator.center_freq)
                    curr_sum, curr_count = self._demod_signal_stats.get(idx, (0.0, 0))
                    self._demod_signal_stats[idx] = (curr_sum + strength, curr_count + 1)

        self._channels = self._add_metadata(raw_channels)

        await self._process_current_demodulators(self._channels)

        await self._assign_channels_to_demodulators(self._channels)

        self.channels = self._channels
        # logging.debug(f'{self._channels=}')


    def _get_raw_channels(self) -> NDArray[np.float64]:
        # Grab the FFT data, set threshold, and estimate baseband channels
        self.spectrum = self.receiver.probe_signal_vf.level()
        threshold = 10**(self.threshold_db/10.0)
        channels = np.array(
            estimate.channel_estimate(self.spectrum, threshold))

        # Convert channels from bin indices to baseband frequency in Hz
        channels = bin_to_baseband(channels, self.samp_rate, len(self.spectrum))

        # Round channels to channel spacing
        # Note this affects tuning the demodulators
        # 5000 Hz is adequate for NBFM
        # Note that channel spacing is with respect to the center + baseband offset,
        # not just the offset itself
        real_channels = channels + self.center_freq
        real_channels = np.round(
            real_channels / self.channel_spacing) * self.channel_spacing
        channels = (real_channels - self.center_freq).astype(int)
        # Remove 0 as this also represents an unassigned demodulator
        # As a result, valid signals at the center point will be ignored
        channels = channels[channels != 0]

        return channels

    def _get_signal_strength(self, bb: int) -> float:
        if self.spectrum is None or len(self.spectrum) == 0:
            return -100.0

        bin_idx = baseband_to_bin(bb, self.samp_rate, len(self.spectrum))

        power = self.spectrum[bin_idx]
        if power <= 0:
            return -200.0

        return 10.0 * np.log10(power) - 70.0

    async def _process_current_demodulators(self, channels: ChannelList) -> None:

        the_now = time.time()
        for idx in range(len(self.receiver.demodulators)):
            demodulator = self.receiver.demodulators[idx]
            if demodulator.center_freq == 0:
                continue

            # Stop locked out demodulator (lockout was just added via UI)
            if self.frequency_manager.locked_out(demodulator.center_freq):
                curr_sum, curr_count = self._demod_signal_stats.get(idx, (0.0, 0))
                avg_signal = int(np.round(curr_sum / curr_count)) if curr_count > 0 else None
                self._demod_signal_stats[idx] = (0.0, 0)
                await demodulator.set_center_freq(0, self.center_freq, avg_signal=avg_signal)
                continue

            # Stop demodulator if CTCSS tone is mismatched
            if demodulator.is_ctcss_mismatched():
                rf_freq = baseband_to_frequency(demodulator.center_freq, self.center_freq)
                if rf_freq not in self.mismatched_freqs:
                    self.mismatched_freqs[rf_freq] = the_now
                curr_sum, curr_count = self._demod_signal_stats.get(idx, (0.0, 0))
                avg_signal = int(np.round(curr_sum / curr_count)) if curr_count > 0 else None
                # Mark the current transmission for discard. While is_ctcss_mismatched() already
                # sets discard_current = True internally, setting it here explicitly is the
                # primary/canonical way of signaling that the scanner loop has rejected the
                # active transmission.
                self._demod_signal_stats[idx] = (0.0, 0)
                demodulator.discard_current = True
                await demodulator.set_center_freq(0, self.center_freq, avg_signal=avg_signal)
                continue

            # Stop the demodulator if not being scanned and outside the hang time
            if any(channel.hanging and channel.bb == demodulator.center_freq for channel in channels):
                if the_now - demodulator.last_heard > self.hang_time:
                    curr_sum, curr_count = self._demod_signal_stats.get(idx, (0.0, 0))
                    avg_signal = int(np.round(curr_sum / curr_count)) if curr_count > 0 else None
                    self._demod_signal_stats[idx] = (0.0, 0)
                    await demodulator.set_center_freq(0, self.center_freq, avg_signal=avg_signal)
            else:
                demodulator.set_last_heard(the_now)

            # Stop any long running modulators
            if self.max_recording > 0:
                if time.time() - demodulator.time_stamp >= self.max_recording:
                    # clear the demodulator to reset file
                    curr_sum, curr_count = self._demod_signal_stats.get(idx, (0.0, 0))
                    avg_signal = int(np.round(curr_sum / curr_count)) if curr_count > 0 else None
                    self._demod_signal_stats[idx] = (0.0, 0)
                    await demodulator.set_center_freq(0, self.center_freq, avg_signal=avg_signal)

    async def _assign_channels_to_demodulators(self, channels: ChannelList) -> None:

        # assign channels to available demodulators
        for channel in [channel for channel in channels if not channel.hanging]:
        #for channel in channels:
            # Skip if the channel is currently suppressed due to CTCSS mismatch
            if channel.rf in self.mismatched_freqs:
                continue

            # If channel not in demodulators
            if channel.bb not in self.receiver.get_demod_freqs() and not channel.locked:
                # Sequence through each demodulator
                for idx in range(len(self.receiver.demodulators)):
                    demodulator = self.receiver.demodulators[idx]
                    # If channel is higher priority than what is being demodulated
                    if self.frequency_manager.is_higher_priority(channel.bb, demodulator.center_freq):
                        # Calculate avg_signal for the completed transmission if it was actively running
                        avg_signal = None
                        if demodulator.center_freq != 0:
                            curr_sum, curr_count = self._demod_signal_stats.get(idx, (0.0, 0))
                            avg_signal = int(np.round(curr_sum / curr_count)) if curr_count > 0 else None
                        self._demod_signal_stats[idx] = (0.0, 0)

                        # Assigning channel to empty demodulator
                        await demodulator.set_center_freq(
                            channel.bb, self.center_freq, avg_signal=avg_signal)
                        break
                    else:
                        pass
            else:
                pass

    def _add_metadata(self, active_channels: NDArray) -> ChannelList:

        all_channels = active_channels   # start out with the active channels

        # If a demodulator is not in channel list than it is waiting for hang time to end
        # There is no activity on it so it was not in the scan
        demod_map = self.receiver.get_demod_freq_map()
        demod_freqs = list(demod_map.keys())
        for demod_freq in demod_freqs:
            if demod_freq != 0 and demod_freq not in all_channels:
               all_channels = np.append(all_channels, demod_freq)

        sweep: ChannelList = []
        for channel in all_channels:
            frequency = baseband_to_frequency(channel, self.receiver.center_freq)
            priority = self.frequency_manager.is_priority(channel)
            
            is_active = channel in demod_freqs and channel in active_channels
            is_hanging = channel in demod_freqs and channel not in active_channels

            matched_tone = None
            if channel in demod_map:
                matched_tone = demod_map[channel].matched_ctcss_tone

            idx = 0 if priority is not None else len(sweep)  # priority channels up front
            sweep.insert(idx, ChannelFrequency(bb=channel,
                                      rf=frequency,
                                      locked=self.frequency_manager.locked_out(channel),
                                      active=is_active,
                                      priority=priority,
                                      hanging=is_hanging,
                                      ctcss=self.frequency_manager.get_ctcss_info(frequency),
                                      matched_ctcss=matched_tone,
                                      label=self.frequency_manager.get_label(frequency, matched_tone),
                                      ctcss_tones=self.frequency_manager.get_ctcss_tones(frequency)))


        return sweep

    async def add_lockout(self, rf: float) -> None:
        """Lock out the given RF frequency.

        The RF value is supplied directly by the UI layer (translated from a
        pressed digit via ChannelWindow.get_rf_by_row) so no subset indexing
        is required here.
        """
        self.frequencies = await self.frequency_manager.change(
            {'single': rf, 'locked': True, 'mode': 'add'})

    async def clear_lockout(self) -> None:
        """
        Clears lockout channels and rebuilds based on config.  Usually called
        by the user interface ('l' key).
        """
        self.mismatched_freqs.clear()
        self.frequencies = await self.frequency_manager.load()

    async def load_frequencies(self) -> None:
        self.mismatched_freqs.clear()
        self.frequencies = await self.frequency_manager.load()

    def set_center_freq(self, center_freq: int) -> None:
        """Sets RF center frequency of hardware, update lockout
        baseband frequencies, and notify interface that things have changed

        Args:
            center_freq (int): Hardware RF center frequency in Hz
        """
        # Tune the receiver then update with actual frequency
        # and on frequency provider info
        self.receiver.set_center_freq(center_freq)
        self.center_freq = self.receiver.center_freq
        self.step = self.frequency_provider.step
        self.steps = self.frequency_provider.steps

        # Recreate baseband lockout since frequency is changing
        self.frequency_manager.set_center(self.center_freq)

    def filter_and_set_gains(self, gain_config: GainConfig) -> list[dict]:
        """Validate, filter, and set gains from a GainConfig against hardware support.

        Args:
            gain_config: GainConfig instance from CLParser
        Returns:
            list of dicts with hardware-confirmed {name, value} entries
        """
        self.gains = self.receiver.filter_and_set_gains(gain_config)
        return self.gains

    def set_gains(self, gains: list[dict]) -> list[dict]:
        """Set all the gains

        Args:
            gains (list of dictionary): Supported gains in dB
        """
        self.gains = self.receiver.set_gains(gains)
        return self.gains

    def set_squelch(self, squelch_db: int) -> None:
        """Sets squelch of all demodulators

        Args:
            squelch_db (int): Squelch in dB
        """
        self.receiver.set_squelch(squelch_db)
        self.squelch_db = self.receiver.squelch_db

    def set_volume(self, volume_db: int) -> None:
        """Sets volume of all demodulators

        Args:
            volume_db (int): Volume in dB
        """
        self.receiver.set_volume(volume_db)
        self.volume_db = self.receiver.volume_db

    def set_threshold(self, threshold_db: int) -> None:
        """Sets threshold in dB for channel detection

        Args:
            threshold_db (int): Threshold in dB
        """
        self.threshold_db = threshold_db

    def _process_completed_transmission(
            self, msg: ChannelMessage
    ) -> tuple[ChannelMessage, TransmissionRecord | None]:
        """Apply keep/discard decisions to a completed tmp WAV recording.

        Called from got_channel_activity() AFTER msg.label and msg.priority
        have been embellished by FrequencyManager.

        On success, builds an immutable TransmissionRecord capturing all
        transmission metadata and returns it as the second element of the
        tuple.  On discard (CTCSS mismatch, short recording, unwanted
        classification) the record is None.

        Returns:
            (msg, record): msg updated in-place with final file path and
            duration_sec; record is a populated TransmissionRecord when the
            WAV was kept, or None when discarded.
        """

        tmp_path = msg.wav_tmp_path
        msg.wav_tmp_path = None   # consumed

        if not tmp_path or not os.path.exists(tmp_path):
            return msg, None

        # 1. Pre-flagged discard (CTCSS mismatch set by BaseTuner)
        if msg.discard:
            msg.discard = False
            _delete_file(tmp_path, "mismatched CTCSS")
            msg.detail = 'Discarded mismatched CTCSS'
            return msg, None

        # 2. Minimum duration check: reject recordings shorter than min_recording_sec.
        # wav_bytes_per_sec encapsulates DEFAULT_AUDIO_RATE and bit-depth, keeping
        # this formula consistent with the duration_sec calculation below.
        bytes_per_sec = wav_bytes_per_sec(self.config.audio.bit_depth)
        min_size = 44 + bytes_per_sec * self.config.audio.min_recording_sec
        if os.stat(tmp_path).st_size <= min_size:
            _delete_file(tmp_path, "short recording")
            msg.detail = 'Discarded short recording'
            return msg, None

        # 3. Component evaluation (WavGatekeeper)
        classification: str | None = None
        comp_metadata: dict[str, object] = {}
        if self._component_manager.has_wav_component():

            info = ChannelInfo(
                rf=msg.rf,
                bb_hz=msg.bb,
                channel=msg.channel,
                label=msg.label,
                priority=msg.priority,
                matched_ctcss_hz=msg.matched_ctcss,
                signal_db=msg.signal_db,
                timestamp=msg.started_at or time.time(),
                wav_tmp_path=tmp_path,
            )
            res = self._component_manager.process_wav(tmp_path, info)
            classification = res.classification
            msg.classification = classification
            if res.detail:
                msg.detail = res.detail
            if res.metadata:
                comp_metadata = res.metadata
            if not res.keep:
                _delete_file(tmp_path, "unwanted classification")
                if not msg.detail:
                    msg.detail = "Discarded unwanted classification"
                return msg, None

        # 4. Build final filename using Scanner's config & shared helpers
        freq_str = format_freq_mhz(msg.rf)
        started_at = msg.started_at or time.time()
        tstamp_str = format_timestamp(started_at)

        name_parts = [freq_str]
        if classification:
            name_parts.append(classification)
        if 'priority' in self.config.audio.file_metadata:
            priority, is_auto = self.frequency_manager.get_priority_info(msg.bb)
            if priority is not None:
                name_parts.append("PA" if is_auto else f"P{priority}")
        if 'strength' in self.config.audio.file_metadata and msg.signal_db is not None:
            name_parts.append(f"{msg.signal_db}dB")
        if 'ctcss' in self.config.audio.file_metadata and msg.matched_ctcss is not None:
            name_parts.append(f"{msg.matched_ctcss:.1f}Hz")
        name_parts.append(tstamp_str)

        final_path = f'{self._wav_dir}/{"_".join(name_parts)}.wav'
        os.rename(tmp_path, final_path)
        msg.file = final_path

        # Compute duration from final WAV file size via shared utility helper.
        duration_sec = wav_duration_sec(
            os.stat(final_path).st_size, self.config.audio.bit_depth
        )
        msg.duration_sec = duration_sec

        # Write sidecar JSON file if component returned metadata
        if comp_metadata:
            sidecar_path = os.path.splitext(final_path)[0] + ".json"
            try:
                sidecar_payload = {
                    "rf": msg.rf,
                    "channel": msg.channel,
                    "label": msg.label,
                    "classification": classification,
                    "duration_sec": duration_sec,
                    "metadata": comp_metadata,
                }
                with open(sidecar_path, "w", encoding="utf-8") as sf:
                    json.dump(sidecar_payload, sf, indent=2)
            except Exception as se:
                logger.warning("Failed to write sidecar JSON %s: %s", sidecar_path, se)

        record = TransmissionRecord(
            rf=msg.rf,
            bb_hz=msg.bb,
            channel=msg.channel,
            label=msg.label,
            priority=msg.priority,
            matched_ctcss_hz=msg.matched_ctcss,
            signal_db=msg.signal_db,
            classification=classification,
            wav_path=final_path,
            started_at=started_at,
            duration_sec=duration_sec,
            metadata=comp_metadata,
        )
        return msg, record

    async def got_channel_activity(self, msg: ChannelMessage) -> None:
        '''
        This callback is to let the demodulators inform us about a
        transmission.

        1. Embellish metadata FIRST (label/priority)
        2. Process recording persistence / classification SECOND
        3. Log activity via channel logger
        4. If channel is interesting, notify frequency provider
        5. Assess if channel should be auto-priority
        '''

        if msg is None:
            return

        # 1. Embellish metadata FIRST so label/priority are present for persistence & classification
        msg.label = self.frequency_manager.get_label(msg.rf, msg.matched_ctcss)
        msg.priority = self.frequency_manager.is_priority(msg.bb)

        # 2. Process recording persistence / classification SECOND
        record: TransmissionRecord | None = None
        if msg.wav_tmp_path is not None:
            msg, record = self._process_completed_transmission(msg)

        logger.debug(str(msg))
        await self.activity_logger.log(msg, record)

        if self.interesting(msg):
            await self.frequency_provider.interesting_activity()
            if record is not None:
                await self._component_manager.dispatch_transmission(record)

        await self.priority_assess(msg.rf, msg.classification)

    def get_matched_ctcss(self, bb: int) -> float | None:
        '''Return the live matched CTCSS tone for the demodulator currently
        tuned to baseband frequency `bb`, or None if unknown/unmatched.
        Used by the channel logger to populate act-heartbeat messages.
        '''
        d = self.receiver.get_demod_freq_map().get(bb)
        return d.matched_ctcss_tone if d is not None else None

    def interesting(self, msg: ChannelMessage) -> bool:
        '''
        What is interesting?
        1.  If recording and a wav file was created
        2.  If not recording and channel was set to active
        '''
        if (self.record and msg.file is not None) or \
            (not self.record and msg.state == 'on'):
            return True
        else:
            return False

    def stop(self) -> None:
        """Stop the receiver
        """
        self.receiver.stop()
        self.receiver.wait()

    async def priority_assess(self, freq: float, classification: str | None) -> None:
        '''
        Track classification of transmisions and use the ratio of wanted/unwanted to
        set the priority.
        '''

        if not self.auto_priority:
            return

        if classification is None:  # ignore start of transission and thrown away short ones
            return

        if freq not in self.xmit_stats:
            self.xmit_stats[freq] = ClassificationCount()
            setattr(self.xmit_stats[freq], classification, 1)
        else:
            setattr(self.xmit_stats[freq], classification,
                    getattr(self.xmit_stats[freq], classification) + 1)

        bb_freq = frequency_to_baseband(float(freq), self.center_freq, self.channel_spacing)

        metrics: ClassificationCount = self.xmit_stats[freq]
        if metrics.V > metrics.D and metrics.V > metrics.S:  # Flag voice frequency as priority if not already set
            if self.frequency_manager.is_priority(bb_freq) is None:
                logger.debug(f'adding {freq=} to priority list')
                self.frequencies = await self.frequency_manager.change({'single': freq, 'priority': 1, 'mode': 'add'})
        else: # If not voice, remove from priority list if it currently a priority
            if self.frequency_manager.is_priority(bb_freq) is not None:
                logger.debug(f'removing {freq=} from the priority list')
                self.frequencies = await self.frequency_manager.change({'single': freq, 'priority': None, 'mode': 'add'})

    async def clean_up(self) -> None:
        await self._component_manager.stop_all_async()
        # cleanup terminating all demodulators

        for idx, demod in enumerate(self.receiver.demodulators):
            curr_sum, curr_count = self._demod_signal_stats.get(idx, (0.0, 0))
            avg_signal = int(np.round(curr_sum / curr_count)) if curr_count > 0 else None
            self._demod_signal_stats[idx] = (0.0, 0)
            await demod.set_center_freq(0, self.center_freq, avg_signal=avg_signal)


async def main() -> None:
    """Test the scanner

    Gets options from parser
    Sets up the scanner
    Assigns a channel to lockout
    Executes scan cycles
    Prints channels as they change
    """

    # Create parser object
    parser = prsr.CLParser()

    if not len(sys.argv) > 1:
        parser.print_help() #pylint: disable=maybe-no-member
        raise SystemExit(1)

    # Create scanner object
    scanner = Scanner(parser.master_config, parser.frequency_params)

    # Set frequency, gain, squelch, and volume
    print("\n")
    print("Started %s at %.3f Msps" % (parser.master_config.hardware.args, scanner.samp_rate/1E6))
    scanner.filter_and_set_gains(parser.master_config.gains)
    for gain in scanner.gains:
        print("gain %s at %d dB" % (gain["name"], gain["value"]))
    scanner.set_squelch(parser.master_config.receiver.squelch_db)
    scanner.set_volume(parser.master_config.audio.volume_db)
    print("%d demods of type %d at %d dB squelch and %d dB volume" % \
        (parser.master_config.receiver.demodulators, parser.master_config.receiver.mode, scanner.squelch_db, scanner.volume_db))

    # Create this empty list to allow printing to screen
    old_freqs: list[float] = []

    while 1:
        # No need to go faster than 10 Hz rate of GNU Radio probe
        await asyncio.sleep(0.1)

        # Execute a scan cycle
        await scanner.scan_cycle()

        # Print the tuned channels if they have changed

        channels =  scanner.channels

        freqs = [freq.rf for freq in channels if freq.active]

        freqs.sort()
        if freqs != old_freqs:
            print(freqs)
        old_freqs = freqs


if __name__ == '__main__':

    import asyncio

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
