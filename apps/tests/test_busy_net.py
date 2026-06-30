"""
Integration test that exercises the synthetic IQ testing framework with a
single, sustained scenario meant to resemble a "busy" real-world scanning
session rather than an isolated unit-test condition.

Unlike the focused tests in test_receiver.py (which each isolate
one mechanism -- squelch, CTCSS match/mismatch, multi-channel separation,
etc.), this test combines several of those mechanisms into one 10-second
recording with three demodulators that get reused across multiple separate
transmissions, mixed CTCSS/non-CTCSS channels, conversational pacing (gaps
between transmissions rather than wall-to-wall keyups), and per-speaker
signal characteristics that stay consistent for that speaker across the
scenario.

Each simulated speaker has both a fixed RF carrier amplitude (their "signal
strength," verified against the raw IQ file) and a fixed FM deviation (their
recovered "loudness," verified via FFT tone-energy in the WAV audio). These
are deliberately two separate knobs: FM demodulation recovers instantaneous
frequency and is insensitive to carrier amplitude -- that's the whole appeal
of FM. Raw WAV RMS is not used for loudness comparisons because a fixed
demodulator noise floor compresses the ratio between loud and quiet speakers
in a way that makes the threshold fragile; measuring energy at each speaker's
specific tone frequency sidesteps that entirely.

Although not required for the test itself, these frequencies are synced with
`doc/frequencies-example.yaml`.  See `doc/simulate-radio-tranmissions.md` for details.

Scenario narrative (synthetic labels, not real-world frequencies):

  RF centre: 462.550 MHz  |  IQ sample rate: 1 Msps  |  Duration: 10 s

  - "Local repeater output"    462.730 MHz, CSQ (no CTCSS), two speakers:
      base station (strong, hot modulation) and handheld (weaker, softer).
      Each trades two exchanges.

  - "Some dispatch"  462.400 MHz, CTCSS 100.0 Hz, two speakers:
      patrol unit and dispatch, tone-squelched repeater pair.

  - "General talkaround" 462.610 MHz, CTCSS 167.9 Hz:
      single brief weak/distant transmission, borrows the third demodulator.

  - Background QRM     462.290 MHz, CTCSS 71.9 Hz:
      present in the IQ but no demodulator is ever tuned to it -- purely
      for scene realism, not asserted on.

  - "Data telemetry burst link" 462.150 MHz, locked (no CTCSS):
      intermittent, packet-like bursts (short, evenly-spaced keyups rather
      than conversational speech) meant to read as an automated data
      link rather than a voice channel. Placed close to the edge of the
      1 Msps capture band rather than near the centre. Marked `locked:
      true` in frequencies-example.yaml, so a real scanner using that
      file would never tune to it -- this test mirrors that by never
      assigning a demodulator to it. The synthetic IQ generator applies
      no special filtering for locked status: the signal is mixed into
      the raw IQ exactly like every other channel, and the lockout is
      purely a scanning-policy decision, not a property of the RF itself.

The frequencies-example.yaml companion file describes these same frequencies
so that loading the saved IQ into the real ham2mon application shows correct
labels, CTCSS annotations, and priority in the Channel list.

Classification is explicitly disabled (receiver_factory's default), so this
test only exercises squelch, CTCSS routing, and recording behaviour,
never the ML classifier.

Loading the IQ file into ham2mon:
    uv run apps/ham2mon.py \\
        -a "file=<path>/signal_busy_net.iq,rate=1E6,repeat=true,throttle=true,freq=462.550E6" \\
        -r 1E6 -t 20 -d 0 -s -70 -v 20 -w -m -b 16 -n 3 \\
        -F doc/frequencies-example.yaml

    NOTE: rate= inside -a must match -r exactly (1E6). The file source
    driver uses rate= as its playback throttle; -r tells the flowgraph what
    sample rate to assume. A mismatch (e.g. rate=3E6 with -r 1E6) causes
    the file to play back 3x too fast relative to what the flowgraph expects.
"""

import pytest
import os
import glob
import wave
import asyncio
import numpy as np
from signal_generator import generate_test_iq


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------
# Center frequency and channel frequencies are defined here as absolute
# RF frequencies in MHz, matching the entries in frequencies-example.yaml
# one-for-one (e.g. WAREHOUSE_FREQ_MHZ == the "Warehouse Ops" `single:`
# value in that file). A human comparing the two files can match channels
# by frequency directly, with no offset arithmetic required.
#
# Internally, the synthetic IQ builder and the Receiver/demodulator APIs
# work in terms of a baseband offset (Hz, relative to RF_CENTER_FREQ), so
# _channel_offset_hz() below converts each absolute channel frequency
# (MHz) into that offset for internal use. All resulting offsets are well
# inside +/-500 kHz of a 1 Msps baseband and spaced far enough apart
# (>=60 kHz) to keep the NBFM channel filters (12.5 kHz BW) from bleeding
# into one another.

RF_CENTER_FREQ_MHZ = 462.550                        # MHz -- matches freq= in the -a CLI argument
RF_CENTER_FREQ = round(RF_CENTER_FREQ_MHZ * 1.0e6)   # Hz, derived for use by the Receiver API

WAREHOUSE_FREQ_MHZ    = 462.730   # CSQ
PATROL_FREQ_MHZ       = 462.400   # CTCSS 100.0 Hz
MAINTENANCE_FREQ_MHZ  = 462.610   # CTCSS 167.9 Hz
INTERFERENCE_FREQ_MHZ = 462.290   # untuned background QRM
DATA_LINK_FREQ_MHZ    = 462.150   # locked (yaml) -- intermittent data-like bursts, near band edge


def _channel_offset_hz(freq_mhz: float, center_mhz: float = RF_CENTER_FREQ_MHZ) -> int:
    """Converts an absolute RF channel frequency (MHz) into the baseband
    offset (Hz, relative to center_mhz) expected by generate_test_iq()'s
    "carrier_offset" field and by Receiver/TunerDemod.set_center_freq().
    """
    return round((freq_mhz - center_mhz) * 1.0e6)


WAREHOUSE_OFFSET    = _channel_offset_hz(WAREHOUSE_FREQ_MHZ)
PATROL_OFFSET       = _channel_offset_hz(PATROL_FREQ_MHZ)
MAINTENANCE_OFFSET  = _channel_offset_hz(MAINTENANCE_FREQ_MHZ)
INTERFERENCE_OFFSET = _channel_offset_hz(INTERFERENCE_FREQ_MHZ)
DATA_LINK_OFFSET    = _channel_offset_hz(DATA_LINK_FREQ_MHZ)

PATROL_CTCSS_HZ       = 100.0
MAINTENANCE_CTCSS_HZ  = 167.9
INTERFERENCE_CTCSS_HZ =  71.9   # present but never queried

SAMPLE_RATE = 1_000_000         # sps -- matches -r and rate= in CLI
DURATION    = 10.0              # seconds

# Per-speaker RF carrier amplitudes (signal strength at the antenna/front end,
# NOT recovered audio loudness -- FM demod is amplitude-insensitive).
WAREHOUSE_BASE_AMPLITUDE     = 1.00
WAREHOUSE_HANDHELD_AMPLITUDE = 0.40
PATROL_UNIT_AMPLITUDE        = 0.80
PATROL_DISPATCH_AMPLITUDE    = 1.00
MAINTENANCE_AMPLITUDE        = 0.28
INTERFERENCE_AMPLITUDE       = 0.55

# Per-speaker FM deviation (Hz) -- this IS what controls recovered audio
# loudness after quadrature demodulation, since demod output is proportional
# to instantaneous frequency deviation, not carrier amplitude.
WAREHOUSE_BASE_DEVIATION     = 3800.0   # hot mic / strong modulation
WAREHOUSE_HANDHELD_DEVIATION = 1800.0   # noticeably softer
PATROL_UNIT_DEVIATION        = 3000.0
PATROL_DISPATCH_DEVIATION    = 3000.0
MAINTENANCE_DEVIATION        = 2500.0
DATA_LINK_DEVIATION          = 2000.0   # steady deviation -- no "voice-like" wobble

# Audio tone frequencies (simulated "voice" tones). Also used as FFT bins
# when verifying loudness levels in the demodulated WAV files.
WAREHOUSE_BASE_TONE_HZ     =  800.0
WAREHOUSE_HANDHELD_TONE_HZ = 1400.0

# Data link: RF carrier amplitude and modulating "tone." A constant high
# audio frequency (well above any of the voice tones above, evocative of a
# modem/telemetry sub-carrier) plus short, evenly-spaced keyups -- rather
# than the sparse, variable-length events used for the voice channels --
# is what makes this channel read as automated/data-like instead of
# conversational.
DATA_LINK_AMPLITUDE = 0.50
DATA_LINK_TONE_HZ   = 2400.0


# ---------------------------------------------------------------------------
# IQ signal builder
# ---------------------------------------------------------------------------

def _build_scenario_iq() -> np.ndarray:
    """Builds the full 10 s synthetic IQ recording for the busy-net scenario."""
    channels = [
        # --- Warehouse Ops (CSQ) --------------------------------------------
        {
            "carrier_offset": WAREHOUSE_OFFSET,
            "amplitude": WAREHOUSE_BASE_AMPLITUDE,
            "audio_freq": WAREHOUSE_BASE_TONE_HZ,
            "audio_dev": WAREHOUSE_BASE_DEVIATION,
            "events": [(0.4, 2.1), (5.9, 7.8)],        # base keys up twice
        },
        {
            "carrier_offset": WAREHOUSE_OFFSET,
            "amplitude": WAREHOUSE_HANDHELD_AMPLITUDE,
            "audio_freq": WAREHOUSE_HANDHELD_TONE_HZ,
            "audio_dev": WAREHOUSE_HANDHELD_DEVIATION,
            "events": [(2.6, 3.6), (8.2, 9.5)],        # handheld replies twice
        },
        # --- Security Patrol (CTCSS 100.0 Hz) -------------------------------
        {
            "carrier_offset": PATROL_OFFSET,
            "amplitude": PATROL_UNIT_AMPLITUDE,
            "audio_freq": 600.0,
            "audio_dev": PATROL_UNIT_DEVIATION,
            "ctcss_freq": PATROL_CTCSS_HZ,
            "ctcss_dev": 500.0,
            "events": [(1.0, 2.4), (6.3, 8.5)],
        },
        {
            "carrier_offset": PATROL_OFFSET,
            "amplitude": PATROL_DISPATCH_AMPLITUDE,
            "audio_freq": 1000.0,
            "audio_dev": PATROL_DISPATCH_DEVIATION,
            "ctcss_freq": PATROL_CTCSS_HZ,
            "ctcss_dev": 500.0,
            "events": [(2.9, 4.3)],
        },
        # --- Maintenance Crew (CTCSS 167.9 Hz) -- one brief transmission ----
        {
            "carrier_offset": MAINTENANCE_OFFSET,
            "amplitude": MAINTENANCE_AMPLITUDE,
            "audio_freq": 1800.0,
            "audio_dev": MAINTENANCE_DEVIATION,
            "ctcss_freq": MAINTENANCE_CTCSS_HZ,
            "ctcss_dev": 500.0,
            "events": [(4.6, 5.6)],
        },
        # --- Background QRM: present but no demodulator ever tuned to it ----
        {
            "carrier_offset": INTERFERENCE_OFFSET,
            "amplitude": INTERFERENCE_AMPLITUDE,
            "audio_freq": 2200.0,
            "audio_dev": 3000.0,
            "ctcss_freq": INTERFERENCE_CTCSS_HZ,
            "ctcss_dev": 500.0,
            "events": [(5.0, 9.0)],
        },
        # --- Data telemetry burst link: locked in frequencies-example.yaml --
        # No CTCSS (raw carrier keying, as with Warehouse Ops). Present in
        # the IQ exactly like every other channel -- locked status is a
        # scanning-policy decision applied later by ham2mon/the test
        # schedule below, never a filter applied at signal-generation time.
        # Short, uniformly-spaced keyups (rather than the sparse, variable
        # gaps used for voice) are what read as "data-like" here.
        {
            "carrier_offset": DATA_LINK_OFFSET,
            "amplitude": DATA_LINK_AMPLITUDE,
            "audio_freq": DATA_LINK_TONE_HZ,
            "audio_dev": DATA_LINK_DEVIATION,
            "events": [(start, start + 0.18) for start in np.arange(0.3, 10.0, 0.9)],
        },
    ]

    return generate_test_iq(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        channels=channels,
        snr_db=22.0,
    )


# ---------------------------------------------------------------------------
# WAV audio helpers
# ---------------------------------------------------------------------------

def _wav_tone_energy(path: str, tone_hz: float) -> float:
    """FFT magnitude at tone_hz for a recorded WAV file (int16 samples).

    Measuring energy at the specific audio tone frequency isolates each
    speaker's signal from the constant-ish demodulator noise floor.
    Raw RMS comparisons are fragile here: a shared noise floor compresses
    the loudness ratio between speakers far more than it compresses their
    absolute difference. Isolating the tone bin avoids that entirely.
    """
    with wave.open(path, "rb") as w:
        frames = w.readframes(w.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
        audio_rate = w.getframerate()
    if len(samples) == 0:
        return 0.0
    n_fft = 4096
    fft_vals = np.abs(np.fft.rfft(samples, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / audio_rate)
    return float(fft_vals[np.argmin(np.abs(freqs - tone_hz))])


def _wav_has_audio(path: str, min_peak_energy: float = 500.0) -> bool:
    """Returns True if the WAV file contains real demodulated audio.

    A file with audio shows a clear spectral peak well above the noise
    floor. A silence-only or noise-only file has uniformly low energy
    across all bins with no clear peaks.
    """
    with wave.open(path, "rb") as w:
        frames = w.readframes(w.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
        audio_rate = w.getframerate()
    if len(samples) == 0:
        return False
    fft_vals = np.abs(np.fft.rfft(samples, n=4096))
    return float(np.max(fft_vals)) > min_peak_energy


# ---------------------------------------------------------------------------
# Real-time tune/detune choreography
# ---------------------------------------------------------------------------
# (sleep_before_this_action_s, demod_index, baseband_offset_hz, narrative)
#
# demod 0 = Warehouse Ops, demod 1 = Security Patrol, demod 2 = Maintenance.
# offset=0 means "detune and persist the open recording."

_SCHEDULE = [
    (0.25, 0, WAREHOUSE_OFFSET,   "Warehouse base keys up"),
    (0.60, 1, PATROL_OFFSET,      "Patrol unit keys up"),
    (1.40, 0, 0,                  "Warehouse base unkeys -> persist"),
    (0.20, 0, WAREHOUSE_OFFSET,   "Warehouse handheld keys up"),
    (0.10, 1, 0,                  "Patrol unit unkeys -> persist"),
    (0.20, 1, PATROL_OFFSET,      "Dispatch replies"),
    (1.00, 0, 0,                  "Warehouse handheld unkeys -> persist"),
    (0.70, 1, 0,                  "Dispatch unkeys -> persist"),
    (0.00, 2, MAINTENANCE_OFFSET, "Maintenance tech keys up"),
    (1.30, 0, WAREHOUSE_OFFSET,   "Warehouse base keys up again"),
    (0.00, 2, 0,                  "Maintenance tech unkeys -> persist"),
    (0.40, 1, PATROL_OFFSET,      "Patrol unit keys up again"),
    (1.80, 0, 0,                  "Warehouse base unkeys -> persist"),
    (0.10, 0, WAREHOUSE_OFFSET,   "Warehouse handheld keys up again"),
    (0.60, 1, 0,                  "Patrol unit unkeys -> persist"),
    (1.00, 0, 0,                  "Warehouse handheld unkeys -> persist"),
]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_busy_net_realistic_scanning_session(receiver_factory, tmp_path, monkeypatch, run_busy_net):
    """Simulates a realistic, moderately busy scanning session.

    Three NBFM demodulators are choreographed across a 10 second recording
    containing three talkgroups plus background interference:

      - Warehouse Ops (CSQ): base station + handheld, two exchanges each.
      - Security Patrol (CTCSS 100.0 Hz): unit + dispatch.
      - Maintenance Crew (CTCSS 167.9 Hz): single brief transmission.
      - Background QRM (untuned): present in the IQ, never recorded.
      - Data telemetry burst link (locked in the yaml, untuned): intermittent
        packet-like keyups present in the IQ, never recorded.

    Each transmission gets its own WAV file (demod retuned away and back
    between transmissions, mirroring how Scanner segments recordings).

    Assertions:
      - Correct number and frequency-prefixed WAV files per talkgroup.
      - No WAV files for the untuned background interference frequency.
      - No WAV files for the locked data-link frequency.
      - Each WAV file contains real demodulated audio (not silence/noise).
      - Same-speaker recordings are consistent in loudness across their
        two transmissions (FFT tone-energy comparison).
      - Base station sounds clearly louder than handheld (higher FM deviation).
      - Base station RF carrier power exceeds handheld's in the raw IQ.
      - Locked data-link channel's carrier is present in the raw IQ during
        its keyups (proving lockout is a scanning policy, not an IQ filter).
    """
    from receiver import Receiver
    from gnuradio import blocks, gr

    def throttled_init_file_source(self, source_file, ask_samp_rate, center_freq):
        file_src = blocks.file_source(gr.sizeof_gr_complex, source_file, repeat=False)
        throttle = blocks.throttle(gr.sizeof_gr_complex, ask_samp_rate)
        self.connect(file_src, throttle)
        return throttle, ask_samp_rate, center_freq

    monkeypatch.setattr(Receiver, "_init_file_source", throttled_init_file_source)

    iq_file = tmp_path / "signal_busy_net.iq"
    iq_data = _build_scenario_iq()
    iq_data.tofile(iq_file)

    # CTCSS lookup mirrors FrequencyManager.get_ctcss_info for a frequencies
    # file with CTCSS configured on Patrol and Maintenance channels.
    # Units are MHz -- BaseTuner calls get_ctcss_info with the absolute RF
    # frequency in MHz (via baseband_to_frequency), not the baseband offset.
    patrol_rf_mhz      = (RF_CENTER_FREQ + PATROL_OFFSET)      / 1.0e6
    maintenance_rf_mhz = (RF_CENTER_FREQ + MAINTENANCE_OFFSET) / 1.0e6

    def mock_ctcss_info(rf_freq_mhz: float):
        if abs(rf_freq_mhz - patrol_rf_mhz) < 1e-4:
            return PATROL_CTCSS_HZ
        if abs(rf_freq_mhz - maintenance_rf_mhz) < 1e-4:
            return MAINTENANCE_CTCSS_HZ
        return None  # CSQ for Warehouse Ops and anything else

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=SAMPLE_RATE,
        center_freq=RF_CENTER_FREQ,
        num_demod=3,
        type_demod=0,           # NBFM
        min_recording=0.3,
        record=True,
        get_ctcss_info=mock_ctcss_info,
    )
    assert all(d.__class__.__name__ == "TunerDemodNBFM" for d in rx.demodulators)
    demods = rx.demodulators  # [Warehouse Ops, Security Patrol, Maintenance Crew]

    rx.set_squelch(-50)
    rx.start()
    loop = asyncio.get_running_loop()

    for delay, demod_idx, offset, _narrative in _SCHEDULE:
        if delay > 0:
            await asyncio.sleep(delay)
        await demods[demod_idx].set_center_freq(offset, RF_CENTER_FREQ)

    await loop.run_in_executor(None, rx.wait)

    # Safety net: ensure all demodulators are detuned so any open recording
    # is persisted before asserting on the output directory.
    for demod in demods:
        await demod.set_center_freq(0, RF_CENTER_FREQ)

    # -----------------------------------------------------------------------
    # Output file assertions
    # -----------------------------------------------------------------------
    wav_files = sorted(glob.glob(os.path.join(rx._wav_dir, "*.wav")))
    tmp_files = glob.glob(os.path.join(rx._wav_dir, "tmp", "*.wav"))
    assert tmp_files == [], f"Recordings left un-persisted in tmp/: {tmp_files}"

    # Expect exactly 8 persisted recordings, one per transmission:
    #   Warehouse Ops:    base#1, handheld#1, base#2, handheld#2   (4 files)
    #   Security Patrol:  unit#1, dispatch#1, unit#2               (3 files)
    #   Maintenance Crew: tech#1                                    (1 file)
    assert len(wav_files) == 8, (
        f"Expected 8 WAV files for the busy-net session, found {len(wav_files)}: "
        f"{[os.path.basename(f) for f in wav_files]}"
    )

    def _freq_str(offset: int) -> str:
        """WAV filename prefix for a given baseband offset."""
        return f"{(RF_CENTER_FREQ + offset) / 1e6:.4f}"

    warehouse_freq_str    = _freq_str(WAREHOUSE_OFFSET)
    patrol_freq_str       = _freq_str(PATROL_OFFSET)
    maintenance_freq_str  = _freq_str(MAINTENANCE_OFFSET)
    interference_freq_str = _freq_str(INTERFERENCE_OFFSET)
    data_link_freq_str    = _freq_str(DATA_LINK_OFFSET)

    basenames         = [os.path.basename(f) for f in wav_files]
    warehouse_files   = sorted(f for f in wav_files if os.path.basename(f).startswith(warehouse_freq_str))
    patrol_files      = sorted(f for f in wav_files if os.path.basename(f).startswith(patrol_freq_str))
    maintenance_files =        [f for f in wav_files if os.path.basename(f).startswith(maintenance_freq_str)]

    assert len(warehouse_files) == 4, (
        f"Expected 4 Warehouse Ops recordings; "
        f"found: {[os.path.basename(f) for f in warehouse_files]}"
    )
    assert len(patrol_files) == 3, (
        f"Expected 3 Security Patrol recordings; "
        f"found: {[os.path.basename(f) for f in patrol_files]}"
    )
    assert len(maintenance_files) == 1, (
        f"Expected 1 Maintenance Crew recording; "
        f"found: {[os.path.basename(f) for f in maintenance_files]}"
    )

    # No demodulator was ever tuned to the background interference frequency,
    # nor to the locked data-link frequency -- lockout is a scanning-policy
    # decision (nothing in this test ever assigns a demodulator to it),
    # not a filter applied to the recorded RF.
    assert not any(f.startswith(interference_freq_str) for f in basenames), (
        f"Background interference frequency ({interference_freq_str} MHz) must "
        f"never be recorded; no demodulator was tuned to it."
    )
    assert not any(f.startswith(data_link_freq_str) for f in basenames), (
        f"Locked data-link frequency ({data_link_freq_str} MHz) must never be "
        f"recorded; no demodulator was tuned to it."
    )

    # -----------------------------------------------------------------------
    # WAV content: each recorded file must contain real demodulated audio
    # -----------------------------------------------------------------------
    for wav_path in warehouse_files + patrol_files + maintenance_files:
        assert _wav_has_audio(wav_path), (
            f"Expected real demodulated audio in {os.path.basename(wav_path)}, "
            f"but the file appears to contain only silence or noise."
        )

    # -----------------------------------------------------------------------
    # Loudness consistency: same speaker sounds similar across transmissions
    # -----------------------------------------------------------------------
    # WAV filenames embed a UTC timestamp that sorts lexicographically in
    # chronological order, so warehouse_files[0..3] are:
    # base#1, handheld#1, base#2, handheld#2.
    base_tone = [
        _wav_tone_energy(warehouse_files[0], WAREHOUSE_BASE_TONE_HZ),
        _wav_tone_energy(warehouse_files[2], WAREHOUSE_BASE_TONE_HZ),
    ]
    handheld_tone = [
        _wav_tone_energy(warehouse_files[1], WAREHOUSE_HANDHELD_TONE_HZ),
        _wav_tone_energy(warehouse_files[3], WAREHOUSE_HANDHELD_TONE_HZ),
    ]

    assert base_tone[0] == pytest.approx(base_tone[1], rel=0.35), (
        f"Warehouse base station tone energy should be consistent across its "
        f"two transmissions; got {[round(v) for v in base_tone]}"
    )
    assert handheld_tone[0] == pytest.approx(handheld_tone[1], rel=0.35), (
        f"Warehouse handheld tone energy should be consistent across its "
        f"two transmissions; got {[round(v) for v in handheld_tone]}"
    )

    # -----------------------------------------------------------------------
    # Loudness distinction: base station is louder than handheld
    # -----------------------------------------------------------------------
    # Driven by WAREHOUSE_BASE_DEVIATION > WAREHOUSE_HANDHELD_DEVIATION.
    # FM demodulation is amplitude-insensitive, so this is purely a function
    # of how hard each speaker modulates the carrier, not how strong it is.
    avg_base_tone     = sum(base_tone)     / len(base_tone)
    avg_handheld_tone = sum(handheld_tone) / len(handheld_tone)
    assert avg_base_tone > avg_handheld_tone * 1.5, (
        f"Expected the base station to sound clearly louder than the handheld "
        f"(FM deviation ratio {WAREHOUSE_BASE_DEVIATION}/{WAREHOUSE_HANDHELD_DEVIATION}); "
        f"got base={avg_base_tone:.1f} handheld={avg_handheld_tone:.1f}"
    )

    # -----------------------------------------------------------------------
    # RF signal strength: verified in raw IQ, not in demodulated audio
    # -----------------------------------------------------------------------
    # Carrier amplitude (WAREHOUSE_BASE_AMPLITUDE vs WAREHOUSE_HANDHELD_AMPLITUDE)
    # sets pre-demodulation SNR/signal strength, not recovered loudness.
    # Windows are chosen to avoid overlap with other concurrently-active channels.
    raw_iq  = np.fromfile(iq_file, dtype=np.complex64)
    t_axis  = np.arange(len(raw_iq)) / SAMPLE_RATE
    lo      = np.exp(-1j * 2.0 * np.pi * WAREHOUSE_OFFSET * t_axis)
    baseband = raw_iq * lo

    def _channel_power(t0: float, t1: float) -> float:
        mask = (t_axis >= t0) & (t_axis <= t1)
        return float(np.mean(np.abs(baseband[mask]) ** 2))

    # Base windows: 0.5-0.95 s (before Patrol starts) and 6.0-6.2 s.
    # Handheld windows: 2.7-2.85 s and 8.3-9.4 s (Patrol on a different offset).
    base_power     = (_channel_power(0.5, 0.95) + _channel_power(6.0, 6.2)) / 2.0
    handheld_power = (_channel_power(2.7, 2.85) + _channel_power(8.3, 9.4)) / 2.0
    assert base_power > handheld_power * 2.0, (
        f"Expected the base station's RF carrier power to be clearly stronger "
        f"than the handheld's (amplitude ratio "
        f"{WAREHOUSE_BASE_AMPLITUDE}/{WAREHOUSE_HANDHELD_AMPLITUDE}); "
        f"got base={base_power:.4f} handheld={handheld_power:.4f}"
    )

    # -----------------------------------------------------------------------
    # Locked channel is present in the raw IQ, unfiltered
    # -----------------------------------------------------------------------
    # "Locked" is a scanning-policy attribute (frequencies-example.yaml,
    # never demodulated above) -- it must have no effect on signal
    # generation. Confirm the data-link carrier is genuinely present during
    # one of its burst windows and absent during a quiet gap between bursts,
    # proving nothing filtered it out of the IQ.
    data_link_lo       = np.exp(-1j * 2.0 * np.pi * DATA_LINK_OFFSET * t_axis)
    data_link_baseband = raw_iq * data_link_lo

    def _data_link_power(t0: float, t1: float) -> float:
        mask = (t_axis >= t0) & (t_axis <= t1)
        return float(np.mean(np.abs(data_link_baseband[mask]) ** 2))

    # Burst window: inside the first data-link keyup (0.3-0.48 s), but
    # before Warehouse Ops keys up at 0.4 s, so nothing else is active.
    # Gap window: before the first data-link burst even starts (0.3 s) and
    # before any other channel is active -- true silence except noise.
    # (A later gap, e.g. 0.60-0.90 s, is *not* clean: Warehouse Ops is
    # transmitting throughout it, and generate_test_iq's SNR-relative noise
    # floor rises with total concurrent signal power, so a "quiet" window
    # during someone else's transmission isn't actually quiet.)
    data_link_burst_power = _data_link_power(0.32, 0.39)   # inside a keyup, isolated
    data_link_gap_power   = _data_link_power(0.05, 0.25)   # true silence, isolated
    assert data_link_burst_power > data_link_gap_power * 5.0, (
        f"Expected the locked data-link channel's carrier to be clearly "
        f"present during a burst and absent between bursts, confirming it "
        f"was mixed into the IQ unfiltered despite being locked; "
        f"got burst={data_link_burst_power:.4f} gap={data_link_gap_power:.4f}"
    )

    rx.stop()
    rx.wait()
