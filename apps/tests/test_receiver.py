import asyncio
import glob
import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from signal_generator import generate_test_iq

@pytest.mark.asyncio
async def test_squelch_activation(receiver_factory, tmp_path):
    """Test that NBFM squelch opens on an active carrier and saves a WAV file."""
    iq_file = tmp_path / "signal_active.iq"

    # 1. Generate 1.5 seconds of baseband IQ with a carrier at +50 kHz (active for 1.0s)
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.5,
        channels=[
            {
                "carrier_offset": 50_000,  # +50 kHz
                "amplitude": 1.0,
                "events": [(0.2, 1.2)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    # 2. Instantiate file-based Receiver
    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        center_freq=144_000_000,
        num_demod=1,
        type_demod=0,  # NBFM
        min_recording=0.2,
        record=True
    )
    assert rx.demodulators[0].__class__.__name__ == "TunerDemodNBFM"

    # Tune the demodulator to the offset frequency (+50 kHz => RF 144.0500 MHz)
    await rx.demodulators[0].set_center_freq(50_000, 144_000_000)
    rx.set_squelch(-50)

    # 3. Run flowgraph to completion
    rx.start()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # De-tune demodulator to trigger WAV file persistence
    await rx.demodulators[0].set_center_freq(0, 144_000_000)

    # 4. Verify output file creation
    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 1, (
        f"Expected 1 WAV in {rx._wav_dir}, found {wav_files}. "
        f"Unpersisted files in tmp/: {glob.glob(os.path.join(rx._wav_dir, 'tmp', '*.wav'))}"
    )

    filename = os.path.basename(wav_files[0])
    assert filename.startswith("144.0500_")

@pytest.mark.asyncio
async def test_squelch_muting_on_noise(receiver_factory, tmp_path):
    """Test that NBFM squelch remains closed on pure noise and saves no files."""
    iq_file = tmp_path / "signal_noise.iq"

    # 1. Generate 1.0 second of pure noise (no active channels)
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.0,
        channels=[],
        snr_db=50.0
    )
    iq_data.tofile(iq_file)

    # 2. Instantiate Receiver
    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        center_freq=144_000_000,
        num_demod=1,
        type_demod=0,
        min_recording=0.2,
        record=True
    )

    await rx.demodulators[0].set_center_freq(50_000, 144_000_000)
    rx.set_squelch(-30)  # Squelch above noise level

    # 3. Run flowgraph to completion
    rx.start()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # De-tune demodulator to trigger clean up
    await rx.demodulators[0].set_center_freq(0, 144_000_000)

    # 4. Verify no WAV files were saved (neither in parent directory nor in tmp)
    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    tmp_files = glob.glob(os.path.join(rx._wav_dir, "tmp", "*.wav"))
    assert len(wav_files) == 0, f"Expected 0 WAV files in {rx._wav_dir}, found {wav_files}."
    assert len(tmp_files) == 0, f"Unsquelched files left in tmp/: {tmp_files}"

@pytest.mark.asyncio
async def test_min_recording_duration_discard(receiver_factory, tmp_path):
    """Test that transmissions shorter than min_recording are discarded."""
    iq_file = tmp_path / "signal_short.iq"

    # 1. Generate carrier active for only 0.1s (under min_recording threshold)
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.0,
        channels=[
            {
                "carrier_offset": 50_000,
                "amplitude": 1.0,
                "events": [(0.2, 0.3)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    # 2. Instantiate Receiver with min_recording=0.5
    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        center_freq=144_000_000,
        num_demod=1,
        type_demod=0,
        min_recording=0.5,
        record=True
    )

    await rx.demodulators[0].set_center_freq(50_000, 144_000_000)
    rx.set_squelch(-50)

    # 3. Run flowgraph to completion
    rx.start()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # De-tune demodulator to trigger clean up/discard
    await rx.demodulators[0].set_center_freq(0, 144_000_000)

    # 4. Verify no final WAV file was saved (discarded as too short, neither in parent directory nor in tmp)
    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    tmp_files = glob.glob(os.path.join(rx._wav_dir, "tmp", "*.wav"))
    assert len(wav_files) == 0, f"Expected 0 WAV files in {rx._wav_dir}, found {wav_files}."
    assert len(tmp_files) == 0, f"Short recording was not discarded/cleaned, found in tmp/: {tmp_files}"


@pytest.mark.asyncio
async def test_zero_min_recording_empty_wav_discard(receiver_factory, tmp_path):
    """Test that when min_recording=0.0, an empty WAV file (header only) is discarded as short recording."""
    iq_file = tmp_path / "signal_empty.iq"
    iq_data = np.zeros(100_000, dtype=np.complex64)  # Quiet IQ noise
    iq_data.tofile(iq_file)

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        center_freq=144_000_000,
        num_demod=1,
        type_demod=0,
        min_recording=0.0,
        record=True
    )

    # Assign tuner and immediately de-tune without audio payload
    await rx.demodulators[0].set_center_freq(50_000, 144_000_000)
    await rx.demodulators[0].set_center_freq(0, 144_000_000)

    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    tmp_files = glob.glob(os.path.join(rx._wav_dir, "tmp", "*.wav"))
    assert len(wav_files) == 0, f"Expected 0 WAV files in {rx._wav_dir}, found {wav_files}."
    assert len(tmp_files) == 0, f"0-byte audio WAV was not discarded from tmp/: {tmp_files}"

@pytest.mark.asyncio
async def test_multi_channel_separation(receiver_factory, tmp_path):
    """Test processing multiple parallel channels simultaneously."""
    iq_file = tmp_path / "signal_multi.iq"

    # 1. Generate two carriers:
    # - Chan 0 at +100 kHz
    # - Chan 1 at -100 kHz
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.5,
        channels=[
            {
                "carrier_offset": 100_000,  # +100 kHz
                "amplitude": 1.0,
                "events": [(0.2, 1.2)]
            },
            {
                "carrier_offset": -100_000,  # -100 kHz
                "amplitude": 1.0,
                "events": [(0.3, 1.3)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    # 2. Instantiate Receiver with 2 demodulators
    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        center_freq=144_000_000,
        num_demod=2,
        type_demod=0,
        min_recording=0.2,
        record=True
    )

    # Tune demodulator 0 to +100 kHz (144.1000 MHz) and demodulator 1 to -100 kHz (143.9000 MHz)
    await rx.demodulators[0].set_center_freq(100_000, 144_000_000)
    await rx.demodulators[1].set_center_freq(-100_000, 144_000_000)
    rx.set_squelch(-50)

    # 3. Run flowgraph to completion
    rx.start()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # De-tune demodulators to trigger WAV file persistence
    await rx.demodulators[0].set_center_freq(0, 144_000_000)
    await rx.demodulators[1].set_center_freq(0, 144_000_000)

    # 4. Verify two distinct WAV files were saved
    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 2, (
        f"Expected 2 WAVs in {rx._wav_dir}, found {wav_files}. "
        f"Unpersisted files in tmp/: {glob.glob(os.path.join(rx._wav_dir, 'tmp', '*.wav'))}"
    )

    filenames = {os.path.basename(f) for f in wav_files}
    assert any(f.startswith("144.1000_") for f in filenames)
    assert any(f.startswith("143.9000_") for f in filenames)

@pytest.mark.asyncio
async def test_am_demodulation(receiver_factory, tmp_path):
    """Test that AM squelch opens on an AM modulated signal and saves a WAV file."""
    iq_file = tmp_path / "signal_am.iq"

    # Generate 1.5 seconds of carrier at +30 kHz (active for 1.0s)
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.5,
        channels=[
            {
                "carrier_offset": 30_000,  # +30 kHz
                "amplitude": 1.0,
                "audio_dev": 0.0,       # Pure carrier to modulate manually
                "events": [(0.2, 1.2)]
            }
        ],
        snr_db=30.0
    )

    # Manually modulate the amplitude at a 1 kHz rate to create AM modulation
    t = np.arange(0, len(iq_data)) / 1.0e6
    modulating_signal = 1.0 + 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)
    iq_data = (iq_data * modulating_signal).astype(np.complex64)
    iq_data.tofile(iq_file)

    # 2. Instantiate file-based Receiver
    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        center_freq=144_000_000,
        num_demod=1,
        type_demod=1,  # AM
        min_recording=0.2,
        record=True
    )
    assert rx.demodulators[0].__class__.__name__ == "TunerDemodAM"

    # Tune the demodulator to the offset frequency (+30 kHz => RF 144.0300 MHz)
    await rx.demodulators[0].set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)

    # 3. Run flowgraph to completion
    rx.start()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # De-tune demodulator to trigger WAV file persistence
    await rx.demodulators[0].set_center_freq(0, 144_000_000)

    # 4. Verify output file creation
    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 1, (
        f"Expected 1 WAV in {rx._wav_dir}, found {wav_files}. "
        f"Unpersisted files in tmp/: {glob.glob(os.path.join(rx._wav_dir, 'tmp', '*.wav'))}"
    )

    filename = os.path.basename(wav_files[0])
    assert filename.startswith("144.0300_")

@pytest.mark.asyncio
async def test_wbfm_demodulation(receiver_factory, tmp_path):
    """Test that WBFM squelch opens on a wideband FM signal and saves a WAV file."""
    iq_file = tmp_path / "signal_wbfm.iq"

    # Generate 1.5 seconds of WBFM signal at -100 kHz offset
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.5,
        channels=[
            {
                "carrier_offset": -100_000,
                "amplitude": 1.0,
                "audio_dev": 75_000,    # Wideband FM deviation
                "audio_freq": 1000.0,
                "events": [(0.2, 1.2)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    # 2. Instantiate file-based Receiver
    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        center_freq=144_000_000,
        num_demod=1,
        type_demod=2,  # WBFM
        min_recording=0.2,
        record=True
    )
    assert rx.demodulators[0].__class__.__name__ == "TunerDemodWBFM"

    # Tune the demodulator to the offset frequency (-100 kHz => RF 143.9000 MHz)
    await rx.demodulators[0].set_center_freq(-100_000, 144_000_000)
    rx.set_squelch(-50)

    # 3. Run flowgraph to completion
    rx.start()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # De-tune demodulator to trigger WAV file persistence
    await rx.demodulators[0].set_center_freq(0, 144_000_000)

    # 4. Verify output file creation
    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 1, (
        f"Expected 1 WAV in {rx._wav_dir}, found {wav_files}. "
        f"Unpersisted files in tmp/: {glob.glob(os.path.join(rx._wav_dir, 'tmp', '*.wav'))}"
    )

    filename = os.path.basename(wav_files[0])
    assert filename.startswith("143.9000_")

@pytest.mark.asyncio
async def test_squelch_clamping(receiver_factory, tmp_path):
    """Test that setting squelch clamps to valid bounds [-100, 0] dB."""
    iq_file = tmp_path / "dummy.iq"
    np.zeros(1000, dtype=np.complex64).tofile(iq_file)

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        record=False
    )

    # Test setting below lower bound (-100 dB)
    rx.set_squelch(-150)
    assert rx.squelch_db == -100
    assert rx.demodulators[0].analog_pwr_squelch_cc.threshold() == -100

    # Test setting above upper bound (0 dB)
    rx.set_squelch(20)
    assert rx.squelch_db == 0
    assert rx.demodulators[0].analog_pwr_squelch_cc.threshold() == 0

    # Test valid value
    rx.set_squelch(-45)
    assert rx.squelch_db == -45
    assert rx.demodulators[0].analog_pwr_squelch_cc.threshold() == -45

@pytest.mark.asyncio
async def test_volume_clamping(receiver_factory, tmp_path):
    """Test that setting volume clamps to valid bounds [-20, 20] dB."""
    iq_file = tmp_path / "dummy.iq"
    np.zeros(1000, dtype=np.complex64).tofile(iq_file)

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        record=False
    )

    # Test setting below lower bound (-20 dB)
    rx.set_volume(-50)
    assert rx.volume_db == -20

    # Test setting above upper bound (20 dB)
    rx.set_volume(50)
    assert rx.volume_db == 20

    # Test valid value
    rx.set_volume(10)
    assert rx.volume_db == 10

from config import GainConfig


@pytest.mark.asyncio
async def test_file_mode_hardware_guards(receiver_factory, tmp_path):
    """Test that hardware getters and gain setters handle file source gracefully."""
    iq_file = tmp_path / "dummy.iq"
    np.zeros(1000, dtype=np.complex64).tofile(iq_file)

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        record=False
    )

    # Verify hardware stubs
    assert rx.get_gain_names() == []
    assert rx.set_gains([{"name": "RF", "value": 10.0}]) == [{"name": "RF", "value": 10.0}]
    assert rx.filter_and_set_gains(GainConfig(rf=10.0)) == []


@pytest.mark.asyncio
async def test_filter_and_set_gains_validation(receiver_factory, tmp_path):
    """Test strict validation raises ValueError if explicit gain is not supported by hardware."""
    iq_file = tmp_path / "dummy.iq"
    np.zeros(1000, dtype=np.complex64).tofile(iq_file)

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        record=False
    )
    # Mock source_type to simulate real hardware with specific supported gain names
    rx._source_type = "hardware"
    rx.get_gain_names = lambda: ["LNA", "MIX", "IF"]

    # Explicitly requesting an unsupported gain (e.g. TIA) should raise ValueError
    with pytest.raises(ValueError, match="Gain\\(s\\) \\['TIA'\\] are not supported"):
        rx.filter_and_set_gains(GainConfig(lna=10.0, tia=8.0))


@pytest.mark.asyncio
async def test_filter_and_set_gains_agc(receiver_factory, tmp_path):
    """Test filter_and_set_gains skips gain setting when AGC is active."""
    iq_file = tmp_path / "dummy.iq"
    np.zeros(1000, dtype=np.complex64).tofile(iq_file)

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        record=False
    )
    rx._source_type = "hardware"
    rx.get_gain_names = lambda: ["LNA", "MIX", "IF"]

    # When agc=True, filter_and_set_gains should return []
    result = rx.filter_and_set_gains(GainConfig(agc=True, lna=10.0))
    assert result == []


@pytest.mark.asyncio
async def test_file_mode_tuning(receiver_factory, tmp_path):
    """Test that set_center_freq updates internal receiver state but doesn't attempt to tune hardware in file mode."""
    iq_file = tmp_path / "dummy.iq"
    np.zeros(1000, dtype=np.complex64).tofile(iq_file)

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        center_freq=144_000_000,
        num_demod=1,
        record=False
    )

    rx.set_center_freq(145_000_000)
    assert rx.center_freq == 145_000_000

@pytest.mark.asyncio
async def test_get_demod_freqs(receiver_factory, tmp_path):
    """Test that get_demod_freqs returns correct tuned baseband frequencies for all demodulators."""
    iq_file = tmp_path / "dummy.iq"
    np.zeros(1000, dtype=np.complex64).tofile(iq_file)

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=3,
        record=False
    )

    # Initially, demodulator frequencies default to 0 or setup default
    freqs = rx.get_demod_freqs()
    assert len(freqs) == 3

    # Tune a specific demodulator to verify it is tracked correctly
    await rx.demodulators[0].set_center_freq(50_000, 144_000_000)
    await rx.demodulators[2].set_center_freq(-20_000, 144_000_000)

    freqs = rx.get_demod_freqs()
    assert freqs[0] == 50_000
    assert freqs[1] == 0
    assert freqs[2] == -20_000

@pytest.mark.asyncio
async def test_fft_spectrum_probe(receiver_factory, tmp_path):
    """Test that probe_signal_vf correctly captures and returns the FFT spectrum level."""
    iq_file = tmp_path / "signal_probe.iq"

    # Generate 1.0s of active carrier at +50 kHz with high SNR
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.0,
        channels=[
            {
                "carrier_offset": 50_000,
                "amplitude": 1.0,
                "events": [(0.0, 1.0)]
            }
        ],
        snr_db=40.0
    )
    iq_data.tofile(iq_file)

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        record=False
    )

    rx.start()

    # Wait briefly for samples to process and populate the FFT probe
    await asyncio.sleep(0.2)

    # Read the spectrum level
    spectrum = rx.probe_signal_vf.level()
    rx.stop()
    rx.wait()

    # Assertions
    assert isinstance(spectrum, (list, tuple)), f"Expected tuple/list, got {type(spectrum)}"
    assert len(spectrum) == 256, f"Expected FFT length 256, got {len(spectrum)}"
    assert all(isinstance(val, float) for val in spectrum)

    # Verify that the spectrum has recorded signals (max power is non-zero)
    max_val = np.max(spectrum)
    assert max_val > 0.0, f"Expected non-zero spectrum power, got max {max_val}"


@pytest.mark.asyncio
async def test_ctcss_match(receiver_factory, tmp_path, monkeypatch):
    """Test that NBFM squelch opens when the correct CTCSS tone is present and saves a WAV file."""
    from gnuradio import blocks, gr
    from receiver import Receiver

    # Throttled file source for real-time timing verification
    def throttled_init_file_source(self, source_file, ask_samp_rate, center_freq):
        file_src = blocks.file_source(gr.sizeof_gr_complex, source_file, repeat=False)
        throttle = blocks.throttle(gr.sizeof_gr_complex, ask_samp_rate)
        self.connect(file_src, throttle)
        return throttle, ask_samp_rate, center_freq

    monkeypatch.setattr(Receiver, "_init_file_source", throttled_init_file_source)

    iq_file = tmp_path / "signal_ctcss_match.iq"

    # Generate NBFM signal at +30 kHz offset with CTCSS tone 100.0 Hz
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.5,
        channels=[
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 100.0,
                "ctcss_dev": 500.0,
                "events": [(0.2, 1.2)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    # Mock that returns 100.0 for any query (CTCSS configured to 100 Hz)
    def mock_ctcss_info(bb):
        return [100.0]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,  # NBFM
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    await rx.demodulators[0].set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)

    # Run flowgraph
    rx.start()

    # Wait for the signal to start playing (0.8s, well within active tone window of 0.2s-1.2s)
    # and confirm it matched (latching ctcss_matched = True)
    await asyncio.sleep(0.8)
    assert rx.demodulators[0].is_ctcss_mismatched() == False
    assert rx.demodulators[0].ctcss_matched == True

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # De-tune to persist WAV file
    await rx.demodulators[0].set_center_freq(0, 144_000_000)

    # Verify output file creation
    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 1, f"Expected 1 WAV, found {wav_files}"


@pytest.mark.asyncio
async def test_ctcss_matching_logic_mocked():
    """Unit test for the sticky CTCSS matching and mismatch logic using mocked states."""
    import time
    from unittest.mock import AsyncMock, MagicMock

    from demodulators.BaseTuner import BaseTuner

    # Create a dummy object representing BaseTuner
    self = MagicMock()
    self.channel = 1
    self._ctcss_enabled = True
    self._active_tone_count = 2  # simulate a 2-tone channel (e.g. primary + backup PL)
    self._active_tones = [100.0, 67.0]
    self.ctcss_matched = False
    self.ctcss_checked = False
    self.discard_current = False
    self._ctcss_start_time = time.time()
    self._CTCSS_GRACE_PERIOD_S = BaseTuner._CTCSS_GRACE_PERIOD_S
    self._CTCSS_MATCH_TOLERANCE_HZ = BaseTuner._CTCSS_MATCH_TOLERANCE_HZ
    self.ctcss_level = 0.0001

    # Bind the methods from BaseTuner
    self.is_ctcss_mismatched = BaseTuner.is_ctcss_mismatched.__get__(self)
    self.set_center_freq = BaseTuner.set_center_freq.__get__(self)

    # Mock the N parallel CTCSS chains and other requirements for set_center_freq.
    # Only indices < _active_tone_count are consulted by is_ctcss_mismatched.
    self._ctcss_squelches = [MagicMock(), MagicMock(), MagicMock()]
    self.analog_pwr_squelch_cc.unmuted.return_value = True
    for squelch in self._ctcss_squelches:
        squelch.unmuted.return_value = False

    # The hybrid matcher's authority is the full-band measurement. Start with "no
    # tone present"; individual steps below flip it to simulate detection.
    self._measure_ctcss_tone = lambda: (False, None)

    self.notify_scanner = AsyncMock()
    self._close_recording = MagicMock(return_value=None)
    self.freq_xlating_fir_filter_ccc = MagicMock()
    self.get_ctcss_info = None

    # 1. In grace period, should return False and set ctcss_checked = True
    assert self.is_ctcss_mismatched() == False
    assert self.ctcss_checked == True
    assert self.ctcss_matched == False

    # 2. Exceed grace period, no match yet, squelch open -> should return True (mismatch)
    self._ctcss_start_time = time.time() - (self._CTCSS_GRACE_PERIOD_S + 0.1)
    assert self.is_ctcss_mismatched() == True
    assert self.discard_current == True

    # 3. Simulate matching tone detection on the SECOND (backup) tone -> should set
    #    matched=True and return False, proving the check isn't limited to chain 0.
    #    The full-band measurement is the authority, so it must report the backup tone.
    self.ctcss_matched = False
    self.discard_current = False
    self._measure_ctcss_tone = lambda: (True, 67.0)
    self._ctcss_squelches[1].unmuted.return_value = True
    assert self.is_ctcss_mismatched() == False
    assert self.ctcss_matched == True
    assert self.matched_ctcss_tone == 67.0
    assert self.discard_current == False

    # 4. Once matched, even if that tone drops (unmuted=False) and grace period exceeded,
    #    should remain False (sticky!)
    self._ctcss_squelches[1].unmuted.return_value = False
    assert self.is_ctcss_mismatched() == False
    assert self.discard_current == False

    # 5. A third, unconfigured chain (index 2, beyond _active_tone_count=2) matching
    #    should NOT count -- is_ctcss_mismatched must only look at active slots.
    #    The measurement also reports a non-configured tone, so no match can latch.
    self.ctcss_matched = False
    self.discard_current = False
    self._measure_ctcss_tone = lambda: (True, 88.5)
    self._ctcss_start_time = time.time() - (self._CTCSS_GRACE_PERIOD_S + 0.1)
    self._ctcss_squelches[2].unmuted.return_value = True  # inactive slot -- should be ignored
    assert self.is_ctcss_mismatched() == True
    assert self.discard_current == True
    self._ctcss_squelches[2].unmuted.return_value = False

    # 6. Verify set_center_freq completed transmission discard logic
    self.record = True
    self._ctcss_enabled = True
    self.ctcss_checked = True
    self.center_freq = 30000
    self.file_name = "test.wav"

    # Case A: Checked but never matched -> should discard
    self.ctcss_matched = False
    self.discard_current = False
    await self.set_center_freq(0, 144000000)
    assert self.discard_current == True

    # Case B: Checked and matched -> should not discard
    self.center_freq = 30000
    self.file_name = "test.wav"
    self.ctcss_matched = True
    self.discard_current = False
    await self.set_center_freq(0, 144000000)
    assert self.discard_current == False


@pytest.mark.asyncio
async def test_ctcss_mismatch(receiver_factory, tmp_path):
    """Test that NBFM squelch remains closed and doesn't record when a mismatched CTCSS tone is present."""
    iq_file = tmp_path / "signal_ctcss_mismatch.iq"

    # Generate NBFM signal at +30 kHz offset with CTCSS tone 150.0 Hz
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.5,
        channels=[
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 150.0,   # Transmitted is 150 Hz
                "ctcss_dev": 500.0,
                "events": [(0.2, 1.2)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    # CTCSS configured to 100 Hz (mismatch!)
    def mock_ctcss_info(bb):
        return [100.0]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,  # NBFM
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    await rx.demodulators[0].set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)

    # Run flowgraph
    rx.start()

    # Wait for the signal to start playing (0.8s) and check mismatch
    await asyncio.sleep(0.8)
    assert rx.demodulators[0].is_ctcss_mismatched() == True

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # De-tune
    await rx.demodulators[0].set_center_freq(0, 144_000_000)

    # Verify no file is saved because the CTCSS block gated it
    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 0, f"Expected 0 WAV due to CTCSS mismatch, found {wav_files}"


@pytest.mark.asyncio
async def test_ctcss_mismatch_suppression(tmp_path):
    """Test that a mismatched CTCSS channel gets detuned and suppressed from reassignment."""
    from frequency_manager import FrequencyConfiguration, FrequencyManager
    from scanner import ChannelFrequency, Scanner

    class MockScanner(Scanner):
        def __init__(self, config, channel_spacing):
            self.squelch_db = -60
            self.volume_db = 0
            self.threshold_db = 10
            self.record = True
            self.play = False
            self.audio_bps = 8
            self.samp_rate = 1_000_000
            self.frequencies = []
            self.channels = []
            self._channels = []
            self.activity_params = None
            self.channel_spacing = channel_spacing
            self.hang_time = 1.0
            self.max_recording = 0.0
            self.file_metadata = []
            self._demod_signal_stats = {0: (0.0, 0)}
            self.mismatched_freqs = {}
            self.center_freq = 144_000_000

            self.frequency_manager = FrequencyManager(config, self.channel_spacing)
            self.frequency_manager.set_center(self.center_freq)

            class MockDemodulator:
                def __init__(self):
                    self.center_freq = 0
                    self._ctcss_enabled = True
                    self.last_heard = 0.0
                def is_ctcss_mismatched(self):
                    return True
                async def set_center_freq(self, bb, rf, avg_signal=None):
                    self.center_freq = bb

            class MockReceiver:
                def __init__(self):
                    self.demodulators = [MockDemodulator()]
                def get_demod_freqs(self):
                    return [d.center_freq for d in self.demodulators]

            self.receiver = MockReceiver()

    config = FrequencyConfiguration(file_name=None, disable_lockout=False, disable_priority=False, max_ctcss_tones=3)
    scanner = MockScanner(config, channel_spacing=5000)

    # Note: scanner.frequency_manager.get_ctcss_info is not directly called here since
    # MockDemodulator has a mocked mismatch status. The actual get_ctcss_info lookup
    # logic is verified in the end-to-end/flowgraph-level tests (test_ctcss_match and
    # test_ctcss_mismatch), but we add the priority frequency here with its CTCSS parameter
    # so the frequency manager has a realistic configuration.
    await scanner.frequency_manager.add({
        'single': 144.03,
        'priority': 1,
        'label': 'Priority 1',
        'tones': [100.0]
    })

    # Tune the mock demodulator to +30 kHz
    demod = scanner.receiver.demodulators[0]
    await demod.set_center_freq(30_000, scanner.center_freq)

    # 1. Process current demodulators - should detune demodulator and add RF frequency to mismatched_freqs
    active_channels = [
        ChannelFrequency(bb=30_000, rf=144.03, locked=False, active=True, hanging=False, priority=1)
    ]
    await scanner._process_current_demodulators(active_channels)

    # Demodulator should be detuned (center frequency 0)
    assert demod.center_freq == 0
    # RF frequency should be suppressed
    assert 144.03 in scanner.mismatched_freqs

    # 2. Try to assign channels - should skip 144.03 MHz because it is suppressed
    await scanner._assign_channels_to_demodulators(active_channels)
    # Demodulator should remain idle/detuned
    assert demod.center_freq == 0

    # 3. Clear mismatch suppression and try again - should assign 144.03 MHz to the idle demodulator
    scanner.mismatched_freqs.clear()
    await scanner._assign_channels_to_demodulators(active_channels)
    # Demodulator should now be tuned to the channel baseband frequency (30_000)
    assert demod.center_freq == 30_000


@pytest.mark.asyncio
async def test_ctcss_match_second_configured_tone(receiver_factory, tmp_path):
    """
    Test that a channel configured with TWO CTCSS tones (e.g. a repeater with a
    primary and backup PL tone) correctly matches and records when the SECOND
    (backup) tone is what's actually transmitted -- not just the first/primary.

    This is the real-signal counterpart to the 2-tone cases in
    test_ctcss_matching_logic_mocked / test_ctcss_dynamic_routing: those prove the
    Python bookkeeping is right, this proves chain 1's actual GNU Radio squelch
    block is correctly wired end-to-end (not just chain 0).
    """
    iq_file = tmp_path / "signal_ctcss_second_tone.iq"

    # Generate NBFM signal at +30 kHz offset with the BACKUP tone (67.0 Hz), not the primary (100.0 Hz)
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.5,
        channels=[
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 67.0,
                "ctcss_dev": 500.0,
                "events": [(0.2, 1.2)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    # Channel configured with two valid tones: primary 100.0 Hz, backup 67.0 Hz
    def mock_ctcss_info(bb):
        return [100.0, 67.0]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,  # NBFM
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    await rx.demodulators[0].set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)
    assert rx.demodulators[0]._active_tone_count == 2

    # Run flowgraph
    rx.start()

    # Wait for the signal to start playing and confirm it matched (via chain 1, not chain 0)
    await asyncio.sleep(0.5)
    assert rx.demodulators[0]._ctcss_squelches[0].unmuted() == False  # primary tone never present
    assert rx.demodulators[0]._ctcss_squelches[1].unmuted() == True   # backup tone is present
    assert rx.demodulators[0].is_ctcss_mismatched() == False

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # De-tune to persist WAV file
    await rx.demodulators[0].set_center_freq(0, 144_000_000)

    # Verify output file creation -- matched via the second configured tone
    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 1, f"Expected 1 WAV (matched via backup tone), found {wav_files}"


@pytest.mark.asyncio
async def test_ctcss_mismatch_with_two_configured_tones(receiver_factory, tmp_path):
    """
    Test that a channel configured with TWO CTCSS tones still correctly reports a
    mismatch (and discards the recording) when the transmitted tone matches
    NEITHER configured tone.
    """
    iq_file = tmp_path / "signal_ctcss_two_tone_mismatch.iq"

    # Transmitted tone (150.0 Hz) matches neither of the two configured tones
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.5,
        channels=[
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 150.0,
                "ctcss_dev": 500.0,
                "events": [(0.2, 1.2)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    # Channel configured with two valid tones, neither of which is 150.0 Hz
    def mock_ctcss_info(bb):
        return [100.0, 67.0]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,  # NBFM
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    await rx.demodulators[0].set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)
    assert rx.demodulators[0]._active_tone_count == 2

    # Run flowgraph
    rx.start()

    # Wait for the signal to start playing (0.8s) and check mismatch
    await asyncio.sleep(0.8)
    assert rx.demodulators[0]._ctcss_squelches[0].unmuted() == False
    assert rx.demodulators[0]._ctcss_squelches[1].unmuted() == False
    assert rx.demodulators[0].is_ctcss_mismatched() == True

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # De-tune
    await rx.demodulators[0].set_center_freq(0, 144_000_000)

    # Verify no file is saved -- neither configured tone matched
    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 0, f"Expected 0 WAV due to two-tone mismatch, found {wav_files}"


@pytest.mark.asyncio
async def test_ctcss_undefined(receiver_factory, tmp_path):
    """Test that NBFM squelch opens and saves a WAV file when a CTCSS provider is present
    but returns None (representing a channel with no CTCSS code configured)."""
    iq_file = tmp_path / "signal_ctcss_undefined.iq"

    # Generate NBFM signal at +30 kHz offset with CTCSS tone 67.0 Hz
    # (even though it has a tone, it should pass because it's not configured/bypassed)
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.5,
        channels=[
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 67.0,
                "ctcss_dev": 500.0,
                "events": [(0.2, 1.2)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    # Mock that returns None for any query (no CTCSS configured)
    def mock_ctcss_info(bb):
        return None

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,  # NBFM
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    await rx.demodulators[0].set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)

    # Run flowgraph
    rx.start()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # De-tune to persist WAV file
    await rx.demodulators[0].set_center_freq(0, 144_000_000)

    # Verify output file creation
    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 1, f"Expected 1 WAV, found {wav_files}"


@pytest.mark.asyncio
async def test_ctcss_dynamic_routing(receiver_factory, tmp_path):
    """Test that CTCSS selector routing correctly switches live when center frequency changes."""
    iq_file = tmp_path / "signal_empty.iq"
    np.zeros(1000, dtype=np.complex64).tofile(iq_file)

    # Mock that returns [100.0] Hz CTCSS for 144.03 MHz, [100.0, 67.0] for 144.06 MHz
    # (a second, multi-tone channel), and None for others
    def mock_ctcss_info(rf_freq):
        if abs(rf_freq - 144.03) < 1e-4:
            return [100.0]
        if abs(rf_freq - 144.06) < 1e-4:
            return [100.0, 67.0]
        return None

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,  # NBFM
        min_recording=0.2,
        record=False,
        get_ctcss_info=mock_ctcss_info
    )

    demod = rx.demodulators[0]

    # Before start: _is_started should be False, gains initialized to 1.0/0.0
    assert demod._is_started == False
    assert demod._ctcss_enabled == False

    # Tune to 30 kHz (144.030 MHz) -> CTCSS tone is [100.0] Hz.
    # Flowgraph is not started, so it should update _ctcss_enabled but NOT change gains (remains 1.0/0.0)
    await demod.set_center_freq(30_000, 144_000_000)
    assert demod._ctcss_enabled == True
    assert demod._active_tone_count == 1
    assert demod._bypass_gain.k() == 1.0
    assert demod._ctcss_gains[0].k() == 0.0

    # Start the receiver -> should update gains (bypass=0.0, chain 0=1.0) since _ctcss_enabled is True
    rx.start()
    assert demod._is_started == True
    assert demod._bypass_gain.k() == 0.0
    assert demod._ctcss_gains[0].k() == 1.0

    # Tune to 50 kHz (144.050 MHz) -> CTCSS tone is None (CSQ mode).
    # Since flowgraph is started, it should immediately switch gains to bypass (bypass=1.0, all ctcss chains=0.0)
    await demod.set_center_freq(50_000, 144_000_000)
    assert demod._ctcss_enabled == False
    assert demod._bypass_gain.k() == 1.0
    for gain in demod._ctcss_gains:
        assert gain.k() == 0.0

    # Tune back to 30 kHz (144.030 MHz) -> CTCSS tone is [100.0] Hz again.
    # Should immediately switch gains back to CTCSS (bypass=0.0, chain 0=1.0)
    await demod.set_center_freq(30_000, 144_000_000)
    assert demod._ctcss_enabled == True
    assert demod._bypass_gain.k() == 0.0
    assert demod._ctcss_gains[0].k() == 1.0

    # Tune to 60 kHz (144.060 MHz) -> two configured tones, [100.0, 67.0].
    # Both chain 0 and chain 1 should be active; chain 2 (unused slot) should stay off.
    await demod.set_center_freq(60_000, 144_000_000)
    assert demod._ctcss_enabled == True
    assert demod._active_tone_count == 2
    assert demod._bypass_gain.k() == 0.0
    assert demod._ctcss_gains[0].k() == 1.0
    assert demod._ctcss_gains[1].k() == 1.0
    assert demod._ctcss_gains[2].k() == 0.0

    rx.stop()
    rx.wait()


@pytest.mark.asyncio
async def test_ctcss_wbfm_match(receiver_factory, tmp_path, monkeypatch):
    """Test that WBFM squelch opens when the correct CTCSS tone is present and saves a WAV file."""
    from gnuradio import blocks, gr
    from receiver import Receiver

    # Throttled file source for real-time timing verification
    def throttled_init_file_source(self, source_file, ask_samp_rate, center_freq):
        file_src = blocks.file_source(gr.sizeof_gr_complex, source_file, repeat=False)
        throttle = blocks.throttle(gr.sizeof_gr_complex, ask_samp_rate)
        self.connect(file_src, throttle)
        return throttle, ask_samp_rate, center_freq

    monkeypatch.setattr(Receiver, "_init_file_source", throttled_init_file_source)

    iq_file = tmp_path / "signal_ctcss_wbfm_match.iq"

    # Generate WBFM signal at -100 kHz offset with CTCSS tone 100.0 Hz
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.5,
        channels=[
            {
                "carrier_offset": -100_000,
                "amplitude": 1.0,
                "audio_dev": 75_000,
                "audio_freq": 1000.0,
                "ctcss_freq": 100.0,
                "ctcss_dev": 1000.0,  # Slightly larger deviation for wideband channel
                "events": [(0.2, 1.2)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    # Mock that returns 100.0 for any query (CTCSS configured to 100 Hz)
    def mock_ctcss_info(bb):
        return [100.0]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=2,  # WBFM
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    await rx.demodulators[0].set_center_freq(-100_000, 144_000_000)
    rx.set_squelch(-50)

    # Run flowgraph
    rx.start()

    # Wait for the signal to start playing (0.8s, well within active tone window of 0.2s-1.2s)
    # and confirm it matched (latching ctcss_matched = True)
    await asyncio.sleep(0.8)
    assert rx.demodulators[0].is_ctcss_mismatched() == False
    assert rx.demodulators[0].ctcss_matched == True

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # De-tune to persist WAV file
    await rx.demodulators[0].set_center_freq(0, 144_000_000)

    # Verify output file creation
    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 1, f"Expected 1 WAV, found {wav_files}"


@pytest.mark.asyncio
async def test_ctcss_recording_contamination(receiver_factory, tmp_path, monkeypatch):
    """Test that the tail of a previous CTCSS transmission does not contaminate a subsequent recording."""
    import wave

    from gnuradio import blocks, gr
    from receiver import Receiver

    # Monkeypatch Receiver._init_file_source to add a throttle block so we can tune/detune in real-time
    def throttled_init_file_source(self, source_file, ask_samp_rate, center_freq):
        file_src = blocks.file_source(gr.sizeof_gr_complex, source_file, repeat=False)
        throttle = blocks.throttle(gr.sizeof_gr_complex, ask_samp_rate)
        self.connect(file_src, throttle)
        return throttle, ask_samp_rate, center_freq

    monkeypatch.setattr(Receiver, "_init_file_source", throttled_init_file_source)

    iq_file = tmp_path / "signal_ctcss_multi.iq"

    # Generate two NBFM signals at +30 kHz:
    # 1. 0.2s to 1.0s, audio = 1000 Hz, CTCSS = 100 Hz
    # 2. 1.8s to 2.6s, audio = 2000 Hz, CTCSS = 100 Hz
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=3.0,
        channels=[
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 100.0,
                "ctcss_dev": 500.0,
                "events": [(0.2, 1.0)]
            },
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 2000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 100.0,
                "ctcss_dev": 500.0,
                "events": [(1.8, 2.6)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    def mock_ctcss_info(bb):
        return [100.0]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,  # NBFM
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    # Initially tuned to +30 kHz
    await rx.demodulators[0].set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)

    # Start the flowgraph
    rx.start()

    # Simulate the scanner:
    # At t = 1.3s: first transmission is finished. Detune to 0 Hz to close the first file.
    await asyncio.sleep(1.3)
    await rx.demodulators[0].set_center_freq(0, 144_000_000)

    # At t = 1.7s (during silence): retune to +30 kHz to prepare for the second transmission (opens second file).
    await asyncio.sleep(0.4)
    await rx.demodulators[0].set_center_freq(30_000, 144_000_000)

    # At t = 2.8s: second transmission is finished. Detune to 0 Hz to close the second file.
    await asyncio.sleep(1.1)
    await rx.demodulators[0].set_center_freq(0, 144_000_000)

    # Wait for the flowgraph to finish processing the remainder of the 3s file
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # Verify both files exist
    wav_files = sorted(glob.glob(os.path.join(rx._wav_dir, "*.wav")))
    assert len(wav_files) == 2, f"Expected 2 WAVs, found {len(wav_files)}: {wav_files}"

    for idx, f in enumerate(wav_files):
        with wave.open(f, 'rb') as w:
            print(f"File {idx}: {f} has {w.getnframes()} frames, duration = {w.getnframes()/w.getframerate():.3f}s")

    # Read the second WAV file
    with wave.open(wav_files[1], 'rb') as w:
        frames = w.readframes(w.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

    # Check the start of the second recording (first 250 ms = 2000 samples at 8 kHz)
    audio_rate = w.getframerate()
    assert audio_rate == 8000
    check_len = int(audio_rate * 0.25)
    assert len(samples) >= check_len
    start_samples = samples[:check_len]

    # Compute FFT
    fft_vals = np.abs(np.fft.rfft(start_samples, n=4096))
    freqs = np.fft.rfftfreq(4096, d=1.0/audio_rate)

    # Measure energy at 1000 Hz and 2000 Hz
    e1000 = fft_vals[np.argmin(np.abs(freqs - 1000.0))]
    e2000 = fft_vals[np.argmin(np.abs(freqs - 2000.0))]

    # The 1000 Hz (previous transmission tone) should not be dominating the start of the second recording.
    # Specifically, the 2000 Hz tone (active in transmission 2) should be much stronger than the 1000 Hz leak.
    assert e2000 > e1000 * 2.0, f"Contamination detected! e1000 (leak) = {e1000:.2f}, e2000 (legit) = {e2000:.2f}"


@pytest.mark.asyncio
async def test_ctcss_sustained_match(receiver_factory, tmp_path, monkeypatch):
    """
    Test that CTCSS tone detection remains active/sustained throughout the transmission,
    and correctly goes back to False when the tone stops, using a throttled real-time source.
    """
    from gnuradio import blocks, gr
    from receiver import Receiver

    # Monkeypatch Receiver._init_file_source to add a throttle block for real-time simulation
    def throttled_init_file_source(self, source_file, ask_samp_rate, center_freq):
        file_src = blocks.file_source(gr.sizeof_gr_complex, source_file, repeat=False)
        throttle = blocks.throttle(gr.sizeof_gr_complex, ask_samp_rate)
        self.connect(file_src, throttle)
        return throttle, ask_samp_rate, center_freq

    monkeypatch.setattr(Receiver, "_init_file_source", throttled_init_file_source)

    iq_file = tmp_path / "signal_ctcss_sustained.iq"

    # Generate NBFM signal at +30 kHz: CTCSS = 100.0 Hz, active from 0.2s to 1.7s.
    # We extend duration to 2.8s so that after the transmission ends at 1.7s,
    # there is a full 1.1s of silence. This guarantees that at least one complete
    # Goertzel block (4000 samples = 0.5s) containing 100% silence is processed,
    # forcing the Goertzel detector to update and mute.
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=2.8,
        channels=[
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 100.0,
                "ctcss_dev": 500.0,
                "events": [(0.2, 1.7)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    def mock_ctcss_info(bb):
        return [100.0]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,  # NBFM
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    demod = rx.demodulators[0]
    await demod.set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)

    # Start flowgraph
    rx.start()

    # 1. At t=0.4s: tone has been present for 0.2s. Goertzel length is 4000 (0.5s),
    # so it should NOT have matched yet.
    await asyncio.sleep(0.4)
    assert demod._ctcss_squelches[0].unmuted() == False

    # 2. At t=0.9s: tone has been present for 0.7s. It should now be matched.
    await asyncio.sleep(0.5)
    assert demod._ctcss_squelches[0].unmuted() == True
    assert demod.is_ctcss_mismatched() == False

    # 3. At t=1.5s: tone has been present for 1.3s. It should remain matched.
    await asyncio.sleep(0.6)
    assert demod._ctcss_squelches[0].unmuted() == True
    assert demod.is_ctcss_mismatched() == False

    # 4. At t=2.6s: transmission ended at 1.7s (silence for 0.9s).
    # Squelch should now be correctly closed (muted) because a full silent block has completed.
    await asyncio.sleep(1.1)
    assert demod._ctcss_squelches[0].unmuted() == False

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # De-tune
    await demod.set_center_freq(0, 144_000_000)

    # Verify file saved
    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 1


@pytest.mark.asyncio
async def test_ctcss_adjacent_tone_rejection(receiver_factory, tmp_path, monkeypatch):
    """
    Test that a configured standard tone (e.g. 100.0 Hz) is NOT matched when an adjacent
    standard tone (e.g. 97.4 Hz, just 2.6 Hz away) is transmitted, proving the frequency
    selectivity of the larger Goertzel filter.
    """
    from gnuradio import blocks, gr
    from receiver import Receiver

    # Throttled file source for real-time timing verification
    def throttled_init_file_source(self, source_file, ask_samp_rate, center_freq):
        file_src = blocks.file_source(gr.sizeof_gr_complex, source_file, repeat=False)
        throttle = blocks.throttle(gr.sizeof_gr_complex, ask_samp_rate)
        self.connect(file_src, throttle)
        return throttle, ask_samp_rate, center_freq

    monkeypatch.setattr(Receiver, "_init_file_source", throttled_init_file_source)

    iq_file = tmp_path / "signal_ctcss_adjacent.iq"

    # Generate NBFM signal at +30 kHz: CTCSS = 97.4 Hz (adjacent neighbor to 100.0 Hz), active from 0.2s to 1.7s
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=2.2,
        channels=[
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 97.4,  # Transmitting adjacent tone
                "ctcss_dev": 500.0,
                "events": [(0.2, 1.7)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    # Configured to 100.0 Hz
    def mock_ctcss_info(bb):
        return [100.0]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,  # NBFM
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    demod = rx.demodulators[0]
    await demod.set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)

    # Start flowgraph
    rx.start()

    # Sleep 1.0s (transmission has been active for 0.8s, well beyond Goertzel/grace period)
    await asyncio.sleep(1.0)

    # Should remain muted (False) and be flagged as mismatched (True)
    assert demod._ctcss_squelches[0].unmuted() == False
    assert demod.is_ctcss_mismatched() == True

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # De-tune
    await demod.set_center_freq(0, 144_000_000)

    # Verify no file is saved
    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 0


@pytest.mark.asyncio
async def test_ctcss_match_high_pl_tone(receiver_factory, tmp_path, monkeypatch):
    """
    Test that a CTCSS tone in the upper part of the standard PL band (141.3 Hz --
    above the 55-105 Hz range that a wrongly-tuned detector could miss) matches
    correctly. The full-band (67-254.1 Hz) measurement is authoritative, so every
    standard PL code must work.
    """
    from gnuradio import blocks, gr
    from receiver import Receiver

    def throttled_init_file_source(self, source_file, ask_samp_rate, center_freq):
        file_src = blocks.file_source(gr.sizeof_gr_complex, source_file, repeat=False)
        throttle = blocks.throttle(gr.sizeof_gr_complex, ask_samp_rate)
        self.connect(file_src, throttle)
        return throttle, ask_samp_rate, center_freq

    monkeypatch.setattr(Receiver, "_init_file_source", throttled_init_file_source)

    iq_file = tmp_path / "signal_ctcss_high_tone.iq"
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.5,
        channels=[
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 141.3,
                "ctcss_dev": 500.0,
                "events": [(0.2, 1.2)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    def mock_ctcss_info(bb):
        return [141.3]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    demod = rx.demodulators[0]
    await demod.set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)

    rx.start()

    # Fast decision: matched well before the 0.7s grace period expires
    await asyncio.sleep(0.8)
    assert demod.is_ctcss_mismatched() == False
    assert demod.ctcss_matched == True
    assert demod.matched_ctcss_tone == 141.3

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    await demod.set_center_freq(0, 144_000_000)

    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 1, f"Expected 1 WAV (141.3 Hz match), found {wav_files}"


@pytest.mark.asyncio
async def test_ctcss_match_band_edge_tone(receiver_factory, tmp_path, monkeypatch):
    """
    Test that a CTCSS tone at the very top edge of the PL band (254.1 Hz) matches
    correctly, proving the measurement covers the full standard CTCSS band rather
    than a narrow sub-range.
    """
    from gnuradio import blocks, gr
    from receiver import Receiver

    def throttled_init_file_source(self, source_file, ask_samp_rate, center_freq):
        file_src = blocks.file_source(gr.sizeof_gr_complex, source_file, repeat=False)
        throttle = blocks.throttle(gr.sizeof_gr_complex, ask_samp_rate)
        self.connect(file_src, throttle)
        return throttle, ask_samp_rate, center_freq

    monkeypatch.setattr(Receiver, "_init_file_source", throttled_init_file_source)

    iq_file = tmp_path / "signal_ctcss_band_edge.iq"
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.5,
        channels=[
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 254.1,
                "ctcss_dev": 500.0,
                "events": [(0.2, 1.2)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    def mock_ctcss_info(bb):
        return [254.1]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    demod = rx.demodulators[0]
    await demod.set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)

    rx.start()

    await asyncio.sleep(0.8)
    assert demod.is_ctcss_mismatched() == False
    assert demod.ctcss_matched == True
    assert demod.matched_ctcss_tone == 254.1

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    await demod.set_center_freq(0, 144_000_000)

    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 1, f"Expected 1 WAV (254.1 Hz match), found {wav_files}"


@pytest.mark.asyncio
async def test_ctcss_adjacent_high_tone_rejection(receiver_factory, tmp_path, monkeypatch):
    """
    Test that a configured tone (141.3 Hz) is NOT matched when the adjacent standard
    tone (146.2 Hz, the next EIA code) is transmitted. This is the upper-band
    counterpart to test_ctcss_adjacent_tone_rejection and exercises the full-band
    relative-power authority at standard code spacing.
    """
    from gnuradio import blocks, gr
    from receiver import Receiver

    def throttled_init_file_source(self, source_file, ask_samp_rate, center_freq):
        file_src = blocks.file_source(gr.sizeof_gr_complex, source_file, repeat=False)
        throttle = blocks.throttle(gr.sizeof_gr_complex, ask_samp_rate)
        self.connect(file_src, throttle)
        return throttle, ask_samp_rate, center_freq

    monkeypatch.setattr(Receiver, "_init_file_source", throttled_init_file_source)

    iq_file = tmp_path / "signal_ctcss_adjacent_high.iq"
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.5,
        channels=[
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 146.2,
                "ctcss_dev": 500.0,
                "events": [(0.2, 1.2)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    def mock_ctcss_info(bb):
        return [141.3]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    demod = rx.demodulators[0]
    await demod.set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)

    rx.start()

    await asyncio.sleep(1.0)
    assert demod._ctcss_squelches[0].unmuted() == False
    assert demod.is_ctcss_mismatched() == True

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    await demod.set_center_freq(0, 144_000_000)

    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 0, f"Expected 0 WAV (146.2 Hz rejection), found {wav_files}"


@pytest.mark.asyncio
async def test_ctcss_adjacent_tone_rejection_150hz(receiver_factory, tmp_path, monkeypatch):
    """
    Test that in-band narrowband audio that is NOT a configured tone does not cause
    a false match (adjacent-tone rejection). A 150 Hz tone inside the CTCSS band is
    transmitted while the channel is configured to 100 Hz: the relative-power
    measurement measures 150 Hz, which matches no configured tone, so the channel
    reports a mismatch and nothing is recorded. (Broadband voice falsing is covered
    separately by test_ctcss_voice_falsing_rejection_broadband.)
    """
    from gnuradio import blocks, gr
    from receiver import Receiver

    def throttled_init_file_source(self, source_file, ask_samp_rate, center_freq):
        file_src = blocks.file_source(gr.sizeof_gr_complex, source_file, repeat=False)
        throttle = blocks.throttle(gr.sizeof_gr_complex, ask_samp_rate)
        self.connect(file_src, throttle)
        return throttle, ask_samp_rate, center_freq

    monkeypatch.setattr(Receiver, "_init_file_source", throttled_init_file_source)

    iq_file = tmp_path / "signal_ctcss_voice_falsing.iq"
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=1.5,
        channels=[
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 150.0,  # low-frequency audio inside the PL band, no PL tone
                "audio_dev": 3000.0,
                "events": [(0.2, 1.2)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    def mock_ctcss_info(bb):
        return [100.0]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    demod = rx.demodulators[0]
    await demod.set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)

    rx.start()

    await asyncio.sleep(1.0)
    assert demod.is_ctcss_mismatched() == True

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    await demod.set_center_freq(0, 144_000_000)

    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 0, f"Expected 0 WAV (voice falsing), found {wav_files}"


@pytest.mark.asyncio
async def test_ctcss_matched_tone_filename_metadata(receiver_factory, tmp_path, monkeypatch):
    """
    Test that if 'ctcss' is included in file_metadata, the matched tone is appended
    to the persisted filename (e.g. _100.0Hz.wav) and returned in ChannelMessage.
    """
    from gnuradio import blocks, gr
    from receiver import Receiver

    def throttled_init_file_source(self, source_file, ask_samp_rate, center_freq):
        file_src = blocks.file_source(gr.sizeof_gr_complex, source_file, repeat=False)
        throttle = blocks.throttle(gr.sizeof_gr_complex, ask_samp_rate)
        self.connect(file_src, throttle)
        return throttle, ask_samp_rate, center_freq

    monkeypatch.setattr(Receiver, "_init_file_source", throttled_init_file_source)

    iq_file = tmp_path / "signal_ctcss_metadata.iq"

    # Generate NBFM signal at +30 kHz: CTCSS = 100.0 Hz, active from 0.2s to 1.7s
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=2.2,
        channels=[
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 100.0,
                "ctcss_dev": 500.0,
                "events": [(0.2, 1.7)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    def mock_ctcss_info(bb):
        return [100.0]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,  # NBFM
        min_recording=0.2,
        record=True,
        file_metadata=["ctcss"],
        get_ctcss_info=mock_ctcss_info
    )

    demod = rx.demodulators[0]
    messages = []
    async def custom_notify(msg):
        if msg is not None:
            if msg.wav_tmp_path is not None:
                from config import MasterHam2MonConfig
                from conftest import make_test_scanner
                cfg = MasterHam2MonConfig()
                cfg.audio.file_metadata = ["ctcss"]
                scanner = make_test_scanner(config=cfg, wav_dir=rx._wav_dir)
                scanner._process_completed_transmission(msg)
            messages.append(msg)
    demod.notify_scanner = custom_notify

    await demod.set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)

    # Start flowgraph
    rx.start()

    # Sleep 1.0s to allow detection to match
    await asyncio.sleep(1.0)
    assert demod.is_ctcss_mismatched() == False
    assert demod.ctcss_matched == True
    assert demod.matched_ctcss_tone == 100.0

    # Detune to trigger file save
    await demod.set_center_freq(0, 144_000_000)

    # Wait for completion
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    # Check that message contains matched_ctcss
    xmit_msg = next((m for m in messages if m.state == 'off'), None)
    assert xmit_msg is not None
    assert xmit_msg.matched_ctcss == 100.0

    # Verify file saved has the CTCSS metadata suffix
    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 1
    filename = os.path.basename(wav_files[0])
    assert "_100.0Hz_" in filename


@pytest.mark.asyncio
async def test_max_ctcss_tones_configurable(receiver_factory, tmp_path, monkeypatch):
    """
    Test that the max_ctcss_tones parameter configured on Receiver is passed to BaseTuner
    and limits the number of tones evaluated.
    """
    # Create a dummy IQ file to satisfy file source initialization
    dummy_file = tmp_path / "dummy.iq"
    dummy_file.write_bytes(b'\x00' * 8000)

    # Create a receiver with max_ctcss_tones = 1
    def mock_ctcss_info(bb):
        return [100.0, 141.3]

    rx = receiver_factory(
        source_file=str(dummy_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info,
        max_ctcss_tones=1
    )

    demod = rx.demodulators[0]
    assert demod.max_ctcss_tones == 1
    assert len(demod._ctcss_squelches) == 1

    # Tune frequency to apply config
    await demod.set_center_freq(30_000, 144_000_000)
    assert demod._active_tone_count == 1
    assert demod._active_tones == [100.0]  # Second tone ignored


@pytest.mark.asyncio
async def test_ctcss_disabled_no_measurement_tap(receiver_factory, tmp_path):
    """max_ctcss_tones == 0 (the default config) must not build the full-band
    measurement tap at all -- no always-on FIR + vector sink -- and a normal
    tune/detune cycle must not crash even when a reported tone is truncated away.
    """
    # Create a dummy IQ file to satisfy file source initialization
    dummy_file = tmp_path / "dummy.iq"
    dummy_file.write_bytes(b'\x00' * 8000)

    # Reported tone is truncated away: only 0 tones are supported here
    def mock_ctcss_info(bb):
        return [100.0]

    rx = receiver_factory(
        source_file=str(dummy_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info,
        max_ctcss_tones=0
    )

    demod = rx.demodulators[0]
    assert demod.max_ctcss_tones == 0

    # Performance intent: no measurement infrastructure exists in this config,
    # so nothing runs an always-on FIR + vector-sink drain.
    for attr in ("_ctcss_capture", "_ctcss_capture_lp", "_ctcss_buffer",
                 "_ctcss_samples_seen", "_ctcss_confirm_min_len"):
        assert not hasattr(demod, attr)

    # Tune/detune exercises _apply_ctcss_config -> _reset_ctcss_measurement on
    # the guarded path (tone branch on tune, empty branch on detune).
    await demod.set_center_freq(30_000, 144_000_000)
    assert demod._ctcss_enabled is False
    assert demod._active_tone_count == 0
    assert demod.is_ctcss_mismatched() is False
    await demod.set_center_freq(0, 144_000_000)
    assert not hasattr(demod, "_ctcss_capture")


@pytest.mark.asyncio
async def test_ctcss_retune_stale_buffer_no_falsing(receiver_factory, tmp_path, monkeypatch):
    """
    Regression test for the stale-confirmation-buffer bug: the full-band
    measurement ring (_ctcss_buffer/_ctcss_samples_seen) must be reset on every
    retune, or a match/mismatch on the new channel can be decided from audio left
    over from the previous channel/transmission.

    Scenario: channel A (+30 kHz, configured 100.0 Hz) carries a wrong-PL
    transmission (88.5 Hz). The demodulator is then retuned to channel B
    (+40 kHz, configured 141.3 Hz) and B's valid 141.3 Hz transmission starts
    within the 2 s measurement-ring window of A's stale audio. B must still
    match 141.3 Hz -- the retune must clear the ring so B's decision never draws
    on A's leftover 88.5 Hz samples.
    """
    from gnuradio import blocks, gr
    from receiver import Receiver

    def throttled_init_file_source(self, source_file, ask_samp_rate, center_freq):
        file_src = blocks.file_source(gr.sizeof_gr_complex, source_file, repeat=False)
        throttle = blocks.throttle(gr.sizeof_gr_complex, ask_samp_rate)
        self.connect(file_src, throttle)
        return throttle, ask_samp_rate, center_freq

    monkeypatch.setattr(Receiver, "_init_file_source", throttled_init_file_source)

    iq_file = tmp_path / "signal_ctcss_retune_stale.iq"
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=3.8,
        channels=[
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 88.5,  # wrong PL for channel A (configured 100.0)
                "ctcss_dev": 500.0,
                "events": [(0.3, 1.4)]
            },
            {
                "carrier_offset": 40_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 141.3,  # correct PL for channel B
                "ctcss_dev": 500.0,
                "events": [(2.3, 3.5)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    def mock_ctcss_info(rf_mhz):
        if abs(rf_mhz - 144.03) < 1e-9:
            return [100.0]
        return [141.3]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    demod = rx.demodulators[0]
    await demod.set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)
    rx.start()

    # Poll through channel A's (wrong-PL) transmission so the measurement ring
    # fills with the stale 88.5 Hz audio and the validity floor is long crossed.
    for _ in range(14):
        demod.is_ctcss_mismatched()
        await asyncio.sleep(0.1)

    # Detune, park briefly, then retune to channel B just before its transmission
    # starts (still within the 2s measurement-ring window of A's stale audio).
    await demod.set_center_freq(0, 144_000_000)
    await asyncio.sleep(0.7)
    await demod.set_center_freq(40_000, 144_000_000)

    # The retune must have reset the measurement state: no stale samples may
    # survive into the new channel's dwell.
    assert demod._ctcss_samples_seen == 0
    assert np.count_nonzero(demod._ctcss_buffer) == 0

    # Poll through B's transmission: it must match 141.3 Hz from a clean ring --
    # well before the 0.7s grace period expires, not from A's leftover 88.5 Hz.
    matched = False
    for _ in range(10):
        await asyncio.sleep(0.1)
        demod.is_ctcss_mismatched()
        if demod.ctcss_matched:
            matched = True
            break

    assert matched, "channel B's 141.3 Hz tone was not matched from a clean measurement ring"
    assert demod.matched_ctcss_tone == 141.3
    assert demod.is_ctcss_mismatched() == False

    await demod.set_center_freq(0, 144_000_000)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)


@pytest.mark.asyncio
async def test_ctcss_direct_preempt_stale_buffer_no_falsing(receiver_factory, tmp_path, monkeypatch):
    """
    Regression test for the stale-confirmation-buffer bug via the scanner's direct
    preemption path (scanner.py: _assign_channels_to_demodulators calls
    demodulator.set_center_freq(channel.bb, ...) straight onto a demodulator that
    is mid-transmission, with no detune to 0 in between). This is the path where
    the measurement ring is hottest, and -- unlike the detour test -- it cannot be
    cleared by the empty-tones branch of _apply_ctcss_config, so it uniquely pins
    that the measurement reset fires on EVERY retune (non-empty -> non-empty too).

    Scenario: channel A (+30 kHz, configured 100.0 Hz) is carrying a wrong-PL
    transmission (88.5 Hz). While A is still live, the demodulator is preempted
    directly to channel B (+40 kHz, configured 141.3 Hz). The retune must clear the
    ring (and any undrained capture), so B's 141.3 Hz match never draws on A's
    leftover 88.5 Hz samples even though B starts inside the 2 s measurement window.
    """
    from gnuradio import blocks, gr
    from receiver import Receiver

    def throttled_init_file_source(self, source_file, ask_samp_rate, center_freq):
        file_src = blocks.file_source(gr.sizeof_gr_complex, source_file, repeat=False)
        throttle = blocks.throttle(gr.sizeof_gr_complex, ask_samp_rate)
        self.connect(file_src, throttle)
        return throttle, ask_samp_rate, center_freq

    monkeypatch.setattr(Receiver, "_init_file_source", throttled_init_file_source)

    iq_file = tmp_path / "signal_ctcss_direct_preempt.iq"
    iq_data = generate_test_iq(
        sample_rate=1.0e6,
        duration=3.2,
        channels=[
            {
                "carrier_offset": 30_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 88.5,  # wrong PL for channel A (configured 100.0)
                "ctcss_dev": 500.0,
                "events": [(0.3, 1.4)]
            },
            {
                "carrier_offset": 40_000,
                "amplitude": 1.0,
                "audio_freq": 1000.0,
                "audio_dev": 3000.0,
                "ctcss_freq": 141.3,  # correct PL for channel B
                "ctcss_dev": 500.0,
                "events": [(1.1, 3.0)]
            }
        ],
        snr_db=30.0
    )
    iq_data.tofile(iq_file)

    def mock_ctcss_info(rf_mhz):
        if abs(rf_mhz - 144.03) < 1e-9:
            return [100.0]
        return [141.3]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    demod = rx.demodulators[0]
    await demod.set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)
    rx.start()

    # Poll through channel A's (wrong-PL) transmission so the measurement ring
    # fills with live 88.5 Hz audio and the validity floor is long crossed.
    for _ in range(8):
        demod.is_ctcss_mismatched()
        await asyncio.sleep(0.1)

    # Preempt DIRECTLY from A to B mid-transmission -- no set_center_freq(0)
    # detour, mirroring scanner.py's direct preemption of a live demodulator.
    await demod.set_center_freq(40_000, 144_000_000)

    # The retune must have reset the measurement state unconditionally: neither
    # A's stale samples nor any undrained capture may survive into B's dwell.
    assert demod._ctcss_samples_seen == 0
    assert np.count_nonzero(demod._ctcss_buffer) == 0

    # Poll through B's transmission: it must match 141.3 Hz from a clean ring,
    # not from A's leftover 88.5 Hz audio.
    matched = False
    for _ in range(10):
        await asyncio.sleep(0.1)
        demod.is_ctcss_mismatched()
        if demod.ctcss_matched:
            matched = True
            break

    assert matched, "channel B's 141.3 Hz tone was not matched after direct preemption"
    assert demod.matched_ctcss_tone == 141.3
    assert demod.is_ctcss_mismatched() == False

    await demod.set_center_freq(0, 144_000_000)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)


@pytest.mark.asyncio
async def test_ctcss_voice_falsing_rejection_broadband(receiver_factory, tmp_path, monkeypatch):
    """
    Test that broadband, speech-like modulating audio does not cause a false CTCSS
    match. A voiced-speech stand-in (harmonic stack + noise, fundamental in-band at
    180 Hz, harmonics up to 2400 Hz) is transmitted with NO CTCSS tone while the
    channel is configured to 100 Hz. The 260 Hz low-pass guard strips the >260 Hz
    harmonics, and the 180 Hz fundamental matches no configured tone, so the
    channel reports a mismatch and nothing is recorded.
    """
    from gnuradio import blocks, gr
    from receiver import Receiver

    def throttled_init_file_source(self, source_file, ask_samp_rate, center_freq):
        file_src = blocks.file_source(gr.sizeof_gr_complex, source_file, repeat=False)
        throttle = blocks.throttle(gr.sizeof_gr_complex, ask_samp_rate)
        self.connect(file_src, throttle)
        return throttle, ask_samp_rate, center_freq

    monkeypatch.setattr(Receiver, "_init_file_source", throttled_init_file_source)

    sample_rate = 1.0e6
    duration = 1.5
    carrier_offset = 30_000
    t = np.arange(0, duration, 1.0 / sample_rate)

    audio = np.zeros_like(t)
    for freq, amp in [(180.0, 1.0), (360.0, 0.7), (720.0, 0.5),
                      (1080.0, 0.35), (1560.0, 0.25), (2400.0, 0.15)]:
        audio += amp * np.sin(2.0 * np.pi * freq * t)
    rng = np.random.default_rng(7)
    audio += 0.2 * rng.standard_normal(len(t))
    audio /= float(np.max(np.abs(audio)))

    active = (t >= 0.2) & (t <= 1.2)
    phase = (2.0 * np.pi * carrier_offset * t
             + 2.0 * np.pi * 3000.0 * np.cumsum(audio) / sample_rate)
    iq = np.exp(1j * phase).astype(np.complex128)
    iq[~active] = 0.0
    iq /= np.max(np.abs(iq))
    iq_data = iq.astype(np.complex64)

    iq_file = tmp_path / "signal_ctcss_voice_broadband.iq"
    iq_data.tofile(iq_file)

    def mock_ctcss_info(bb):
        return [100.0]

    rx = receiver_factory(
        source_file=str(iq_file),
        sample_rate=1_000_000,
        num_demod=1,
        type_demod=0,
        min_recording=0.2,
        record=True,
        get_ctcss_info=mock_ctcss_info
    )

    demod = rx.demodulators[0]
    await demod.set_center_freq(30_000, 144_000_000)
    rx.set_squelch(-50)

    rx.start()

    await asyncio.sleep(1.0)
    assert demod.is_ctcss_mismatched() == True
    assert demod.ctcss_matched == False

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rx.wait)

    await demod.set_center_freq(0, 144_000_000)

    wav_files = glob.glob(os.path.join(rx._wav_dir, "*.wav"))
    assert len(wav_files) == 0, f"Expected 0 WAV (broadband voice falsing), found {wav_files}"

