import pytest
import os
import glob
import asyncio
import numpy as np
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

@pytest.mark.asyncio
async def test_file_mode_hardware_guards(receiver_factory, tmp_path):
    """Test that hardware adjustment methods safely stub out and return empty in file mode."""
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
    assert rx.filter_and_set_gains([{"name": "RF", "value": 10.0}]) == []

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
