import numpy as np

def generate_test_iq(
    sample_rate: float,
    duration: float,
    channels: list[dict],
    snr_db: float | None = None,
    seed: int | None = 42
) -> np.ndarray:
    """Generates complex64 baseband IQ data by superimposing multiple RF channels.

    Each channel dict in `channels` specifies:
      - 'carrier_offset': Frequency offset from baseband center in Hz (e.g. 100_000)
      - 'amplitude': Relative signal amplitude (default: 1.0)
      - 'audio_freq': Modulating tone frequency in Hz (default: 1000.0)
      - 'audio_dev': Frequency deviation in Hz (default: 3000.0)
      - 'ctcss_freq': Optional CTCSS sub-audible tone frequency in Hz (e.g. 100.0)
      - 'ctcss_dev': CTCSS frequency deviation in Hz (default: 500.0)
      - 'events': List of tuples (start_sec, end_sec) when this channel is active
    """
    t = np.arange(0, duration, 1.0 / sample_rate)
    iq_total = np.zeros_like(t, dtype=np.complex128)

    for chan in channels:
        carrier_offset = chan.get('carrier_offset', 0.0)
        amp = chan.get('amplitude', 1.0)
        audio_freq = chan.get('audio_freq', 1000.0)
        audio_dev = chan.get('audio_dev', 3000.0)
        ctcss_freq = chan.get('ctcss_freq', None)
        ctcss_dev = chan.get('ctcss_dev', 500.0)
        events = chan.get('events', [(0.0, duration)])

        # Create active mask based on event intervals
        active_mask = np.zeros_like(t, dtype=bool)
        for start, end in events:
            active_mask |= (t >= start) & (t <= end)

        # Base phase from carrier frequency offset
        phase = 2.0 * np.pi * carrier_offset * t

        # Add Audio Modulation (integral of frequency)
        # True FM phase is the integral of frequency offset:
        # integral(audio_dev * sin(2*pi*audio_freq*t)) = - (audio_dev / audio_freq) * cos(2*pi*audio_freq*t)
        phase += (audio_dev / audio_freq) * (-np.cos(2.0 * np.pi * audio_freq * t))

        # Add CTCSS Modulation if requested
        if ctcss_freq is not None and ctcss_freq > 0:
            phase += (ctcss_dev / ctcss_freq) * (-np.cos(2.0 * np.pi * ctcss_freq * t))

        # Generate complex exponential and apply amplitude + active mask
        chan_iq = amp * np.exp(1j * phase)
        chan_iq[~active_mask] = 0.0

        iq_total += chan_iq

    # Normalize total signal peak to 1.0 to prevent clipping
    peak = np.max(np.abs(iq_total))
    if peak > 0:
        iq_total = iq_total / peak

    # Add White Gaussian Noise if SNR is specified
    if snr_db is not None:
        sig_power = np.mean(np.abs(iq_total) ** 2)
        if sig_power > 0:
            noise_power = sig_power * (10.0 ** (-snr_db / 10.0))
        else:
            # No signal present: treat snr_db as noise level relative to full scale (0 dBFS)
            noise_power = 10.0 ** (-snr_db / 10.0)
        noise_std = np.sqrt(noise_power / 2.0)

        # Use seeded RNG for reproducibility
        rng = np.random.default_rng(seed)
        noise = noise_std * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))
        iq_total += noise

    return iq_total.astype(np.complex64)
