from typing import Callable
import numpy as np
from scanner import Scanner

def test_scanner_get_signal_strength_offset() -> None:
    class DummyScanner:
        samp_rate: int
        spectrum: np.ndarray
        _get_signal_strength: Callable[[int], float]

    scanner = DummyScanner()
    scanner.samp_rate = 4000000
    # Bind the real _get_signal_strength method to our dummy instance
    scanner._get_signal_strength = Scanner._get_signal_strength.__get__(
        scanner, DummyScanner
    )

    # 1. Test weak signal: linear power of 10.0 (10 dB raw power)
    # Calibrated: 10 dB - 70 dB = -60 dB
    scanner.spectrum = np.array([10.0])
    val = scanner._get_signal_strength(0)
    assert np.isclose(val, -60.0)

    # 2. Test strong signal: linear power of 1000.0 (30 dB raw power)
    # Calibrated: 30 dB - 70 dB = -40 dB
    scanner.spectrum = np.array([1000.0])
    val = scanner._get_signal_strength(0)
    assert np.isclose(val, -40.0)

    # 3. Test empty/None spectrum: should return -100.0
    scanner.spectrum = np.empty(0)
    val = scanner._get_signal_strength(0)
    assert np.isclose(val, -100.0)

    # 4. Test zero/negative power: should return -200.0
    scanner.spectrum = np.array([0.0])
    val = scanner._get_signal_strength(0)
    assert np.isclose(val, -200.0)
