"""
Created on Thu Mar 7 2024

@author: john
"""
import bisect
import time
from typing import overload
import numpy as np
from numpy.typing import NDArray

DEFAULT_AUDIO_RATE: int = 8000
"""Sample rate (Hz) used when writing audio WAV files (mono PCM, decimated from SDR rate)."""

WAV_HEADER_BYTES: int = 44
"""Size of a standard PCM WAV file header in bytes."""


def wav_bytes_per_sec(bit_depth: int) -> int:
    """Return bytes per second for a mono PCM WAV file recorded at DEFAULT_AUDIO_RATE.

    Args:
        bit_depth: Sample bit depth (e.g. 16 for 16-bit PCM).

    Returns:
        Bytes per second of audio data (header not included).
    """
    return DEFAULT_AUDIO_RATE * (bit_depth // 8)


def wav_duration_sec(file_size: int, bit_depth: int) -> float:
    """Return duration in seconds for a mono PCM WAV file given its total size on disk.

    Subtracts the standard WAV_HEADER_BYTES (44) before dividing by bytes-per-second.
    Returns 0.0 if the file is smaller than or equal to the header (empty recording).

    Args:
        file_size: Total file size in bytes, as returned by os.stat().st_size.
        bit_depth: Sample bit depth (e.g. 16 for 16-bit PCM).

    Returns:
        Duration in seconds, clamped to >= 0.0.
    """
    bps = wav_bytes_per_sec(bit_depth)
    if bps <= 0:
        return 0.0
    return max(0.0, (file_size - WAV_HEADER_BYTES) / bps)


def format_freq_mhz(rf_mhz: float) -> str:
    """Format RF frequency in MHz to MHz string with 4 decimal places (e.g. 460.125 -> '460.1250')."""
    return f"{np.round(rf_mhz, 4):.4f}"


def format_timestamp(timestamp: float) -> str:
    """Format epoch timestamp float to 'YYYYMMDD_HHMMSS.sss' string."""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp)) + f"{timestamp % 1:.3f}"[1:]


def frequency_to_baseband(freq: float, center_freq: int, channel_spacing: int) -> int:
    """Returns baseband frequency in Hz
    """
    bb_freq = float(freq) * 1E6 - center_freq
    bb_freq = round(bb_freq/channel_spacing) * channel_spacing
    return bb_freq

def baseband_to_frequency(bb_freq: int, center_freq: int) -> float:
    """Return frequency in Mhz
    """
    return (bb_freq + center_freq)/1E6

def baseband_to_bin(bb: float, samp_rate: float, spectrum_len: int) -> int:
    """Convert a baseband offset (Hz) to an FFT bin index, clipped to valid range.

    bb=0 is the center bin; -samp_rate/2 is the left edge (bin 0);
    +samp_rate/2 is the right edge (bin spectrum_len-1).
    """
    if spectrum_len <= 0:
        raise ValueError("spectrum_len must be positive")
    bin_idx = int(round((bb * spectrum_len / samp_rate) + spectrum_len / 2))
    return max(0, min(spectrum_len - 1, bin_idx))

@overload
def bin_to_baseband(bin_idx: float, samp_rate: float, spectrum_len: int) -> float: ...

@overload
def bin_to_baseband(bin_idx: NDArray[np.float64], samp_rate: float, spectrum_len: int) -> NDArray[np.float64]: ...

def bin_to_baseband(
    bin_idx: float | NDArray[np.float64], samp_rate: float, spectrum_len: int
) -> float | NDArray[np.float64]:
    """Convert an FFT bin index (or array of indices, incl. fractional) to
    baseband offset in Hz. Inverse of baseband_to_bin (up to rounding).
    Accepts a scalar or a numpy array, matching channel_estimate()'s output.
    """
    return (bin_idx - spectrum_len / 2) * samp_rate / spectrum_len

def build_column_edges(length: int, num_cols: int) -> list[int]:
    """Partition `length` items into `num_cols` near-equal-width groups.
    Returns a list of num_cols+1 boundaries; group `col` covers
    data[edges[col]:edges[col+1]].
    """
    edges = [int(col * length / num_cols) for col in range(num_cols)]
    edges.append(length)
    return edges

def index_to_column(index: int, col_edges: list[int]) -> int:
    """Given an index into the original data and the edges from
    build_column_edges, return which column it falls in."""
    col = bisect.bisect_right(col_edges, index) - 1
    return max(0, min(len(col_edges) - 2, col))