import numpy as np
from utilities import (
    DEFAULT_AUDIO_RATE,
    WAV_HEADER_BYTES,
    baseband_to_bin,
    bin_to_baseband,
    build_column_edges,
    format_active_banks,
    format_channel_banks,
    index_to_column,
    parse_bank_entry,
    wav_bytes_per_sec,
    wav_duration_sec,
)


def test_wav_bytes_per_sec_standard() -> None:
    """16-bit mono at DEFAULT_AUDIO_RATE should yield 16000 bytes/sec."""
    assert wav_bytes_per_sec(16) == DEFAULT_AUDIO_RATE * 2
    assert wav_bytes_per_sec(16) == 16_000


def test_wav_bytes_per_sec_8bit() -> None:
    """8-bit mono at DEFAULT_AUDIO_RATE should yield 8000 bytes/sec."""
    assert wav_bytes_per_sec(8) == DEFAULT_AUDIO_RATE * 1
    assert wav_bytes_per_sec(8) == 8_000


def test_wav_duration_sec_exact() -> None:
    """file_size = header + N seconds of 16-bit audio should return exactly N seconds."""
    bps = wav_bytes_per_sec(16)  # 16000
    for seconds in [1, 2, 10]:
        file_size = WAV_HEADER_BYTES + bps * seconds
        assert wav_duration_sec(file_size, 16) == seconds


def test_wav_duration_sec_empty_file() -> None:
    """A file containing only the WAV header (no audio data) must return 0.0."""
    assert wav_duration_sec(WAV_HEADER_BYTES, 16) == 0.0


def test_wav_duration_sec_smaller_than_header() -> None:
    """Files smaller than the header are clamped to 0.0, not negative."""
    assert wav_duration_sec(0, 16) == 0.0
    assert wav_duration_sec(WAV_HEADER_BYTES - 1, 16) == 0.0


def test_wav_duration_sec_fractional() -> None:
    """Fractional durations should be computed correctly."""
    bps = wav_bytes_per_sec(16)  # 16000
    half_second_size = WAV_HEADER_BYTES + bps // 2
    assert wav_duration_sec(half_second_size, 16) == 0.5


def test_baseband_to_bin_center_and_edges() -> None:
    # center of spectrum is bb=0
    assert baseband_to_bin(0.0, samp_rate=2_000_000.0, spectrum_len=2048) == 1024
    # left/right edges clip into range rather than overflow
    assert baseband_to_bin(-2_000_000.0, samp_rate=2_000_000.0, spectrum_len=2048) == 0
    assert baseband_to_bin(2_000_000.0, samp_rate=2_000_000.0, spectrum_len=2048) == 2047

def test_baseband_bin_roundtrip() -> None:
    samp_rate: float = 2_000_000.0
    L: int = 2048
    for idx in [0, 1, 512, 1024, 1500, 2047]:
        bb: float = bin_to_baseband(float(idx), samp_rate, L)
        assert abs(baseband_to_bin(bb, samp_rate, L) - idx) <= 1  # rounding tolerance

def test_bin_to_baseband_vectorized() -> None:
    # channel_estimate() returns a numpy array of fractional bin indices;
    # bin_to_baseband must handle that without a per-element Python loop
    idx: np.ndarray = np.array([0.0, 1024.0, 2047.0])
    bb: np.ndarray = bin_to_baseband(idx, samp_rate=2_000_000.0, spectrum_len=2048)
    assert isinstance(bb, np.ndarray)
    assert np.allclose(bb, [-1_000_000.0, 0.0, 999_023.4375])

def test_build_column_edges_covers_all_indices_no_overlap() -> None:
    for L, num_cols in [(2048, 120), (100, 77), (50, 80)]:  # last case: num_cols > L
        edges: list[int] = build_column_edges(L, num_cols)
        assert edges[0] == 0 and edges[-1] == L
        assert all(edges[i] <= edges[i + 1] for i in range(len(edges) - 1))

def test_index_to_column_matches_bar_column() -> None:
    # regression test for the marker/bar alignment guarantee
    L: int = 2048
    num_cols: int = 120
    edges: list[int] = build_column_edges(L, num_cols)
    for bin_idx in [0, 1, 500, 1024, 2000, 2047]:
        col: int = index_to_column(bin_idx, edges)
        assert edges[col] <= bin_idx < edges[col + 1] or col == num_cols - 1


def test_format_active_banks_sorted_comma_separated() -> None:
    assert format_active_banks({"NET_B", "NET_A"}) == "NET_A, NET_B"
    assert format_active_banks({"NET_A"}) == "NET_A"


def test_format_active_banks_empty_is_none() -> None:
    assert format_active_banks(set()) == "none"


def test_format_active_banks_with_bank_labels() -> None:
    labels = {"NET_A": "Net A", "NET_B": "Net B"}
    assert format_active_banks({"NET_A"}, labels) == "NET_A (Net A)"
    assert format_active_banks({"NET_B", "NET_A"}, labels) == "NET_A (Net A), NET_B (Net B)"
    # Unknown tags are left bare (sorted alphabetically)
    assert format_active_banks({"NET_A", "MISC"}, labels) == "MISC, NET_A (Net A)"
    # Empty set still renders "none" regardless of labels
    assert format_active_banks(set(), labels) == "none"


def test_format_channel_banks_empty_is_blank() -> None:
    assert format_channel_banks([], 20) == ""


def test_format_channel_banks_bracketed_join() -> None:
    assert format_channel_banks(["NET_A", "NET_B"], 20) == "[NET_A,NET_B]"


def test_format_channel_banks_truncated_to_max_len() -> None:
    assert format_channel_banks(["NET_A", "NET_B"], 6) == "[NET_A"
    assert len(format_channel_banks(["NET_A", "NET_B"], 6)) == 6


def test_parse_bank_entry_comma_and_space_separated() -> None:
    assert parse_bank_entry("NET_A, NET_B") == ["NET_A", "NET_B"]
    assert parse_bank_entry("NET_A NET_B") == ["NET_A", "NET_B"]
    assert parse_bank_entry("  NET_A ,  NET_B  ") == ["NET_A", "NET_B"]


def test_parse_bank_entry_empty_is_promiscuous() -> None:
    assert parse_bank_entry("") == []
    assert parse_bank_entry("   ") == []
    assert parse_bank_entry("none") == []
    assert parse_bank_entry("NONE") == []
