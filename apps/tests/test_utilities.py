import numpy as np
from utilities import baseband_to_bin, bin_to_baseband, build_column_edges, index_to_column

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
