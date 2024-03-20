"""
Created on Thu Mar 7 2024

@author: john
"""
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