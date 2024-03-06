"""
Created on Thu Feb 29 09:00:22 2024

@author: john
"""

from dataclasses import dataclass

@dataclass(kw_only=True)
class Channel:
    baseband: int
    frequency: float
    locked: bool
    active: bool
    priority: bool
    hanging: bool
