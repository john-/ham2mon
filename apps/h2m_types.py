"""
Created on Thu Feb 29 09:00:22 2024

@author: john
"""

from dataclasses import dataclass
from typing import Optional

@dataclass(kw_only=True)
class Channel:
    baseband: int
    frequency: float
    locked: bool
    active: bool
    priority: bool
    hanging: bool

@dataclass(kw_only=True)
class ChannelMessage:
    state: str
    frequency: float
    channel: int
    file: Optional[str] = None
    classification: Optional[str] = None
    note: Optional[str] = None
