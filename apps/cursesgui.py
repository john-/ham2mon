#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:16:22 2015

@author: madengr
"""
import locale
import curses
import time
import numpy as np
import logging
from pathlib import PurePath
from frequency_manager import ConfigFrequency, RadioFreqRange, ChannelFrequency, ChannelList, FrequencyList

locale.setlocale(locale.LC_ALL, '')
class SpectrumWindow(object):
    """Curses spectrum display window

    Args:
        screen (object): a curses screen object

    Attributes:
        max_db (int): Top of window in dB
        min_db (int): Bottom of window in dB
        threshold_db (int): Threshold horizontal line
    """
    def __init__(self, screen):
        self.screen = screen

        # Set default values
        self.max_db = 50
        self.min_db = -20
        self.threshold_db = 20

        # Create a window object in top half of the screen, within the border
        screen_dims = screen.getmaxyx()
        height = int(screen_dims[0]/2.0)
        width = screen_dims[1]-2
        self.win = curses.newwin(height, width, 1, 1)
        self.dims = self.win.getmaxyx()

        # Right end of window resreved for string of N charachters
        self.chars = 7

    def draw_spectrum(self, data):
        """Scales input spectral data to window dimensions and draws bar graph

        Args:
            data (numpy.ndarray): FFT power spectrum data in linear, not dB

        Test cases for data with min_db=-100 and max_db=0 on 80x24 terminal:
            1.0E-10 draws nothing since it is not above -100 dB
            1.1E-10 draws one row
            1.0E-05 draws 5 rows
            1.0E+00 draws 10 rows
            1.0E+01 draws 10 rows
        """

        # Keep min_db to 10 dB below max_db
        if self.min_db > (self.max_db - 10):
            self.min_db = self.max_db - 10

        # Split the data into N window bins
        # N is window width between border (i.e. self.dims[1]-2 )
        # Data must be at least as long as the window width or crash
        # Use the maximum value from each input data bin for the window bin
        win_bins = np.array_split(data, self.dims[1]-self.chars)
        win_bin_max = []
        for win_bin in win_bins:
            win_bin_max.append(np.max(win_bin))

        # Convert to dB
        win_bin_max_db = 10*np.log10(win_bin_max)

        # The plot windows starts from max_db at the top
        # and draws DOWNWARD to min_db (remember this is a curses window).
        # Thus linear scaling goes from min_y=1 at the top
        # and draws DOWNWARD to max_y=dims[0]-1 at the bottom
        # The "1" and "-1" is to account for the border at top and bottom
        min_y = 1
        max_y = self.dims[0]-1

        # Scaling factor for plot
        scale = (min_y-max_y)/(self.max_db-self.min_db)

        # Generate y position, clip to window, and convert to int
        pos_y = (win_bin_max_db - self.max_db) * scale
        pos_y = np.clip(pos_y, min_y, max_y)
        pos_y = pos_y.astype(int)

        # Generate threshold line, clip to window, and convert to int
        pos_yt = (self.threshold_db - self.max_db) * scale
        pos_yt = np.clip(pos_yt, min_y, max_y-1)
        pos_yt = pos_yt.astype(int)

         # Clear previous contents, draw border, and title
        self.win.erase()
        self.win.border(0)
        self.win.attron(curses.color_pair(6))
        self.win.addnstr(0, int(self.dims[1]/2-6), "SPECTRUM", 8,
                         curses.color_pair(6) | curses.A_DIM | curses.A_BOLD)

        # Draw the bars
        for pos_x in range(len(pos_y)):
            # Invert the y fill since we want bars
            # Offset x (column) by 1 so it does not start on the border
            if pos_y[pos_x] > pos_yt:
                # bar is below threshold, use low value color
                self.win.vline(pos_y[pos_x], pos_x+1, "-", max_y-pos_y[pos_x],curses.color_pair(3) | curses.A_BOLD)
            elif pos_y[pos_x] <= min_y:
                # bar is above max (clipped to min y), use max value color
                self.win.vline(pos_y[pos_x], pos_x+1, "+", max_y-pos_y[pos_x],curses.color_pair(1) | curses.A_BOLD)
            else:
                # bar is between max value and threshold, use threshold color
                self.win.vline(pos_y[pos_x], pos_x+1, "*", max_y-pos_y[pos_x],curses.color_pair(2) | curses.A_BOLD)

        # Draw the max_db and min_db strings
        string = ">" + "%+03d" % self.max_db
        self.win.addnstr(0, 1 + self.dims[1] - self.chars, string, self.chars,
                         curses.color_pair(1))
        string = ">" + "%+03d" % self.min_db
        self.win.addnstr(max_y, 1 + self.dims[1] - self.chars, string,
                         self.chars, curses.color_pair(3))

        # Draw the theshold line
        # x=1 start to account for left border
        self.win.hline(pos_yt, 1, "-", len(pos_y), curses.color_pair(2))

        # Draw the theshold string
        string = ">" + "%+03d" % self.threshold_db
        self.win.addnstr(pos_yt, (1 + self.dims[1] - self.chars), string,
                         self.chars, curses.color_pair(2))

        # Hide cursor
        self.win.leaveok(1)

        # Update virtual window
        self.win.noutrefresh()

    def proc_keyb(self, keyb: int):
        """Process keystrokes

        Args:
            keyb (int): keystroke in ASCII

        Returns:
            bool: True if receiver needs tuning, False if not

        """
        if  keyb == ord('t'):
            self.threshold_db += 5
            return True
        elif keyb == ord('r'):
            self.threshold_db -= 5
            return True
        elif keyb == ord('T'):
            self.threshold_db += 1
            return True
        elif keyb == ord('R'):
            self.threshold_db -= 1
            return True
        elif keyb == ord('p'):
            self.max_db += 5
        elif keyb == ord('o'):
            self.max_db -= 5
        elif keyb == ord('w'):
            self.min_db += 5
        elif keyb == ord('q'):
            self.min_db -= 5
        else:
            pass
        return False


class ChannelWindow(object):
    """Curses channel display window

    Args:
        screen (object): a curses screen object
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, screen):
        self.screen = screen

        # Create a window object in the bottom half of the screen
        # Make it about 1/4 the screen width
        # Place on left side and to the right of the border
        screen_dims = screen.getmaxyx()
        spectrum_height = int(screen_dims[0]/2.0)
        height = screen_dims[0] - spectrum_height - 2
        width = int(screen_dims[1]/4.0)
        self.win = curses.newwin(height, width, screen_dims[0] - height - 1, 1)
        self.dims = self.win.getmaxyx()

        self.entries: list[ChannelWindow.ChannelEntry] = []

    class ChannelEntry(object):
        def __init__(self):
            self.width = ChannelWindow.ChannelEntry.width
            self.prev_channel: ChannelFrequency | None = None

            ChannelWindow.ChannelEntry.rows += 1
            self.row = ChannelWindow.ChannelEntry.rows

            self.attrs = { 'bold_freq': curses.color_pair(2) | curses.A_BOLD,
                           'bold_icon': curses.color_pair(2),
                           'normal_freq': curses.color_pair(6),
                           'normal_icon': curses.color_pair(6) }

        @classmethod
        def set_window(cls, win: any, width: int):
            ChannelWindow.ChannelEntry.win = win
            ChannelWindow.ChannelEntry.rows = 0  # used to track the next index for each row
            ChannelWindow.ChannelEntry.width = width

        def set(self, channel: ChannelFrequency) -> None:
            self.channel = channel

            # draw if changing
            if self.prev_channel != self.channel:
                self.draw()

            self.prev_channel = self.channel

        def draw(self) -> None:

            row = self.row
            channel = self.channel
            win = ChannelWindow.ChannelEntry.win

            if channel is None:
                text = ' ' * self.width
                win.addnstr(row, 1, text, self.width)
                return

            text = f'{self.row-1:02d}: {channel.rf:.3f}'
            icon = 'P' if channel.priority else ' '
            label = channel.label or ' ' * self.width

            attributes = (self.attrs['bold_freq'], self.attrs['bold_icon']) if channel.active else (
                self.attrs['normal_freq'], self.attrs['normal_icon'])

            win.addnstr(row, 1, text, self.width, attributes[0])
            win.addnstr(row, 12, icon, 1, attributes[1])
            win.addnstr(row, 14, label, self.width-14, attributes[0])

    def draw_frame(self) -> None:
        # Clear previous contents, draw border, and title
        self.win.erase()
        self.win.border(0)
        self.win.attron(curses.color_pair(6))
        self.win.addnstr(0, int(self.dims[1]/4), "CHANNELS", 8,
                         curses.color_pair(6) | curses.A_DIM | curses.A_BOLD)

        ChannelWindow.ChannelEntry.set_window(self.win, self.dims[1]-2)

        # Limit the displayed channels to one column
        max_length = self.dims[0]-2

        # Add entries to the list if not already present
        cur_length = len(self.entries)
        if cur_length < max_length:
            for _ in range(cur_length, max_length):
                self.entries.append(ChannelWindow.ChannelEntry())

        # limit the size to available space (the window got smaller)
        self.entries = self.entries[:max_length]

    def draw_channels(self, channels: list[ChannelFrequency]):
        """Draws tuned channels list

        Args:
            rf_channels [string]: List of strings in MHz
        """
        # Draw the tuned channels prefixed by index in list (demodulator index)
        # Use color if tuned channel is active during this scan_cycle
        subset = [c for c in channels if c.active or c.hanging]  # needs to match Scanner.add_lockout

        for idx, entry in enumerate(self.entries):
            if idx >= len(subset):
                entry.set(None)
            else:
                entry.set(subset[idx])

        # Hide cursor
        self.win.leaveok(1)

        # Update virtual window
        self.win.noutrefresh()


class LockoutWindow(object):
    """Curses lockout channel display window

    Args:
        screen (object): a curses screen object
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, screen):
        self.screen = screen

        # Create a window object in the bottom half of the screen
        # Make it about 1/4 the screen width
        # Place on left side and to the right of the border
        screen_dims = screen.getmaxyx()
        spectrum_height = int(screen_dims[0]/2.0)
        height = screen_dims[0] - spectrum_height - 2
        width = int(screen_dims[1]/4.0)
        self.win = curses.newwin(height, width, screen_dims[0] - height - 1, width+1)
        self.dims = self.win.getmaxyx()

        self.lockouts: list[LockoutWindow.LockoutEntry] = []

    class LockoutEntry(object):

        def __init__(self):
            self.width = LockoutWindow.LockoutEntry.width
            self.prev_lockout: ConfigFrequency | None = None
            self.prev_has_activity: bool | None = None

            LockoutWindow.LockoutEntry.rows += 1
            self.row = LockoutWindow.LockoutEntry.rows

            self.attrs = { 'bold_lockout': curses.color_pair(5) | curses.A_BOLD,
                           'normal_lockout': curses.color_pair(6) }

        @classmethod
        def set_window(cls, win: any, width: int):
            LockoutWindow.LockoutEntry.win = win
            LockoutWindow.LockoutEntry.rows = 0  # used to track the next index for each row
            LockoutWindow.LockoutEntry.width = width

        def set(self, lockout: ConfigFrequency, has_activity: bool) -> None:
            self.lockout = lockout
            self.has_activity = has_activity

            # draw if changing
            if self.prev_lockout != self.lockout or self.prev_has_activity != self.has_activity:
                self.draw()

            self.prev_lockout = self.lockout
            self.prev_has_activity = self.has_activity

        def draw(self) -> None:

            row = self.row
            lockout = self.lockout
            has_activity = self.has_activity
            win = LockoutWindow.LockoutEntry.win

            if lockout is None:
                text = ' ' * self.width
                win.addnstr(row, 1, text, self.width)
                return

            if not lockout.saved:
                icon = 'U'
            else:
                icon = None

            attr = self.attrs['normal_lockout']

            if isinstance(lockout.rf, RadioFreqRange):
                text = f"{lockout.rf.lo:.3f}-{lockout.rf.hi:.3f}"
                if has_activity:
                    attr = self.attrs['bold_lockout']
            else:  # handle this single frequency
                text = f"{lockout.rf:.3f}"
                if has_activity:
                    attr = self.attrs['bold_lockout']

            win.addnstr(row, 1, text, self.width, attr)
            if icon:
                win.addnstr(row, len(text)+1, icon, 1, attr & ~curses.A_BOLD)

    def draw_frame(self) -> None:
        # Clear previous contents, draw border, and title
        self.win.erase()
        self.win.border(0)
        self.win.attron(curses.color_pair(6))
        self.win.addnstr(0, int(self.dims[1]/2-3), "LOCKOUT", 7,
                        curses.color_pair(6) | curses.A_DIM | curses.A_BOLD)

        LockoutWindow.LockoutEntry.set_window(self.win, self.dims[1]-2)

        # Limit the displayed channels to one column
        max_length = self.dims[0]-2

        # Add entries to the list if not already present
        cur_length = len(self.lockouts)
        if cur_length < max_length:
            for _ in range(cur_length, max_length):
                self.lockouts.append(LockoutWindow.LockoutEntry())

        # limit the size to available space (the window got smaller)
        self.lockouts = self.lockouts[:max_length]

    def lockout_has_activity(self, lockout: ConfigFrequency) -> bool:
        """Checks if lockout has activity

        Args:
            lockout (RadioFreq): lockout to check
        """
        has_activity = False
        for channel in self.locked_channels:
            if isinstance(lockout.rf, RadioFreqRange):
                if lockout.rf.lo <= channel.rf <= lockout.rf.hi:
                    has_activity = True
            else:  # handle this single frequency
                if lockout.rf == channel.rf:
                    has_activity = True

        return has_activity

    def draw_channels(self, frequencies: FrequencyList, channels: ChannelList):
        """Draws lockout channels list

        Args:
            rf_channels [string]: List of strings in MHz
        """
        # Draw the lockout channels
        # Use color if lockout channel is in active channel list during this scan_cycle
        self.locked_channels = [c for c in channels if c.locked]

        # Extract the frequencues/ranges the user wants locked out
        locked_frequencies = [freq for freq in frequencies if freq.locked]

        # populate the gui list with the locked frequencies configured by the user
        for idx, lockout in enumerate(self.lockouts):
            if idx >= len(locked_frequencies):
                lockout.set(None, None)
            else:
                lockout_freq = locked_frequencies[idx]
                has_activity = self.lockout_has_activity(lockout_freq)
                lockout.set(lockout_freq, has_activity)

        # Hide cursor
        self.win.leaveok(1)

        # Update virtual window
        self.win.noutrefresh()

    def proc_keyb_set_lockout(self, keyb: int):
        """Process keystrokes to lock out channels 0 - 9

        Args:
            keyb (int): keystroke in ASCII

        Returns:
            bool: True if scanner needs adjusting, False if not
        """
        # pylint: disable=no-self-use

        # Check if keys 0 - 9 pressed
        if keyb - 48 in range(10):
            return True
        else:
            return False

    def proc_keyb_clear_lockout(self, keyb: int):
        """Process keystrokes to clear lockout with "l"

        Args:
            keyb (int): keystroke in ASCII

        Returns:
            bool: True if scanner needs adjusting, False if not
        """
        # pylint: disable=no-self-use

        # Check if 'l' is pressed
        if keyb == ord('l'):
            return True
        else:
            return False


class RxWindow(object):
    """Curses receiver paramater window

    Args:
    screen (object): a curses screen object

    Attributes:
        center_freq (float): Hardware RF center frequency in Hz
        samp_rate (float): Hardware sample rate in sps (1E6 min)
        gains (list): Hardware gains in dB
        squelch_db (int): Squelch in dB
        volume_dB (int): Volume in dB
        type_demod (int): Type of demodulation (0 = FM, 1 = AM)
        record (bool): Record audio to file if True
        frequency_file_name (PurePath): Name of file with frequencies
        channel_log_file_name (string): Name of file for channel activity logging
        channel_log_timeout (int): Timeout delay between logging active state of channel in seconds
        log_mode (string): Log system mode (file, database type)
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, screen):
        self.screen = screen

        # Set default values
        self.center_freq = 146E6
        self.step = None
        self.steps: list[int] = []
        self.freq_max = 148E6
        self.samp_rate = 2E6
        self.freq_entry = 'None'
        self.squelch_db = -60
        self.volume_db = 0
        self.type_demod = 0
        self.record = True
        self.frequency_file_name: PurePath = None
        self.channel_log_type = ""
        self.channel_log_target = ""
        self.gains = None
        self.classifier_params = None

        self.demod_map = {
            0: 'NBFM',
            1: 'AM',
            2: 'WBFM',
        }

        # Create a window object in the bottom half of the screen
        # Make it about 1/2 the screen width
        # Place on right side and to the left of the border
        screen_dims = screen.getmaxyx()
        spectrum_height = int(screen_dims[0]/2.0)
        height = screen_dims[0] - spectrum_height - 2
        # subtract the channel and lockout widths
        width = screen_dims[1] - 2 * int(screen_dims[1]/4.0) - 2
        self.win = curses.newwin(height, width, screen_dims[0] - height - 1,
                                 int(screen_dims[1]-width-1))
        self.dims = self.win.getmaxyx()

    class RxEntry(object):

        label_width = 14
        value_width = 10
        #rows = [0, 0]  # used to track the next index for each row

        def __init__(self, label: str | None, column: int,
                     justification: str, can_modify: bool):
            self.label = label
            RxWindow.RxEntry.rows[column-1] += 1
            self.row = RxWindow.RxEntry.rows[column-1]
            self.column = column
            self.justification = justification
            self.can_modify = can_modify
            self.prev_value: str | None = None

            win = RxWindow.RxEntry.win
            label_width = RxWindow.RxEntry.label_width
            value_width = RxWindow.RxEntry.value_width

            self.attrs = { 'bold': curses.color_pair(5),
                           'normal': curses.color_pair(6) }

            if self.label is None:
                return

            if justification == 'left':
                text = f'{self.label:<{label_width}}:'
            else:
                text = f'{self.label:>{label_width-1}} :'

            start = (self.column - 1) * ( label_width + value_width + 3 ) + 1
            win.addnstr(self.row, start, text, len(text)+1, self.attrs['normal'])

        @classmethod
        def set_window(cls, win: any):
            RxWindow.RxEntry.win = win
            RxWindow.RxEntry.rows = [0, 0]  # used to track the next index for each row

        def set(self, value: str ) -> None:
            self.value = value

            # draw if changing
            if self.prev_value != self.value:
                self.draw()

            self.prev_value = self.value

        def draw(self) -> None:

            row = self.row
            value = self.value

            label_width = RxWindow.RxEntry.label_width
            value_width = RxWindow.RxEntry.value_width
            win = RxWindow.RxEntry.win


            if value is None:
                text = ' ' * RxWindow.RxEntry.value_width
                win.addnstr(row, 1, text, value_width)
                return

            attr = self.attrs['bold'] if self.can_modify else self.attrs['normal']

            # -2 is a hack for getting the range counter shifted to the left
            label_offset = -2 if self.label is None else label_width + 3

            start = (self.column - 1) * ( label_width + value_width + 3 ) + label_offset

            win.addnstr(row, start, value, len(value), attr)

    def draw_frame(self) -> None:
        # Clear previous contents, draw border, and title
        self.win.erase()
        self.win.border(0)
        self.win.attron(curses.color_pair(6))
        self.win.addnstr(0, int(self.dims[1]/2-4), "RECEIVER", 8,
                         curses.color_pair(6) | curses.A_DIM | curses.A_BOLD)

        RxWindow.RxEntry.set_window(self.win)

        self.rf_freq_field = RxWindow.RxEntry(
            "RF Freq (MHz)", 1, 'left', True)
        self.from_freq_field = RxWindow.RxEntry(
            "From", 1, 'right', False)
        self.to_freq_field = RxWindow.RxEntry(
            "To", 1, 'right', False)

        self.gain_entries_field: list[RxWindow.RxEntry] = []
        for gain in self.gains:
            text = f'{gain["name"]} Gain (dB)'
            self.gain_entries_field.append(RxWindow.RxEntry(
                text, 1, 'left', True))

        self.bb_rate_field = RxWindow.RxEntry(
            "BB Rate (Msps)", 1, 'left', False)

        self.squelch_db_field = RxWindow.RxEntry(
            "BB Sql  (dB)", 1, 'left', True)

        # status line when range scanning
        num_steps: int = len(self.steps)
        if num_steps > 1:
            self.step_status_field = RxWindow.RxEntry(
                None, 2, 'left', False)

        self.volume_db_field = RxWindow.RxEntry(
            "AF Vol  (dB)", 2, 'left', True)

        self.record_field = RxWindow.RxEntry(
            "Record", 2, 'left', False)

        self.type_demod_field = RxWindow.RxEntry(
            "Demod Type", 2, 'left', False)

        self.frequency_file_name_field = RxWindow.RxEntry(
            "Freq File", 2, 'left', False)

        self.channel_log_type_field = RxWindow.RxEntry(
            "Log Type", 2, 'left', False)

        if self.channel_log_target is not None:
            self.channel_log_target_field = RxWindow.RxEntry(
                "Log Target", 2, 'left', False)

    def draw_rx(self) -> None:
        """Draws receiver parameters
        """

        if self.freq_entry != 'None':
            freq = self.freq_entry
        else:
            freq = f'{self.center_freq/1E6:0.3f}'
        self.rf_freq_field.set(freq)

        self.from_freq_field.set(f'{(self.center_freq - self.samp_rate/2)/1E6:0.3f}')
        self.to_freq_field.set(f'{(self.center_freq + self.samp_rate/2)/1E6:0.3f}')

        for index, gain in enumerate(self.gains):
            self.gain_entries_field[index].set(f'{gain["value"]:0.1f}')

        self.bb_rate_field.set(f'{self.samp_rate/1E6:0.1f}')
        self.squelch_db_field.set(f'{self.squelch_db:d}')

        # status line when range scanning
        num_steps: int = len(self.steps)
        if num_steps > 1:
            step = self.step + 1  # start at 1 instead of 0
            percent = int((step/num_steps)*100)
            text = f'-> Step {step} of {num_steps} ({percent}%)  '
            self.step_status_field.set(text)

        self.volume_db_field.set(f'{self.volume_db:d} ')

        text = ''
        for key in self.classifier_params.wanted.keys():
            if self.classifier_params.wanted[key]:
                text = text + key
        if text == '':
            text = str(self.record)
        self.record_field.set(text)

        text = self.demod_map[self.type_demod]
        self.type_demod_field.set(text)

        self.frequency_file_name_field.set(self.frequency_file_name.name)

        self.channel_log_type_field.set(self.channel_log_type)

        if self.channel_log_target is not None:
            self.channel_log_target_field.set(self.channel_log_target)

        # Hide cursor
        self.win.leaveok(1)

        # Update virtual window
        self.win.noutrefresh()

    def proc_keyb_hard(self, keyb: int):
        """Process keystrokes to adjust hard receiver settings

        Tune center_freq in 100 MHz steps with 'x' and 'c'
        Tune center_freq in 10 MHz steps with 'v' and 'c'
        Tune center_freq in 1 MHz steps with 'm' and 'n'
        Tune center_freq in 100 kHz steps with 'k' and 'j'

        Args:
            keyb (int): keystroke in ASCII

        Returns:
            bool: True if receiver needs adjusting, False if not
        """
        # pylint: disable=too-many-return-statements
        # pylint: disable=too-many-branches

        # Tune self.center_freq in 100 MHz steps with 'x' and 'c'
        if keyb == ord('x'):
            self.center_freq += 1E8
            return True
        elif keyb == ord('z'):
            self.center_freq -= 1E8
            return True
        # Tune self.center_freq in 10 MHz steps with 'v' and 'c'
        elif keyb == ord('v'):
            self.center_freq += 1E7
            return True
        elif keyb == ord('c'):
            self.center_freq -= 1E7
            return True
        # Tune self.center_freq in 1 MHz steps with 'm' and 'n'
        elif  keyb == ord('m'):
            self.center_freq += 1E6
            return True
        elif keyb == ord('n'):
            self.center_freq -= 1E6
            return True
        # Tune self.center_freq in 100 kHz steps with 'k' and 'j'
        elif keyb == ord('k'):
            self.center_freq += 1E5
            return True
        elif keyb == ord('j'):
            self.center_freq -= 1E5
            return True
        elif keyb == ord('/'):
            # set mode to frequency entry
            self.freq_entry = ''
            return False
        elif keyb == 27:  # ESC
            # end frequncy entry mode without seting the frequency
            self.freq_entry = 'None'
            return False
        elif keyb == ord('\n'):
            # set the frequency from what was entered
            try:
                self.center_freq = float(self.freq_entry) * 1E6
            except:
                pass
            self.freq_entry = 'None'
            return True
        elif self.freq_entry != 'None' and (keyb - 48 in range (10) or keyb == ord('.')):
            # build up frequency from 1-9 and '.'
            self.freq_entry = self.freq_entry + chr(keyb)
            return False
        elif keyb == curses.KEY_BACKSPACE:
            self.freq_entry = self.freq_entry[:-1]
            return False
        else:
            return False

    def proc_keyb_soft(self, keyb: int):
        """Process keystrokes to adjust soft receiver settings

        Tune gain_db in 10 dB steps with 'g' and 'f'
        Tune squelch_db in 1 dB steps with 's' and 'a'
        Tune volume_db in 1 dB steps with '.' and ','

        Args:
            keyb (int): keystroke in ASCII

        Returns:
            bool: True if receiver needs tuning, False if not
        """
        # pylint: disable=too-many-return-statements
        # pylint: disable=too-many-branches

        # Tune 1st gain element in 10 dB steps with 'g' and 'f'
        if keyb == ord('g'):
            self.gains[0]["value"] += 10
            return True
        elif keyb == ord('f'):
            self.gains[0]["value"] -= 10
            return True

        # Tune 1st gain element in 1 dB steps with 'G' and 'F'
        if keyb == ord('G'):
            self.gains[0]["value"] += 1
            return True
        elif keyb == ord('F'):
            self.gains[0]["value"] -= 1
            return True

        # Tune 2nd gain element in 10 dB steps with 'u' and 'y'
        if keyb == ord('u'):
            self.gains[1]["value"] += 10
            return True
        elif keyb == ord('y'):
            self.gains[1]["value"] -= 10
            return True

        # Tune 2nd gain element in 1 dB steps with 'U' and 'Y'
        if keyb == ord('U'):
            self.gains[1]["value"] += 1
            return True
        elif keyb == ord('Y'):
            self.gains[1]["value"] -= 1
            return True

        # Tune 3rd gain element in 10 dB steps with ']' and '['
        if keyb == ord(']'):
            self.gains[2]["value"] += 10
            return True
        elif keyb == ord('['):
            self.gains[2]["value"] -= 10
            return True

        # Tune 3rd gain element in 1 dB steps with '}' and '{'
        if keyb == ord('}'):
            self.gains[2]["value"] += 1
            return True
        elif keyb == ord('{'):
            self.gains[2]["value"] -= 1
            return True

        # Tune self.squelch_db in 1 dB steps with 's' and 'a'
        elif keyb == ord('s'):
            self.squelch_db += 1
            return True
        elif keyb == ord('a'):
            self.squelch_db -= 1
            return True
        # Tune self.volume_db in 1 dB steps with '.' and ','
        elif keyb == ord('.'):
            self.volume_db += 1
            return True
        elif keyb == ord(','):
            self.volume_db -= 1
            return True
        else:# pylint: disable=too-many-return-statements
            return False

def setup_screen(screen):
    """Sets up screen
    """

    # hide cursor
    curses.curs_set(0)

    # do not echo keystrokes
    curses.noecho()

    # break on ctrl-c
    curses.cbreak()

    # Define some colors
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(7, curses.COLOR_BLUE, curses.COLOR_BLACK)

    # Add border
    screen.border(0)

def main():
    """Test most of the GUI (except lockout processing)

    Initialize and set up screen
    Create windows
    Generate dummy spectrum data
    Update windows with dummy values
    Process keyboard strokes
    """
    # Use the curses.wrapper() to crash cleanly
    # Note the screen object is passed from the wrapper()
    # http://stackoverflow.com/questions/9854511/ppos_ython-curses-dilemma
    # The issue is the debuuger won't work with the wrapper()
    # So enable the next 2 lines and don't pass screen to main()
    screen = curses.initscr()
    curses.start_color()

    # Setup the screen
    setup_screen(screen)

    # Create windows
    specwin = SpectrumWindow(screen)
    chanwin = ChannelWindow(screen)
    lockoutwin = LockoutWindow(screen)
    rxwin = RxWindow(screen)

    while 1:
        # Generate some random spectrum data from -dyanmic_range to 0 dB
        #   then offset_db
        length = 128
        dynamic_range_db = 100
        offset_db = 50
        data = np.power(10, (-dynamic_range_db*np.random.rand(length)/10)\
            + offset_db/10)
        #data = 1E-5*np.ones(length)
        specwin.draw_spectrum(data)

        # Put some dummy values in the channel, lockout, and receiver windows
        chanwin.draw_channels(['144.100', '142.40', '145.00', '144.10',\
        '142.40', '145.00', '144.10', '142.40', '145.00', '144.10', '142.40',\
        '145.00', '142.40', '145.00', '144.10', '142.400', '145.00', '145.00'])
        lockoutwin.draw_channels(['144.100', '142.40', '145.00'])
        rxwin.draw_rx()

        # Update physical screen
        curses.doupdate()

        # Check for keystrokes and process
        keyb = screen.getch()
        specwin.proc_keyb(keyb)
        rxwin.proc_keyb_hard(keyb)
        rxwin.proc_keyb_soft(keyb)

        if keyb == ord('Q'):
            break

        # Sleep to get about a 10 Hz refresh
        time.sleep(0.1)

if __name__ == '__main__':
    try:
        #curses.wrapper(main)
        main()
    except KeyboardInterrupt:
        pass
