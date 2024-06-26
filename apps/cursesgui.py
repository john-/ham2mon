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
from h2m_types import Channel
from lockout_manager import RadioFreqRange, LockoutListRadioFreq

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

         # Clear previous contents, draw border, and title
        self.win.erase()
        self.win.border(0)
        self.win.attron(curses.color_pair(6))
        self.win.addnstr(0, int(self.dims[1]/2-6), "SPECTRUM", 8,
                         curses.color_pair(6) | curses.A_DIM | curses.A_BOLD)

        # Generate threshold line, clip to window, and convert to int
        pos_yt = (self.threshold_db - self.max_db) * scale
        pos_yt = np.clip(pos_yt, min_y, max_y-1)
        pos_yt = pos_yt.astype(int)

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


    def draw_channels(self, channels: list[Channel]):
        """Draws tuned channels list

        Args:
            rf_channels [string]: List of strings in MHz
        """

        # Clear previous contents, draw border, and title
        self.win.erase()
        self.win.border(0)
        self.win.attron(curses.color_pair(6))
        self.win.addnstr(0, int(self.dims[1]/4), "CHANNELS", 8,
                         curses.color_pair(6) | curses.A_DIM | curses.A_BOLD)

        # Limit the displayed channels to no more than two rows
        max_length = 2*(self.dims[0]-2)

        # Draw the tuned channels prefixed by index in list (demodulator index)
        # Use color if tuned channel is active during this scan_cycle
        subset = channels[:max_length]
        subset = [c for c in subset if c.active or c.hanging]  # needs to match Scanner.add_lockout
        for idx, channel in enumerate(subset):
            icon = 'P' if channel.priority else ''
            text = f'{idx:02d}: {channel.frequency:.3f}'
            if idx < self.dims[0]-2:
                # Display in first column
                # text color based on activity
                if channel.active:
                    self.win.addnstr(idx+1, 1, text, 12, curses.color_pair(2) | curses.A_BOLD)
                    self.win.addnstr(idx+1, 12, icon, 1, curses.color_pair(2))
                else:
                    self.win.addnstr(idx+1, 1, text, 12, curses.color_pair(6))
                    self.win.addnstr(idx+1, 12, icon, 1, curses.color_pair(6))
            else:
                # Display in second column
                self.win.addnstr(idx-self.dims[0]+3, 13, text, 11)
                if channel.active:
                    self.win.addnstr(idx-self.dims[0]+3, 13, text, 11, curses.color_pair(2) | curses.A_BOLD)
                    self.win.addnstr(idx-self.dims[0]+3, 24, icon, 1, curses.color_pair(2))
                else:
                    self.win.addnstr(idx-self.dims[0]+3, 13, text, 11, curses.color_pair(6))
                    self.win.addnstr(idx-self.dims[0]+3, 24, icon, 1, curses.color_pair(6))

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

    def draw_channels(self, lockout_channels: LockoutListRadioFreq, channels: list[Channel]):
        """Draws lockout channels list

        Args:
            rf_channels [string]: List of strings in MHz
        """
        # Clear previous contents, draw border, and title
        self.win.erase()
        self.win.border(0)
        self.win.attron(curses.color_pair(6))
        self.win.addnstr(0, int(self.dims[1]/2-3), "LOCKOUT", 7,
                        curses.color_pair(6) | curses.A_DIM | curses.A_BOLD)

        # Draw the lockout channels
        # Use color if lockout channel is in active channel list during this scan_cycle
        locked_channels = [c for c in channels if c.locked]
        for idx, lockout in enumerate(lockout_channels):
            # Don't draw past height of window
            if idx <= self.dims[0]-3:
                if not lockout.saved:
                    icon = 'U'
                else:
                    icon = None
                attr = curses.color_pair(6)
                if isinstance(lockout, RadioFreqRange):
                    text = f"{lockout.min:.3f}-{lockout.max:.3f}"
                    for channel in locked_channels:
                        if lockout.min <= channel.frequency <= lockout.max:
                            attr = curses.color_pair(5) | curses.A_BOLD
                else:  # handle this single frequency
                    text = f"{lockout.freq:.3f}"
                    for channel in locked_channels:
                        if lockout.freq == channel.frequency:
                                attr = curses.color_pair(5) | curses.A_BOLD
                self.win.addnstr(idx+1, 1, text, 20, attr)
                if icon:
                    self.win.addnstr(idx+1, len(text)+1, icon, 1, attr & ~curses.A_BOLD)
            else:
                pass

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
        lockout_file_name (PurePath): Name of file with channels to lockout
        priority_file_name (string): Name of file with channels for priority
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
        self.lockout_file_name: PurePath = None
        self.priority_file_name = ""
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

    def draw_rx(self) -> None:
        """Draws receiver parameters
        """

        # Clear previous contents, draw border, and title
        self.win.erase()
        self.win.border(0)
        self.win.attron(curses.color_pair(6))
        self.win.addnstr(0, int(self.dims[1]/2-4), "RECEIVER", 8,
                         curses.color_pair(6) | curses.A_DIM | curses.A_BOLD)

        # Draw the receiver info prefix fields
        index = 1
        text = "RF Freq (MHz)  : "
        self.win.addnstr(index, 1, text, 18, curses.color_pair(6))

        index = index+1
        text = "         From  : "
        self.win.addnstr(index, 1, text, 18, curses.color_pair(6))

        index = index+1
        text = "           To  : "
        self.win.addnstr(index, 1, text, 18, curses.color_pair(6))

        index3 = 1  # handle case where there are no gains (processing from file)
        for index2, gain in enumerate(self.gains, 2):
            text = "{} Gain (dB){} : ".format(gain["name"], (4-len(gain["name"]))*' ')
            self.win.addnstr(index+index2-1, 1, text, 18)
            index3 = index2

        index = index+index3
        text = "BB Rate (Msps) : "
        self.win.addnstr(index, 1, text, 18, curses.color_pair(6))

        index = index+1
        text = "BB Sql  (dB)   : "
        self.win.addnstr(index, 1, text, 18, curses.color_pair(6))

        index = 1

        # skip a line when range scanning to insert the status
        if len(self.steps) > 1:
            index = index + 1

        text = "AF Vol  (dB) : "
        self.win.addnstr(index, 29, text, 18, curses.color_pair(6))

        index = index+1
        text = "Record       : "
        self.win.addnstr(index, 29, text, 18, curses.color_pair(6))

        index = index+1
        text = "Demod Type   : "
        self.win.addnstr(index, 29, text, 18, curses.color_pair(6))

        index = index+1
        text = "Lockout File : "
        self.win.addnstr(index, 29, text, 18, curses.color_pair(6))

        index = index+1
        text = "Priority File: "
        self.win.addnstr(index, 29, text, 18, curses.color_pair(6))

        index = index+1
        text = "Log Type     : "
        self.win.addnstr(index, 29, text, 18, curses.color_pair(6))

        if self.channel_log_target is not None:
            index = index+1
            text = "Log Target   : "
            self.win.addnstr(index, 29, text, 18, curses.color_pair(6))

        # Draw the receiver info suffix fields
        index = 1
        if self.freq_entry != 'None':
            text = self.freq_entry
        else:
            text = '{:.3f}'.format((self.center_freq)/1E6)
        self.win.addnstr(index, 18, text, 8, curses.color_pair(5))

        index = index+1
        text = '{:.3f}'.format((self.center_freq - self.samp_rate/2)/1E6)
        self.win.addnstr(index, 18, text, 8, curses.color_pair(6))

        index = index+1
        text = '{:.3f}'.format((self.center_freq + self.samp_rate/2)/1E6)
        self.win.addnstr(index, 18, text, 8, curses.color_pair(6))

        index3 = 1  # handle case where there are no gains (processing from file)
        for index2, gain in enumerate(self.gains, 2):
            text = str(gain["value"])
            self.win.addnstr(index+index2-1, 18, text, 8, curses.color_pair(5))
            index3 = index2

        index = index+index3
        text = str(self.samp_rate/1E6)
        self.win.addnstr(index, 18, text, 8, curses.color_pair(5))

        index = index+1
        text = str(self.squelch_db)
        self.win.addnstr(index, 18, text, 8, curses.color_pair(5))

        index = 1

        # status line when range scanning
        num_steps: int = len(self.steps)
        if num_steps > 1:
            step = self.step + 1  # start at 1 instead of 0
            percent = int((step/num_steps)*100)
            text = f'-> Step {step} of {num_steps} ({percent}%)'
            self.win.addnstr(index, 26, text, 22, curses.color_pair(6))
            index = index + 1

        text = str(self.volume_db)
        self.win.addnstr(index, 44, text, 8, curses.color_pair(5))

        index = index+1
        text = ''
        for key in self.classifier_params.wanted.keys():
            if self.classifier_params.wanted[key]:
                text = text + key
        if text == '':
            text = str(self.record)
        self.win.addnstr(index, 44, text, 8)

        index = index+1
        text = self.demod_map[self.type_demod]
        self.win.addnstr(index, 44, text, 8, curses.color_pair(6))

        index = index+1
        text = str(self.lockout_file_name.name)
        self.win.addnstr(index, 44, text, 20, curses.color_pair(6))

        index = index+1
        text = str(self.priority_file_name)
        self.win.addnstr(index, 44, text, 20, curses.color_pair(6))

        index = index+1
        text = str(self.channel_log_type)
        self.win.addnstr(index, 44, text, 20, curses.color_pair(6))

        if self.channel_log_target is not None:
            index = index+1
            text = str(self.channel_log_target)
            self.win.addnstr(index, 44, text, 20, curses.color_pair(6))

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
