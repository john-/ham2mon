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
from pathlib import Path, PurePath
from frequency_manager import ConfigFrequency, ChannelFrequency, ChannelList, FrequencyList
from utilities import baseband_to_bin, build_column_edges, index_to_column
from ui_theme import THEME, ThemeConfiguration

logger = logging.getLogger(f"ham2mon.{__name__}")

locale.setlocale(locale.LC_ALL, '')


def compute_panel_widths(total_width, min_widths, weights):
    """Splits total_width among named panels, giving each its minimum first
    and distributing any remaining width by weight.

    This lets panels with open-ended content (e.g. user-supplied labels)
    claim more of the extra space as the terminal widens, while a panel
    with fixed-format content (e.g. columns of numbers) stops growing once
    it has enough room, instead of all panels growing at the same fixed
    proportion of the terminal width.

    Args:
        total_width (int): total width available to split among panels
        min_widths (dict): name -> minimum width for that panel
        weights (dict): name -> relative share of leftover width; a panel
            with weight 2 grows twice as fast as one with weight 1 once
            all minimums are satisfied

    Returns:
        dict: name -> allocated width, keys matching min_widths, values
            summing to total_width (or to sum(min_widths) if total_width
            is too small to satisfy every minimum)
    """
    names = list(min_widths.keys())
    base_sum = sum(min_widths.values())

    if total_width < base_sum:
        # Not enough room to satisfy every minimum; fall back to splitting
        # total_width proportionally by minimum size, floored at 1 each.
        widths = {}
        allocated = 0
        for i, name in enumerate(names):
            if i == len(names) - 1:
                widths[name] = max(1, total_width - allocated)
            else:
                share = max(1, int(total_width * min_widths[name] / base_sum))
                widths[name] = share
                allocated += share
        return widths

    extra = total_width - base_sum
    total_weight = sum(weights[name] for name in names) or 1
    widths = {}
    allocated_extra = 0
    for i, name in enumerate(names):
        if i == len(names) - 1:
            # Last panel absorbs any leftover from integer rounding.
            share = extra - allocated_extra
        else:
            share = int(extra * weights[name] / total_weight)
            allocated_extra += share
        widths[name] = min_widths[name] + share
    return widths


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
        self.samp_rate = 0

        # Create a window object in top half of the screen, within the border
        screen_dims = screen.getmaxyx()
        height = int(screen_dims[0]/2.0)
        width = screen_dims[1]-2
        self.outer_win = curses.newwin(height, width, 1, 1)
        self.win = self.outer_win.derwin(height - 2, width - 2, 1, 1)
        self.dims = self.win.getmaxyx()

        # Right end of window reserved for string of N characters
        self.chars = 5

    def cleanup(self) -> None:
        if hasattr(self, 'win') and self.win:
            try:
                self.win.erase()
                del self.win
            except Exception:
                pass
        if hasattr(self, 'outer_win') and self.outer_win:
            try:
                self.outer_win.erase()
                del self.outer_win
            except Exception:
                pass

    def draw_spectrum(self, data, channels=None, row_map=None):
        """Scales input spectral data to window dimensions and draws bar graph

        Args:
            data (numpy.ndarray): FFT power spectrum data in linear, not dB
            channels (list[ChannelFrequency] | None): Full channel list from
                the scanner.  When provided a self.samp_rate, lockout index
                label is drawn at the column matching each tuned (active or hanging)
                channel's baseband frequency.

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
        # N is window width of the inner window.
        # col_edges[col] is the first data index belonging to column `col`;
        # col_edges[num_cols] is a trailing sentinel equal to L.  This is the
        # single source of truth for the data-index <-> column mapping and is
        # reused below to place channel markers, so the two can never drift
        # apart the way separately-derived formulas could.
        num_cols = self.dims[1] - self.chars
        L = len(data)
        col_edges = build_column_edges(L, num_cols)
        win_bin_max = []
        for col in range(num_cols):
            start_idx = col_edges[col]
            end_idx = col_edges[col + 1]
            if start_idx < end_idx:
                win_bin_max.append(np.max(data[start_idx:end_idx]))
            else:
                win_bin_max.append(data[min(start_idx, L - 1)])

        # Convert to dB
        win_bin_max_db = 10*np.log10(win_bin_max)

        # Since self.win is the inner window, it has no borders.
        min_y = 0
        max_y = self.dims[0]-1

        # Scaling factor for plot
        scale = (min_y-max_y)/(self.max_db-self.min_db)

        # Generate y position, clip to window, and convert to int
        # Use nan_to_num to prevent OverflowError when log10(0) yields -inf/nan
        pos_y = (win_bin_max_db - self.max_db) * scale
        pos_y = np.nan_to_num(pos_y, nan=max_y, posinf=max_y, neginf=max_y)
        pos_y = np.clip(pos_y, min_y, max_y)
        pos_y = pos_y.astype(int)

        # Generate threshold line, clip to window, and convert to int
        pos_yt = (self.threshold_db - self.max_db) * scale
        pos_yt = np.clip(pos_yt, min_y, max_y)
        pos_yt = np.round(pos_yt).astype(int)

        # Clear outer window, draw border and title on the outer frame
        self.outer_win.erase()
        self.outer_win.attron(THEME.get('spectrum.border'))
        self.outer_win.border(0)
        outer_width = self.dims[1] + 2
        self.outer_win.addnstr(0, (outer_width - 8) // 2, "SPECTRUM", 8,
                               THEME.get('spectrum.title'))
        self.outer_win.leaveok(1)
        self.outer_win.noutrefresh()

        # Clear the inner drawing area
        self.win.erase()

        # Draw the bars
        for pos_x in range(len(pos_y)):
            # Invert the y fill since we want bars
            # Since we have no borders on inner window, we write exactly at pos_x
            if pos_y[pos_x] > pos_yt:
                # bar is below threshold, use low value color
                self.win.vline(pos_y[pos_x], pos_x, "-", max_y-pos_y[pos_x]+1, THEME.get('spectrum.bar_below_threshold'))
            elif pos_y[pos_x] <= min_y:
                # bar is above max (clipped to min y), use max value color
                self.win.vline(pos_y[pos_x], pos_x, "+", max_y-pos_y[pos_x]+1, THEME.get('spectrum.bar_above_max'))
            else:
                # bar is between max value and threshold, use threshold color
                self.win.vline(pos_y[pos_x], pos_x, "*", max_y-pos_y[pos_x]+1, THEME.get('spectrum.bar_between'))

        # Draw the max_db and min_db strings
        string = ">" + "%+03d" % self.max_db
        self.win.addnstr(0, self.dims[1] - self.chars, string, self.chars,
                         THEME.get('spectrum.max_db_label'))
        string = ">" + "%+03d" % self.min_db
        self.win.addnstr(max_y, self.dims[1] - self.chars, string,
                         self.chars, THEME.get('spectrum.min_db_label'))

        # Draw the threshold line
        self.win.hline(pos_yt, 0, "-", len(pos_y), THEME.get('spectrum.threshold_line'))

        # Draw the threshold string on the same row as the threshold line.
        # The inner window has no border, so addnstr at pos_yt == max_y is safe.
        string = ">" + "%+03d" % self.threshold_db
        self.win.addnstr(pos_yt, (self.dims[1] - self.chars), string,
                         self.chars, THEME.get('spectrum.threshold_label'))

        # Place a channel marker above the signal peak at its baseband frequency column.
        if channels is not None and self.samp_rate > 0:
            subset = [c for c in channels if c.active or c.hanging]
            occupied_rows = {}  # Tracks occupied columns per row: {row_y: set(col_indices)}

            def is_free(y: int, c: int, length: int) -> bool:
                occupied = occupied_rows.get(y, set())
                return all(idx not in occupied for idx in range(c, c + length))

            for display_idx, channel in enumerate(subset):
                # bb=0 is centre; -samp_rate/2 is left edge; +samp_rate/2 is right.
                # Look up the column using col_edges -- the exact same
                # partition the bars above were built from -- rather than
                # re-deriving the index->column mapping with a second
                # formula, so markers can never disagree with the bars.
                bin_idx = baseband_to_bin(channel.bb, self.samp_rate, L)
                col = index_to_column(bin_idx, col_edges)
                if row_map is not None and channel.rf in row_map:
                    label = str(row_map[channel.rf])
                else:
                    label = str(display_idx)
                # Clamp column so the label fits within the usable columns.
                col = max(0, min(num_cols - len(label), col))

                # Retrieve the top of the signal at this column
                signal_top = pos_y[col]
                target_y = max(0, signal_top - 2)

                # Find the nearest free row (staggering upwards first, then downwards)
                marker_y = target_y
                found = False
                for y in range(target_y, -1, -1):
                    if is_free(y, col, len(label)):
                        marker_y = y
                        found = True
                        break

                if not found:
                    for y in range(target_y + 1, self.dims[0]):
                        if is_free(y, col, len(label)):
                            marker_y = y
                            found = True
                            break

                # If every row is occupied at this column, marker_y stays at
                # target_y and overwrites whatever is there — acceptable graceful
                # degradation for extremely congested displays.

                # Record the occupied columns for this row
                if marker_y not in occupied_rows:
                    occupied_rows[marker_y] = set()
                for idx in range(col, col + len(label)):
                    occupied_rows[marker_y].add(idx)

                attr = (THEME.get('spectrum.channel_marker_active')
                        if channel.active else THEME.get('spectrum.channel_marker_hanging'))
                try:
                    self.win.addnstr(marker_y, col, label, len(label), attr)
                except curses.error:
                    pass
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

    def __init__(self, screen, width=None, num_demod: int | None = None):
        self.screen = screen

        self.num_demod = num_demod

        # Create a window object in the bottom half of the screen
        # Place on left side and to the right of the border
        # width defaults to 1/4 the screen width if not given explicitly
        # (e.g. by a caller coordinating widths across the bottom-row panels)
        screen_dims = screen.getmaxyx()
        spectrum_height = int(screen_dims[0]/2.0)
        height = screen_dims[0] - spectrum_height - 2
        if width is None:
            width = int(screen_dims[1]/4.0)
        self.outer_win = curses.newwin(height, width, screen_dims[0] - height - 1, 1)
        self.win = self.outer_win.derwin(height - 2, width - 2, 1, 1)
        self.dims = self.win.getmaxyx()

        self.entries: list[ChannelWindow.ChannelEntry] = []

        # Sticky-row state: maps rf frequency (float) -> row index so that a
        # frequency keeps the same display row for as long as it is visible
        # (active or hanging).  Rows are reclaimed into _free_rows only when a
        # frequency completely disappears from the channel list.
        self._row_map: dict[float, int] = {}
        self._rf_by_row: dict[int, float] = {}   # reverse of _row_map; O(1) row -> rf lookup
        self._free_rows: list[int] = []

    class ChannelEntry(object):
        win = None

        def __init__(self, row: int, col_offset: int, width: int):
            self.row = row
            self.col_offset = col_offset
            self.width = width
            self.prev_channel: ChannelFrequency | None = None
            self.prev_idx: int | None = None
            self.prev_show_placeholder: bool | None = None

            self.attrs = { 'bold_freq': THEME.get('channel.freq_active'),
                           'bold_icon': THEME.get('channel.icon_active'),
                           'normal_freq': THEME.get('channel.freq_inactive'),
                           'normal_icon': THEME.get('channel.icon_inactive') }

        def reset_cache(self) -> None:
            self.prev_channel = None
            self.prev_idx = None
            self.prev_show_placeholder = None

        def set(self, channel: ChannelFrequency, idx: int = 0,
                show_placeholder: bool = True) -> None:
            if (self.prev_channel != channel or self.prev_idx != idx
                    or self.prev_show_placeholder != show_placeholder):
                self.channel = channel
                self.idx = idx
                self.show_placeholder = show_placeholder
                self.draw()

            self.prev_channel = channel
            self.prev_idx = idx
            self.prev_show_placeholder = show_placeholder

        def draw(self) -> None:
            row = self.row
            col = self.col_offset
            channel = self.channel
            win = ChannelWindow.ChannelEntry.win

            if win is None:
                return

            # Clear the entire row width to prevent stale characters from previous draws (e.g. uncleared dots)
            win.addnstr(row, col, ' ' * self.width, self.width)

            if channel is None:
                if not self.show_placeholder:
                    return
                idx_str = f'{self.idx:>2d}:'
                scanning_str = ' Scanning...'
                win.addnstr(row, col, idx_str, len(idx_str),
                            THEME.get('channel.placeholder_index'))
                win.addnstr(row, col + len(idx_str), scanning_str,
                            max(0, self.width - len(idx_str)),
                            THEME.get('channel.placeholder_text'))
                return

            idx_str = f'{self.idx:>2d}:'
            freq_str = f'{channel.rf:>9.4f}'
            if freq_str.endswith('0'):
                freq_str = freq_str[:-1] + ' '

            icon = 'P' if channel.priority else ' '

            attributes = (self.attrs['bold_freq'], self.attrs['bold_icon']) if channel.active else (
                self.attrs['normal_freq'], self.attrs['normal_icon'])

            idx_attr = (THEME.get('channel.index_active')
                        if channel.active else THEME.get('channel.index_inactive'))
            win.addnstr(row, col, idx_str, len(idx_str), idx_attr)
            win.addnstr(row, col + 3, freq_str, len(freq_str), attributes[0])
            win.addnstr(row, col + 12, icon, 1, attributes[1])

            label_start = 14

            matched_ctcss = getattr(channel, 'matched_ctcss', None)
            primary_ctcss = channel.ctcss or (channel.ctcss_tones[0] if channel.ctcss_tones else None)

            has_multiple_ctcss = len(channel.ctcss_tones) > 1
            is_testing_ctcss = channel.active and not channel.hanging and has_multiple_ctcss and matched_ctcss is None

            if is_testing_ctcss:
                label = f"{channel.rf:.4f}..."
                display_ctcss = None
            else:
                label = channel.label or ''
                display_ctcss = matched_ctcss if matched_ctcss is not None else primary_ctcss


            if display_ctcss is not None:
                ctcss_str = f'{display_ctcss:>5.1f}'
                win.addnstr(row, col + self.width - 5, ctcss_str , 5, attributes[1] | curses.A_ITALIC)
                win.addnstr(row, col + self.width - 6, ' ', 1, attributes[0])
                label_end = self.width - 6
            else:
                label_end = self.width

            if label_end > label_start:
                remainder = label_end - label_start
                win.addnstr(row, col + label_start, label.ljust(remainder), remainder, attributes[0])

    def draw_frame(self) -> None:
        self.outer_win.erase()
        self.outer_win.attron(THEME.get('channel.border'))
        self.outer_win.border(0)
        outer_width = self.dims[1] + 2
        self.outer_win.addnstr(0, (outer_width - 8) // 2, "CHANNELS", 8,
                               THEME.get('channel.title'))
        self.outer_win.leaveok(1)
        self.outer_win.noutrefresh()

        self.win.erase()

        ChannelWindow.ChannelEntry.win = self.win

        # Force a single column layout to maximize label width
        usable_height = self.dims[0] - 1
        actual_col_width = self.dims[1]

        # Only rebuild the entry list when the window geometry changes so that
        # per-entry dirty-tracking caches (prev_channel, prev_idx) are preserved
        # across normal draw cycles.
        if len(self.entries) != usable_height or (
                self.entries and self.entries[0].width != actual_col_width):
            self.entries = [
                ChannelWindow.ChannelEntry(i, 0, actual_col_width)
                for i in range(usable_height)
            ]
            # Reset caches after a geometry-driven rebuild to force a full redraw
            for entry in self.entries:
                entry.reset_cache()
            # Geometry changed: discard all sticky-row assignments so they are
            # rebuilt cleanly against the new entry count.
            self._row_map.clear()
            self._rf_by_row.clear()
            self._free_rows = list(range(usable_height))

    def draw_channels(self, channels: list[ChannelFrequency]):
        """Draws tuned channels list using sticky row anchoring.

        Each RF frequency is assigned a row index the first time it becomes
        visible (active or hanging) and keeps that row until it completely
        disappears from the channel list.  Rows are reclaimed into a free pool
        in ascending order so newly arriving frequencies take the lowest
        available slot, giving a natural top-to-bottom fill pattern.

        The label shown next to each frequency is its row index, which is
        permanent for as long as the frequency is visible.  The keyboard
        lockout hotkeys (``0``–``9``) correspond to these row indices.
        ham2mon uses get_rf_by_row() to translate the pressed digit into an
        RF frequency which is passed directly to Scanner.add_lockout().
        """
        subset = [c for c in channels if c.active or c.hanging]
        visible_rfs = {c.rf for c in subset}

        # Initialise _free_rows on the very first call (before any geometry
        # change has fired draw_frame with a mismatched entry count).
        if not self._row_map and not self._free_rows and self.entries:
            self._free_rows = list(range(len(self.entries)))

        # Release rows belonging to frequencies that are no longer visible.
        released = [row for rf, row in self._row_map.items() if rf not in visible_rfs]
        for rf in [rf for rf in list(self._row_map) if rf not in visible_rfs]:
            row = self._row_map.pop(rf)
            self._rf_by_row.pop(row, None)
        # Keep free list sorted ascending so the lowest slot is always taken
        # first, preserving a tidy top-to-bottom fill order.
        self._free_rows = sorted(self._free_rows + released)

        # Assign a row to any newly visible frequency.
        for channel in subset:
            if channel.rf not in self._row_map and self._free_rows:
                row = self._free_rows.pop(0)
                self._row_map[channel.rf] = row
                self._rf_by_row[row] = channel.rf

        # Build a row-indexed view: row -> channel.
        row_content: dict[int, ChannelFrequency] = {}
        for channel in subset:
            row = self._row_map.get(channel.rf)
            if row is not None:
                row_content[row] = channel

        # Render every entry: draw the assigned channel at its stable row index,
        # passing the row number as the display index (lockout hotkey label).
        # row_idx is always passed, even for idle rows, so the "Scanning..."
        # placeholder in ChannelEntry.draw() can display the correct row
        # number for empty slots too.
        for row_idx, entry in enumerate(self.entries):
            is_demod_slot = self.num_demod is None or row_idx < self.num_demod
            entry.set(row_content.get(row_idx), row_idx,
                      show_placeholder=is_demod_slot)

        # Hide cursor
        self.win.leaveok(1)

        # Update virtual window
        self.win.noutrefresh()

    def get_rf_by_row(self, row: int) -> float | None:
        """Return the RF frequency currently anchored to *row*, or None.

        Used by the lockout key handler in ham2mon to translate a pressed
        digit (row number) into an RF frequency for Scanner.add_lockout().
        """
        return self._rf_by_row.get(row)

    def get_row_map(self) -> dict[float, int]:
        """Return a snapshot of the current RF-frequency → row-index mapping.

        The returned dict is a shallow copy so callers cannot inadvertently
        mutate internal state.  Used by SpectrumWindow to draw channel index
        markers that match the Channel panel row labels and lockout hotkeys.
        """
        return dict(self._row_map)

    def cleanup(self) -> None:
        if hasattr(self, 'win') and self.win:
            try:
                self.win.erase()
                del self.win
            except Exception:
                pass
        if hasattr(self, 'outer_win') and self.outer_win:
            try:
                self.outer_win.erase()
                del self.outer_win
            except Exception:
                pass


class LockoutWindow(object):
    """Curses lockout channel display window

    Args:
        screen (object): a curses screen object
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, screen, width=None, x_offset=None):
        self.screen = screen

        # Create a window object in the bottom half of the screen
        # Place to the right of the Channel window
        # width/x_offset default to the old 1/4-screen-each formula if not
        # given explicitly (e.g. by a caller coordinating widths across the
        # bottom-row panels)
        screen_dims = screen.getmaxyx()
        spectrum_height = int(screen_dims[0]/2.0)
        height = screen_dims[0] - spectrum_height - 2
        if width is None:
            width = int(screen_dims[1]/4.0)
        if x_offset is None:
            x_offset = width + 1
        self.outer_win = curses.newwin(height, width, screen_dims[0] - height - 1, x_offset)
        self.win = self.outer_win.derwin(height - 2, width - 2, 1, 1)
        self.dims = self.win.getmaxyx()

        self.lockouts: list[LockoutWindow.LockoutEntry] = []

    class LockoutEntry(object):
        win = None

        def __init__(self, row: int, col_offset: int, width: int):
            self.row = row
            self.col_offset = col_offset
            self.width = width
            self.prev_lockout: ConfigFrequency | None = None
            self.prev_has_activity: bool | None = None

            self.attrs = { 'bold_lockout': THEME.get('lockout.active'),
                           'normal_lockout': THEME.get('lockout.inactive') }

        def reset_cache(self) -> None:
            self.prev_lockout = None
            self.prev_has_activity = None

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
            col = self.col_offset
            lockout = self.lockout
            has_activity = self.has_activity
            win = LockoutWindow.LockoutEntry.win

            if win is None:
                return

            if lockout is None:
                text = ' ' * self.width
                win.addnstr(row, col, text, self.width)
                return

            if not lockout.saved:
                icon = 'U'
            else:
                icon = ' '

            attr = self.attrs['normal_lockout']

            if lockout.is_single:
                text = f"{lockout.single:.4f}"
                if text.endswith('0'):
                    text = text[:-1] + ' '
                if has_activity:
                    attr = self.attrs['bold_lockout']
            else:
                text = f"{lockout.lo:.3f}-{lockout.hi:.3f}"
                if has_activity:
                    attr = self.attrs['bold_lockout']

            # Make sure we don't write out of column bounds
            win.addnstr(row, col, text, self.width, attr)

            icon_pos = len(text)
            if self.width > icon_pos:
                win.addnstr(row, col + icon_pos, icon, 1, attr & ~curses.A_BOLD)

            label = lockout.label or ''
            label_start = icon_pos + 2  # leave 1 space separator after the icon

            if self.width > label_start:
                remainder = self.width - label_start
                win.addnstr(row, col + label_start, label.ljust(remainder), remainder, attr & ~curses.A_BOLD)

    def draw_frame(self) -> None:
        self.outer_win.erase()
        self.outer_win.attron(THEME.get('lockout.border'))
        self.outer_win.border(0)
        outer_width = self.dims[1] + 2
        self.outer_win.addnstr(0, (outer_width - 7) // 2, "LOCKOUT", 7,
                        THEME.get('lockout.title'))
        self.outer_win.leaveok(1)
        self.outer_win.noutrefresh()

        self.win.erase()

        LockoutWindow.LockoutEntry.win = self.win

        # Force a single column layout to maximize label/range space
        usable_height = self.dims[0] - 1
        actual_col_width = self.dims[1]

        # Only rebuild the lockout list when the window geometry changes so that
        # per-entry dirty-tracking caches (prev_lockout, prev_has_activity) are
        # preserved across normal draw cycles.
        if len(self.lockouts) != usable_height or (
                self.lockouts and self.lockouts[0].width != actual_col_width):
            self.lockouts = [
                LockoutWindow.LockoutEntry(i, 0, actual_col_width)
                for i in range(usable_height)
            ]
            # Reset caches after a geometry-driven rebuild to force a full redraw
            for lockout in self.lockouts:
                lockout.reset_cache()

    def lockout_has_activity(self, lockout: ConfigFrequency) -> bool:
        """Checks if lockout has activity

        Args:
            lockout (RadioFreq): lockout to check
        """
        has_activity = False
        for channel in self.locked_channels:
            if lockout.is_single:
                if lockout.single == channel.rf:
                    has_activity = True
            else:
                if lockout.lo <= channel.rf <= lockout.hi:
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

        # Extract the frequencies/ranges the user wants locked out
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

    def cleanup(self) -> None:
        if hasattr(self, 'win') and self.win:
            try:
                self.win.erase()
                del self.win
            except Exception:
                pass
        if hasattr(self, 'outer_win') and self.outer_win:
            try:
                self.outer_win.erase()
                del self.outer_win
            except Exception:
                pass

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
        activity_dest (string): Name of file or endpoint for channel activity logging
        activity_interval (int): Timeout delay between logging active state of channel in seconds
        log_mode (string): Log system mode (file, database type)
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, screen, width=None):
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
        self.activity_type = ""
        self.activity_dest = ""
        self.gains = None
        self.classifier_params = None

        self.demod_map = {
            0: 'NBFM',
            1: 'AM',
            2: 'WBFM',
        }

        # Create a window object in the bottom half of the screen
        # Place on right side and to the left of the border
        # width defaults to the old screen-remainder formula if not given
        # explicitly (e.g. by a caller coordinating widths across the
        # bottom-row panels)
        screen_dims = screen.getmaxyx()
        spectrum_height = int(screen_dims[0]/2.0)
        height = screen_dims[0] - spectrum_height - 2
        if width is None:
            # subtract the channel and lockout widths
            width = screen_dims[1] - 2 * int(screen_dims[1]/4.0) - 2
        self.outer_win = curses.newwin(height, width, screen_dims[0] - height - 1,
                                 int(screen_dims[1]-width-1))
        self.win = self.outer_win.derwin(height - 2, width - 2, 1, 1)
        self.dims = self.win.getmaxyx()

    class RxEntry(object):

        label_width = 14
        value_width = 10
        column_width = 27 # label_width + value_width + 3
        #rows = [0, 0]  # used to track the next index for each row

        def __init__(self, label: str | None, column: int,
                     justification: str, can_modify: bool):
            self.label = label
            self.row = RxWindow.RxEntry.rows[column-1]
            RxWindow.RxEntry.rows[column-1] += 1
            self.column = column
            self.justification = justification
            self.can_modify = can_modify
            self.prev_value: str | None = None

            win = RxWindow.RxEntry.win
            label_width = RxWindow.RxEntry.label_width

            self.attrs = { 'bold': THEME.get('receiver.value_editable'),
                           'normal': THEME.get('receiver.value_readonly') }

            if self.label is None:
                return

            if justification == 'left':
                text = f'{self.label:<{label_width}}:'
            else:
                text = f'{self.label:>{label_width-1}} :'

            col_start = (self.column - 1) * RxWindow.RxEntry.column_width
            win.addnstr(self.row, col_start, text, len(text), self.attrs['normal'])

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
            win = RxWindow.RxEntry.win

            if win is None:
                return

            # Dynamically calculate the maximum allowed width for this value to avoid wrapping and ERR
            win_width = win.getmaxyx()[1]
            col_start = (self.column - 1) * RxWindow.RxEntry.column_width
            val_start = col_start + (RxWindow.RxEntry.label_width + 2 if self.label is not None else 0)

            if self.column == 1:
                # Column 1 values must not overlap Column 2
                max_len = RxWindow.RxEntry.column_width - (val_start - col_start) - 1
            else:
                # Column 2 (or last column) values can utilize all remaining space to the right border
                max_len = win_width - val_start - 1

            max_len = max(0, max_len)

            # None check must come before truncation so the blank-clear path is reachable
            if value is None:
                try:
                    win.addnstr(row, val_start, ' ' * max_len, max_len)
                except curses.error:
                    pass
                return

            if len(value) > max_len:
                value = value[:max_len]

            attr = self.attrs['bold'] if self.can_modify else self.attrs['normal']
            try:
                win.addnstr(row, val_start, value, len(value), attr)
            except curses.error:
                pass

    def draw_frame(self) -> None:
        self.outer_win.erase()
        self.outer_win.attron(THEME.get('receiver.border'))
        self.outer_win.border(0)
        outer_width = self.dims[1] + 2
        self.outer_win.addnstr(0, (outer_width - 8) // 2, "RECEIVER", 8,
                         THEME.get('receiver.title'))
        self.outer_win.leaveok(1)
        self.outer_win.noutrefresh()

        self.win.erase()

        # Warn when the inner window is too narrow for the two-column layout.
        # column_width * 2 columns are needed; narrower terminals will silently
        # drop column-2 values (max_len clamps to 0) without this notice.
        min_two_col_width = RxWindow.RxEntry.column_width * 2
        if self.dims[1] < min_two_col_width:
            logger.warning(
                "RxWindow inner width %d < %d; column-2 fields will not be drawn",
                self.dims[1], min_two_col_width)

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

        self.activity_type_field = RxWindow.RxEntry(
            "Activity Type", 2, 'left', False)

        if self.activity_dest is not None:
            self.activity_dest_field = RxWindow.RxEntry(
                "Activity Dest", 2, 'left', False)

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

        file_name = self.frequency_file_name.name if self.frequency_file_name else "none"
        self.frequency_file_name_field.set(file_name)

        self.activity_type_field.set(self.activity_type)

        if self.activity_dest is not None:
            self.activity_dest_field.set(self.activity_dest)

        # Hide cursor
        self.win.leaveok(1)

        # Update virtual window
        self.win.noutrefresh()

    def cleanup(self) -> None:
        if hasattr(self, 'win') and self.win:
            try:
                self.win.erase()
                del self.win
            except Exception:
                pass
        if hasattr(self, 'outer_win') and self.outer_win:
            try:
                self.outer_win.erase()
                del self.outer_win
            except Exception:
                pass

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

    def _adjust_gain_stage(self, index: int, delta: float) -> bool:
        if index < len(self.gains):
            self.gains[index]["value"] += delta
            return True
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
            return self._adjust_gain_stage(0, 10)
        elif keyb == ord('f'):
            return self._adjust_gain_stage(0, -10)

        # Tune 1st gain element in 1 dB steps with 'G' and 'F'
        if keyb == ord('G'):
            return self._adjust_gain_stage(0, 1)
        elif keyb == ord('F'):
            return self._adjust_gain_stage(0, -1)

        # Tune 2nd gain element in 10 dB steps with 'u' and 'y'
        if keyb == ord('u'):
            return self._adjust_gain_stage(1, 10)
        elif keyb == ord('y'):
            return self._adjust_gain_stage(1, -10)

        # Tune 2nd gain element in 1 dB steps with 'U' and 'Y'
        if keyb == ord('U'):
            return self._adjust_gain_stage(1, 1)
        elif keyb == ord('Y'):
            return self._adjust_gain_stage(1, -1)

        # Tune 3rd gain element in 10 dB steps with ']' and '['
        if keyb == ord(']'):
            return self._adjust_gain_stage(2, 10)
        elif keyb == ord('['):
            return self._adjust_gain_stage(2, -10)

        # Tune 3rd gain element in 1 dB steps with '}' and '{'
        if keyb == ord('}'):
            return self._adjust_gain_stage(2, 1)
        elif keyb == ord('{'):
            return self._adjust_gain_stage(2, -1)

        # Tune self.squelch_db in 1 dB steps with 's' and 'a'
        if keyb == ord('s'):
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


def create_bottom_row_windows(screen, chan_min_width=20, lock_min_width=20,
                              weights=None, num_demod: int | None = None):
    """Creates and positions the Channel, Lockout, and Receiver windows.

    Channel and Lockout display open-ended, user-supplied label text, so
    they benefit from extra terminal width more than Receiver does, whose
    content is fixed-format label:value columns that stop needing more
    room once the columns fit. As the terminal widens, this gives
    Channel/Lockout a larger share of the new space instead of all three
    panels growing at the same fixed proportion of the terminal width.

    Args:
        screen (object): a curses screen object
        chan_min_width (int): minimum width to reserve for the Channel panel
        lock_min_width (int): minimum width to reserve for the Lockout panel
        weights (dict): optional override of {'chan', 'lock', 'rx'} growth
            weights; defaults to Channel/Lockout growing twice as fast as
            Receiver once minimums are satisfied
        num_demod (int): optional count of available demodulators to cap
            scanning placeholder rendering

    Returns:
        tuple: (ChannelWindow, LockoutWindow, RxWindow) instances
    """
    if weights is None:
        weights = {'chan': 2, 'lock': 2, 'rx': 1}

    screen_dims = screen.getmaxyx()
    # Matches the original layout math: 1 border column reserved on each
    # side of the screen, with no gap between the three panels themselves.
    total_usable_width = screen_dims[1] - 2

    # Receiver's minimum is its actual content requirement: two columns of
    # label:value pairs plus its own left/right border.
    rx_min_width = RxWindow.RxEntry.column_width * 2 + 2

    min_widths = {'chan': chan_min_width, 'lock': lock_min_width, 'rx': rx_min_width}
    widths = compute_panel_widths(total_usable_width, min_widths, weights)

    chanwin = ChannelWindow(screen, width=widths['chan'], num_demod=num_demod)
    lockoutwin = LockoutWindow(screen, width=widths['lock'], x_offset=widths['chan'] + 1)
    rxwin = RxWindow(screen, width=widths['rx'])
    return chanwin, lockoutwin, rxwin


def setup_screen(screen, theme_config_path: Path | None = None) -> None:
    """Sets up screen

    Args:
        theme_config_path (Path, optional): Path to a theme YAML
            file (see ui_theme.py). If omitted, default-theme.yaml is used.
            This parameter exists so a future --theme-config CLI flag can be
            added.
    """

    # hide cursor
    curses.curs_set(0)

    # do not echo keystrokes
    curses.noecho()

    # break on ctrl-c
    curses.cbreak()

    # Load and resolve the UI theme config. This is the only place any
    # curses.init_pair()/color_pair() resolution happens -- a one-time,
    # startup-only cost. curses.start_color() must already have been called
    # by the caller (see main() below) before this runs.
    THEME.config = ThemeConfiguration(file_name=theme_config_path)
    THEME.load()
    THEME.resolve()

    # Add border
    screen.attron(THEME.get('screen.border'))
    screen.border(0)
    screen.attroff(THEME.get('screen.border'))

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
    chanwin, lockoutwin, rxwin = create_bottom_row_windows(screen)

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
