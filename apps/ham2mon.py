#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:38:36 2015

@author: madengr
"""

import scanner as scnr
from curses import ERR, KEY_RESIZE, curs_set, wrapper
import curses
import cursesgui
import h2m_parser as h2m_parser
import asyncio
import errors as err
import logging
import logging.handlers
import traceback
from pathlib import Path
from os.path import realpath, dirname

import _curses

logger = logging.getLogger("ham2mon")

class MyDisplay():

    def __init__(self, stdscr: "_curses._CursesWindow") -> None:
        self.stdscr = stdscr
        self.scanner = None
        self.too_small = False

    async def run(self) -> None:
        curs_set(0)
        self.stdscr.nodelay(True)

        self.scanner = await self.init_scanner()

        await self.make_display()

        while True:
            char = self.stdscr.getch()

            if char == ord('Q') or (self.scanner and self.scanner.eof):
                break
            if char == ERR:
                await asyncio.sleep(0.1)
            elif char == KEY_RESIZE:
                await self.make_display()
            else:
                await self.handle_char(char)

            await self.cycle()

        self.scanner.stop()
        await self.scanner.clean_up()

    async def make_display(self) -> None:
        """Start scanner with GUI interface

        Initialize and set up screen
        Create windows
        """
        # pylint: disable=too-many-statements
        # pylint: disable-msg=R0914

        # Clean up existing sub-windows to prevent C-level resource memory leaks
        for win_name in ('specwin', 'chanwin', 'lockoutwin', 'rxwin'):
            if hasattr(self, win_name):
                win_obj = getattr(self, win_name)
                if win_obj and hasattr(win_obj, 'cleanup'):
                    win_obj.cleanup()

        # Erase the full screen and reset ncurses parameters
        self.stdscr.clear()
        cursesgui.setup_screen(self.stdscr)

        # Enforce minimum terminal dimensions to prevent negative window sizing crash
        screen_dims = self.stdscr.getmaxyx()
        if screen_dims[0] < 24 or screen_dims[1] < 80:
            self.too_small = True
            self.stdscr.erase()
            self.stdscr.border(0)
            self.stdscr.addstr(1, 2, "Terminal too small!", curses.A_BOLD | curses.color_pair(1))
            self.stdscr.addstr(2, 2, f"Current: {screen_dims[1]}x{screen_dims[0]}", curses.A_DIM)
            self.stdscr.addstr(3, 2, "Required minimum: 80x24", curses.A_DIM)
            self.stdscr.noutrefresh()
            return
        else:
            self.too_small = False

        # Create windows
        self.specwin = cursesgui.SpectrumWindow(self.stdscr)
        num_demod = None
        if self.scanner and hasattr(self.scanner, 'receiver') and hasattr(self.scanner.receiver, 'demodulators'):
            num_demod = len(self.scanner.receiver.demodulators)
        self.chanwin, self.lockoutwin, self.rxwin = cursesgui.create_bottom_row_windows(
            self.stdscr, num_demod=num_demod)

        # Get the initial settings for GUI
        self.rxwin.gains = self.scanner.filter_and_set_gains(PARSER.master_config.gains)
        self.rxwin.center_freq = self.scanner.center_freq
        self.rxwin.step = self.scanner.step
        self.rxwin.steps = self.scanner.steps
        self.rxwin.samp_rate = self.scanner.samp_rate
        self.specwin.samp_rate = self.scanner.samp_rate
        self.rxwin.squelch_db = self.scanner.squelch_db
        self.rxwin.volume_db = self.scanner.volume_db
        self.rxwin.record = self.scanner.record
        self.rxwin.type_demod = PARSER.master_config.receiver.mode
        self.rxwin.frequency_file_name = self.scanner.frequency_file_name
        self.rxwin.activity_type = self.scanner.activity_params.type
        # not all activity types use a dest
        if self.scanner.activity_params.type == 'fixed-field':
            dest = self.scanner.activity_params.dest
        else:
            dest = None
        self.rxwin.activity_dest = dest

        self.specwin.max_db = PARSER.master_config.display.max_db
        self.specwin.min_db = PARSER.master_config.display.min_db
        self.specwin.threshold_db = self.scanner.threshold_db

        # Update virtual representation of root screen first
        self.stdscr.noutrefresh()

        self.chanwin.draw_frame()
        self.lockoutwin.draw_frame()
        self.rxwin.draw_frame()

    async def cycle(self) -> None:
        # Initiate a scan cycle

        # No need to go faster than 10 Hz rate of GNU Radio probe
        await asyncio.sleep(0.1)

        await self.scanner.scan_cycle()

        if getattr(self, 'too_small', False):
            # Double-buffer refresh only the resized root window holding the warning
            curses.doupdate()
            return

        # Update the spectrum, channel, and rx displays
        self.chanwin.draw_channels(self.scanner.channels)
        self.specwin.draw_spectrum(self.scanner.spectrum, self.scanner.channels, self.chanwin.get_row_map())
        self.lockoutwin.draw_channels(self.scanner.frequencies, self.scanner.channels)
        self.rxwin.draw_rx()

        # Update physical screen via optimized double buffering
        curses.doupdate()

    async def init_scanner(self) -> scnr.Scanner:
        frequency_params = PARSER.frequency_params
        frequency_params.notify_interface = self.center_freq_changed

        scanner = scnr.Scanner(PARSER.master_config, frequency_params)

        if PARSER.master_config.frequency_policies.active_banks:
            scanner.frequency_manager.set_active_banks(
                PARSER.master_config.frequency_policies.active_banks
            )

        await scanner.load_frequencies()
        # Set the parameters
        scanner.set_center_freq(scanner.center_freq)
        scanner.set_squelch(PARSER.master_config.receiver.squelch_db)
        scanner.set_volume(PARSER.master_config.audio.volume_db)
        scanner.set_threshold(PARSER.master_config.receiver.threshold_db)

        return scanner

    def center_freq_changed(self):
        '''
        Callback that notifies when the scanner changed
        the center frequency.  This occurs when range scanning.
        '''
        self.rxwin.center_freq = self.scanner.center_freq
        self.rxwin.step = self.scanner.step
        self.rxwin.steps = self.scanner.steps

    async def handle_char(self, keyb: int) -> None:
        # Send keystroke to spectrum window and update scanner if True
        if self.specwin.proc_keyb(keyb):
            self.scanner.set_threshold(self.specwin.threshold_db)

        # Send keystroke to RX window and update scanner if True
        if self.rxwin.proc_keyb_hard(keyb):
            # Set and update frequency
            self.scanner.set_center_freq(int(self.rxwin.center_freq))
            self.rxwin.center_freq = self.scanner.center_freq

        if self.rxwin.proc_keyb_soft(keyb):
            # Set all the gains
            self.rxwin.gains = self.scanner.set_gains(self.rxwin.gains)
            # Set and update squelch
            self.scanner.set_squelch(self.rxwin.squelch_db)
            self.rxwin.squelch_db = self.scanner.squelch_db
            # Set and update volume
            self.scanner.set_volume(self.rxwin.volume_db)
            self.rxwin.volume_db = self.scanner.volume_db

        # Send keystroke to lockout window and update lockout channels if True
        if self.lockoutwin.proc_keyb_set_lockout(keyb) and self.rxwin.freq_entry == 'None':
            # Translate pressed digit to the stable row number, then look up
            # the RF frequency anchored to that row.  get_rf_by_row() returns
            # None when no channel occupies the row, in which case we skip.
            row = keyb - 48
            rf = self.chanwin.get_rf_by_row(row)
            if rf is not None:
                await self.scanner.add_lockout(rf)
        if self.lockoutwin.proc_keyb_clear_lockout(keyb):
            await self.scanner.clear_lockout()

async def display_main(stdscr) -> None:
    display = MyDisplay(stdscr)
    await display.run()

def main(stdscr) -> None:
    return asyncio.run(display_main(stdscr))

if __name__ == '__main__':

    try:
        # Do this since curses wrapper won't let parser write to screen
        PARSER = h2m_parser.CLParser()

        log_cfg = PARSER.master_config.logging
        if log_cfg.dest != 'none':
            logging.raiseExceptions = False
            log_level_map = {
                'debug': logging.DEBUG,
                'info': logging.INFO,
                'warn': logging.WARNING,
                'error': logging.ERROR
            }
            level = log_level_map.get(log_cfg.level, logging.WARNING)

            logger.setLevel(level)
            logger.propagate = False

            formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

            if log_cfg.dest == 'syslog':
                syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')
                syslog_handler.setFormatter(logging.Formatter('ham2mon[%(process)d]: [%(name)s] %(levelname)s: %(message)s'))
                logger.addHandler(syslog_handler)
            elif log_cfg.dest == 'stderr':
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(formatter)
                logger.addHandler(stream_handler)
            elif log_cfg.dest == 'file':
                log_file_path = log_cfg.file
                if not log_file_path:
                    script_dir = realpath(dirname(__file__))
                    log_file_path = '%s/ham2mon.log' % (script_dir)
                file_handler = logging.FileHandler(log_file_path, delay=True)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            # Suppress chatty third-party loggers that would otherwise pollute output
            logging.getLogger("urllib3").setLevel(logging.WARNING)

        wrapper(main)
    except KeyboardInterrupt:
        pass
    except RuntimeError as error:
        print("")
        print("RuntimeError: SDR hardware not detected or insufficient USB permissions. Try running as root or with --log-level=debug option.")
        print("")
        print(f'RuntimeError: {error=}, {type(error)=}')
        logger.debug(traceback.format_exc())
        print("")
    except err.LogError:
        print("")
        print("LogError: database logging not active, to be expanded.")
        print("")
    except OSError as error:
        print("")
        print(f'OS error: {error=}, {type(error)=}')
        logger.debug(traceback.format_exc())
        print("")
