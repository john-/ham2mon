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
#from pathlib import Path
from os.path import realpath, dirname

import _curses

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
        self.chanwin, self.lockoutwin, self.rxwin = cursesgui.create_bottom_row_windows(self.stdscr)

        # Get the initial settings for GUI
        self.rxwin.gains = self.scanner.filter_and_set_gains(PARSER.gains)
        self.rxwin.center_freq = self.scanner.center_freq
        self.rxwin.step = self.scanner.step
        self.rxwin.steps = self.scanner.steps
        self.rxwin.samp_rate = self.scanner.samp_rate
        self.rxwin.squelch_db = self.scanner.squelch_db
        self.rxwin.volume_db = self.scanner.volume_db
        self.rxwin.record = self.scanner.record
        self.rxwin.type_demod = PARSER.type_demod
        self.rxwin.frequency_file_name = self.scanner.frequency_file_name
        self.rxwin.channel_log_type = self.scanner.channel_log_params.type
        # not all channel_log types use a target
        if self.scanner.channel_log_params.type == 'fixed-field':
            target = self.scanner.channel_log_params.target
        else:
            target = None
        self.rxwin.channel_log_target = target

        self.specwin.max_db = PARSER.max_db
        self.specwin.min_db = PARSER.min_db
        self.rxwin.classifier_params = PARSER.classifier_params
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
        self.specwin.draw_spectrum(self.scanner.spectrum)
        self.chanwin.draw_channels(self.scanner.channels)
        self.lockoutwin.draw_channels(self.scanner.frequencies, self.scanner.channels)
        self.rxwin.draw_rx()

        # Update physical screen via optimized double buffering
        curses.doupdate()

    async def init_scanner(self) -> scnr.Scanner:
        # Create scanner object
        ask_samp_rate = PARSER.ask_samp_rate
        num_demod = PARSER.num_demod
        type_demod = PARSER.type_demod
        hw_args = PARSER.hw_args
        record = PARSER.record
        play = PARSER.play
        frequency_configuration = PARSER.frequency_configuration
        channel_log_params = PARSER.channel_log_params
        freq_correction = PARSER.freq_correction
        audio_bps = PARSER.audio_bps
        channel_spacing = PARSER.channel_spacing

        frequency_params = PARSER.frequency_params
        frequency_params.notify_interface = self.center_freq_changed

        agc = PARSER.agc

        min_recording = PARSER.min_recording
        max_recording = PARSER.max_recording

        classifier_params = PARSER.classifier_params

        auto_priority = PARSER.auto_priority
        file_metadata = PARSER.file_metadata

        scanner = scnr.Scanner(ask_samp_rate, num_demod, type_demod, hw_args,
                               freq_correction, record, frequency_configuration,
                               channel_log_params,
                               play, audio_bps, channel_spacing,
                               frequency_params, min_recording, max_recording,
                               classifier_params, auto_priority, agc,
                               file_metadata=file_metadata)

        await scanner.load_frequencies()
        # Set the parameters
        scanner.set_center_freq(scanner.center_freq)
        scanner.set_squelch(PARSER.squelch_db)
        scanner.set_volume(PARSER.volume_db)
        scanner.set_threshold(PARSER.threshold_db)

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
            self.scanner.set_center_freq(self.rxwin.center_freq)
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
            # Subtract 48 from ascii keyb value to obtain 0 - 9
            idx = keyb - 48
            await self.scanner.add_lockout(idx)
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

        if PARSER.log_dest != 'none':
            log_level_map = {
                'debug': logging.DEBUG,
                'info': logging.INFO,
                'warn': logging.WARNING,
                'error': logging.ERROR
            }
            level = log_level_map.get(PARSER.log_level, logging.WARNING)

            logger = logging.getLogger()
            logger.setLevel(level)

            formatter = logging.Formatter('%(asctime)s %(message)s')

            if PARSER.log_dest == 'syslog':
                syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')
                syslog_handler.setFormatter(logging.Formatter('ham2mon[%(process)d]: %(message)s'))
                logger.addHandler(syslog_handler)
            elif PARSER.log_dest == 'stderr':
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(formatter)
                logger.addHandler(stream_handler)
            elif PARSER.log_dest == 'file':
                log_file_path = PARSER.log_file
                if not log_file_path:
                    script_dir = realpath(dirname(__file__))
                    log_file_path = '%s/ham2mon.log' % (script_dir)
                file_handler = logging.FileHandler(log_file_path, delay=True)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

        wrapper(main)
    except KeyboardInterrupt:
        pass
    except RuntimeError as error:
        print("")
        print("RuntimeError: SDR hardware not detected or insufficient USB permissions. Try running as root or with --log-level=debug option.")
        print("")
        print(f'RuntimeError: {error=}, {type(error)=}')
        logging.debug(traceback.format_exc())
        print("")
    except err.LogError:
        print("")
        print("LogError: database logging not active, to be expanded.")
        print("")
    except OSError as error:
        print("")
        print(f'OS error: {error=}, {type(error)=}')
        logging.debug(traceback.format_exc())
        print("")
