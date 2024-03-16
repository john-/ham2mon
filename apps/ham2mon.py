#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:38:36 2015

@author: madengr
"""

import scanner as scnr
from curses import ERR, KEY_RESIZE, curs_set, wrapper, echo, nocbreak, endwin
import cursesgui
import h2m_parser as h2m_parser
import asyncio
import errors as err
import logging
import traceback
from os.path import realpath, dirname


import _curses

class MyDisplay():

    def __init__(self, stdscr: "_curses._CursesWindow") -> None:
        self.stdscr = stdscr

    async def run(self) -> None:
        curs_set(0)
        self.stdscr.nodelay(True)

        self.scanner = self.init_scanner()

        self.make_display()

        while True:
            char = self.stdscr.getch()

            if char == ord('Q'):
                break
            if char == ERR:
                await asyncio.sleep(0.1)
            elif char == KEY_RESIZE:
                self.make_display()
            else:
                self.handle_char(char)

            await self.cycle()

        await self.scanner.clean_up()

    def make_display(self) -> None:
        """Start scanner with GUI interface

        Initialize and set up screen
        Create windows
        """
        # pylint: disable=too-many-statements
        # pylint: disable-msg=R0914

        # Setup the screen
        cursesgui.setup_screen(self.stdscr)

        # Create windows
        self.specwin = cursesgui.SpectrumWindow(self.stdscr)
        self.chanwin = cursesgui.ChannelWindow(self.stdscr)
        self.lockoutwin = cursesgui.LockoutWindow(self.stdscr)
        self.rxwin = cursesgui.RxWindow(self.stdscr)

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
        self.rxwin.lockout_file_name = self.scanner.lockout_file_name
        self.rxwin.priority_file_name = self.scanner.priority_file_name
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

    async def cycle(self) -> None:
        # Initiate a scan cycle

        # No need to go faster than 10 Hz rate of GNU Radio probe
        await asyncio.sleep(0.1)

        await self.scanner.scan_cycle()

        # Update the spectrum, channel, and rx displays
        self.specwin.draw_spectrum(self.scanner.spectrum)
        self.chanwin.draw_channels(self.scanner.get_channels())
        self.lockoutwin.draw_channels(self.scanner.gui_lockout_channels, self.scanner.get_channels())
        self.rxwin.draw_rx()

        # Update physical screen
        self.stdscr.refresh()

    def init_scanner(self) -> scnr.Scanner:
        # Create scanner object
        ask_samp_rate = PARSER.ask_samp_rate
        num_demod = PARSER.num_demod
        type_demod = PARSER.type_demod
        hw_args = PARSER.hw_args
        record = PARSER.record
        play = PARSER.play
        lockout_file_name = PARSER.lockout_file_name
        priority_file_name = PARSER.priority_file_name
        channel_log_params = PARSER.channel_log_params
        freq_correction = PARSER.freq_correction
        audio_bps = PARSER.audio_bps
        channel_spacing = PARSER.channel_spacing

        frequency_params = PARSER.frequency_params
        frequency_params.notify_interface = self.center_freq_changed

        min_recording = PARSER.min_recording
        max_recording = PARSER.max_recording
        classifier_params = PARSER.classifier_params

        scanner = scnr.Scanner(ask_samp_rate, num_demod, type_demod, hw_args,
                               freq_correction, record, lockout_file_name,
                               priority_file_name, channel_log_params,
                               play, audio_bps, channel_spacing,
                               frequency_params,
                               min_recording, max_recording, classifier_params)

        # Set the parameters
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

    def handle_char(self, keyb: int) -> None:
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
            self.scanner.add_lockout(idx)
        if self.lockoutwin.proc_keyb_clear_lockout(keyb):
            self.scanner.clear_lockout()

async def display_main(stdscr) -> None:
    display = MyDisplay(stdscr)
    await display.run()

def main(stdscr) -> None:
    return asyncio.run(display_main(stdscr))

if __name__ == '__main__':

    try:
        # Do this since curses wrapper won't let parser write to screen
        PARSER = h2m_parser.CLParser()

        if PARSER.debug:
            dir = realpath(dirname(__file__))
            logging.basicConfig(filename='%s/ham2mon.log'%(dir), \
            level=logging.DEBUG, format='%(asctime)s %(message)s')

        wrapper(main)
    except KeyboardInterrupt:
        pass
    except RuntimeError as error:
        print("")
        print("RuntimeError: SDR hardware not detected or insufficient USB permissions. Try running as root or with --debug option.")
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
        print("")
    except BaseException as error:
        print("")
        print(f'Unexpected: {error=}, {type(error)=}')
        logging.debug(traceback.format_exc())
        print("")

    finally:
        # --- Cleanup on exit ---
        echo()
        nocbreak()
        endwin()
