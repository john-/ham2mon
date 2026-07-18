'''
Log channel activity in various formats and provide channel activity to scanner.
'''
import logging
import datetime
from frequency_manager import ChannelMessage
from abc import ABC
from dataclasses import dataclass, asdict
from importlib import import_module
import asyncio
from typing import Callable

logger = logging.getLogger(f"ham2mon.{__name__}")

@dataclass(kw_only=True)
class ActivityParams:
    '''
    Holds channel activity logging command line options provided by the user
    '''
    type: str
    dest: str
    interval: int

class ActivityLogger(ABC):
    '''
    Base class for all loggers.  Also notify scanner of activity.
    '''
    def __init__(self, params: ActivityParams,
                 get_ctcss: Callable[[int], float | None] | None = None) -> None:
        logger.debug(f'Creating {self.__class__.__name__} channel logger')
        self.interval: int = 0  # overridden by child classes
        self.log_task: dict[int, asyncio.Task] = {}  # activity logging tasks are channel specific
        self.params = params
        self.get_ctcss = get_ctcss  # optional callback: bb_freq -> matched ctcss tone or None

    async def log(self, msg: ChannelMessage | None) -> None:
        '''
        Abstract method to log an event.  Also provide message to scanner
        with receiver details (files created and/or classified)

        Overridden in each child class for specific loggers
        '''
        if msg is None:
            return

    @staticmethod
    def get_logger(params: ActivityParams,
                   get_ctcss: Callable[[int], float | None] | None = None) -> 'ActivityLogger':
        '''
        Factory to generate a class instance based on command line options
        '''
        if params.type == 'fixed-field':
            return FixedField(params, get_ctcss=get_ctcss)
        elif params.type == 'json-server':
            return JsonToServer(params, get_ctcss=get_ctcss)
        else:
            return NoOp(params, get_ctcss=get_ctcss)

    def handle_channel_state(self, msg: ChannelMessage) -> None:
        '''
        Use on/off events to start/stop activity timer
        '''
        if self.interval == 0:
            return

        channel = msg.channel
        if msg.state == 'on':
            # Cancel any orphaned task for this channel before starting a new one
            existing = self.log_task.get(channel)
            if existing and not existing.done():
                existing.cancel()
            # start reoccurring task to log that channel is active
            self.log_task[channel] = asyncio.create_task(self.log_active(msg))
        elif msg.state == 'off':
            # stop the reoccurring task
            task = self.log_task.get(channel)
            if task:
                was_cancelled = task.cancel()
                if not was_cancelled:
                    logger.error('Could not cancel logging task')
                try:
                    del self.log_task[channel]
                except KeyError:
                    pass

    async def log_active(self, msg: ChannelMessage) -> None:
        '''
        While the channel is active log at an interval.
        Reads matched_ctcss live from the demodulator via the get_ctcss
        callback (if provided) so heartbeats reflect the settled tone rather
        than the None that was present at channel-open time.
        '''
        while True:
            await asyncio.sleep(self.interval)
            live_ctcss = self.get_ctcss(msg.bb) if self.get_ctcss is not None else None
            await self.log(ChannelMessage(state='act',
                                    rf=msg.rf,
                                    bb=msg.bb,
                                    channel=msg.channel,
                                    matched_ctcss=live_ctcss))

class NoOp(ActivityLogger):
    '''
    Logger that ignores all events
    '''
    def __init__(self, params: ActivityParams,
                 get_ctcss: Callable[[int], float | None] | None = None) -> None:
        super().__init__(params, get_ctcss=get_ctcss)

        self.interval: int = 0

    async def log(self, msg: ChannelMessage | None) -> None:
        if msg is None:
            return

        await super().log(msg)

class FixedField(ActivityLogger):
    '''
    Send channel events to a file with fixed field length records
    '''
    def __init__(self, params,
                 get_ctcss: Callable[[int], float | None] | None = None) -> None:
        super().__init__(params, get_ctcss=get_ctcss)

        self.file_name = params.dest
        self.interval = params.interval

    async def log(self, msg: ChannelMessage | None) -> None:
        if msg is None:
            return

        await super().log(msg)

        now = datetime.datetime.now()
        with open(self.file_name, 'a') as file:
            text = (f'{now.strftime("%Y-%m-%d, %H:%M:%S.%f")}: {msg.state:<4}{msg.rf:<10}'
                    f'{msg.channel:<2}{msg.priority if msg.priority else "":<2}'
                    f'{msg.classification if msg.classification else "":<2}'
                    f'{f"{msg.matched_ctcss:.1f}" if msg.matched_ctcss else "":<7}'
                    f'{msg.file if msg.file else "":<50}\n'
                    )
            file.write(text)

        self.handle_channel_state(msg)

class JsonToServer(ActivityLogger):
    '''
    Send channels events as json messages to  a remote server
    '''
    def __init__(self, params,
                 get_ctcss: Callable[[int], float | None] | None = None) -> None:
        super().__init__(params, get_ctcss=get_ctcss)

        self.server = params.dest
        self.interval = params.interval

        self.requests = import_module('requests')
        # urllib3 log suppression is configured at application startup in ham2mon.py

    async def log(self, msg: ChannelMessage | None) -> None:
        if msg is None:
            return

        await super().log(msg)

        msg_dict = asdict(msg)
        logger.debug(f'{msg =}')

        try:
            # TODO: open the connection once
            request = self.requests.post(self.server, json=msg_dict)
            request.raise_for_status()
        except self.requests.exceptions.HTTPError as errh:
            logger.error(f'HTTP Error: {errh.args[0]}')
        except self.requests.exceptions.ConnectionError as errc:
            logger.error(f'Connection Error: {errc.args[0]}')
        except self.requests.exceptions.Timeout as errt:
            logger.error(f'Timeout Error: {errt.args[0]}')
        except self.requests.exceptions.RequestException as err:
            logger.error(f'Some kind of Error: {err.args[0]}')

        self.handle_channel_state(msg)
