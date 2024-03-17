'''
Log channel activity in various formats and provide channel activity to scanner.
'''
import logging
import datetime
from h2m_types import ChannelMessage
from abc import ABC
from dataclasses import dataclass, asdict, field
from importlib import import_module
import asyncio
import typing

@dataclass(kw_only=True)
class ChannelLogParams:
    '''
    Holds channel log command line options provided by the user
    '''
    type: str
    target: str
    timeout: int
    notify_scanner: typing.Callable = field(default=lambda: None)

class ChannelLogger(ABC):
    '''
    Base class for all loggers.  Also notify scanner of activity.
    '''
    def __init__(self, params: ChannelLogParams) -> None:
        logging.debug(f'Creating {self.__class__.__name__} channel logger')
        self.timeout: int = 0  # overridden by child classes
        self.log_task: dict[int, asyncio.Task] = {}  # activity logging tasks are channel specific
        self.params = params

    async def log(self, msg: ChannelMessage | None) -> None:
        '''
        Abstract method to log an event.  Also provide message to scanner
        with receiver details (files created and/or classified)

        Overridden in each child class for specific loggers
        '''
        if msg is None:
            return
        await self.params.notify_scanner(msg)

    @staticmethod
    def get_logger(params: ChannelLogParams) -> 'ChannelLogger':
        '''
        Factory to generate a class instance based on command line options
        '''
        if params.type == 'fixed-field':
            return FixedField(params)
        elif params.type == 'json-server':
            return JsonToServer(params)
        elif params.type == 'debug':
            return Debug(params)
        else:
            return NoOp(params)
        
    def handle_channel_state(self, msg: ChannelMessage) -> None:
        '''
        Use on/off events to start/stop activity timer
        '''
        if self.timeout == 0:
            return

        channel = msg.channel
        if msg.state == 'on':
            # start reoccurring task to log that channel is active
            self.log_task[channel] = asyncio.create_task(self.log_active(msg))
        elif msg.state == 'off':
            # stop the reoccurring task
            if self.log_task[channel]:
                was_cancelled = self.log_task[channel].cancel()
                if not was_cancelled:
                    logging.error('Could not cancel logging task')

    async def log_active(self, msg: ChannelMessage) -> None:
        '''
        While the channel is active log at an interval
        '''
        while True:
            await asyncio.sleep(self.timeout)
            await self.log(ChannelMessage(state='act',
                                    frequency=msg.frequency,
                                    channel=msg.channel))

class NoOp(ChannelLogger):
    '''
    Logger that ignores all events
    '''
    def __init__(self, params: ChannelLogParams) -> None:
        super().__init__(params)

        self.timeout: int = 0

    async def log(self, msg: ChannelMessage | None) -> None:
        if msg is None:
            return

        await super().log(msg)

class Debug(ChannelLogger):
    '''
    Send channel events to the debug log (enable with --debug)
    '''
    def __init__(self, params: ChannelLogParams) -> None:
        super().__init__(params)

        self.timeout = params.timeout

    async def log(self, msg: ChannelMessage | None) -> None:
        if msg is None:
            return
        
        await super().log(msg)

        # for this logger we just write to the debug log
        logging.debug(msg)

        self.handle_channel_state(msg)
        
class FixedField(ChannelLogger):
    '''
    Send channel events to a file with fixed field length records
    '''
    def __init__(self, params) -> None:
        super().__init__(params)

        self.file_name = params.target
        self.timeout = params.timeout

    async def log(self, msg: ChannelMessage | None) -> None:
        if msg is None:
            return

        await super().log(msg)
        
        now = datetime.datetime.now()
        with open(self.file_name, 'a') as file:
            file.write(f'{now.strftime("%Y-%m-%d, %H:%M:%S.%f")}: {msg.state:<4}{msg.frequency:<10}{msg.channel:<2}\n')

        self.handle_channel_state(msg)
class JsonToServer(ChannelLogger):
    '''
    Send channels events as json messages to  a remote server
    '''
    def __init__(self, params) -> None:
        super().__init__(params)

        self.server = params.target
        self.timeout = params.timeout

        self.requests = import_module('requests')
        # urllib3 is very chatty.  Uncomment is log event for every connection is needed.
        # remove this if there is a single connection approach
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    async def log(self, msg: ChannelMessage | None) -> None:
        if msg is None:
            return

        await super().log(msg)

        msg_dict = asdict(msg)
        logging.debug(f'{msg =}')

        try:
            # TODO: open the connection once
            request = self.requests.post(self.server, json=msg_dict)
            request.raise_for_status()
        except self.requests.exceptions.HTTPError as errh:
            logging.error(f'HTTP Error: {errh.args[0]}')
        except self.requests.exceptions.ConnectionError as errc:
            logging.error(f'Connection Error: {errc.args[0]}')
        except self.requests.exceptions.Timeout as errt:
            logging.error(f'Timeout Error: {errt.args[0]}')
        except self.requests.exceptions.RequestException as err:
            logging.error(f'Some kind of Error: {err.args[0]}')

        self.handle_channel_state(msg)
