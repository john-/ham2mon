'''Functionality to accept an event from a demodulator channel and format it in various formats

   Created on Fri Mar 8 2024

   @author: john
'''
import logging
import datetime
from h2m_types import ChannelMessage
from abc import ABC
from dataclasses import dataclass, asdict
import json
from importlib import import_module

@dataclass(kw_only=True)
class ChannelLogParams:
    '''Holds channel log command line options provided by the user'''
    type: str
    target: str
    timeout: int

class ChannelLogger(ABC):
    '''Base class for all loggers'''

    def __init__(self) -> None:
        logging.debug(f'creating {self.__class__.__name__} channel logger')
        self.timeout: int = 0

    def log(self, msg: ChannelMessage | None) -> None:
        pass

    @staticmethod
    def get_logger(params: ChannelLogParams):
        '''Factory to generate a class instance based on command line options'''

        logging.debug(f'user wants to create {params}')
        if params.type == 'fixed-field':
            return FixedField(params.target, params.timeout)
        elif params.type == 'json-server':
            return JsonToServer(params.target, params.timeout)
        elif params.type == 'debug':
            return Debug(params.timeout)
        else:
            return NoOp()

class NoOp(ChannelLogger):
    '''Logger that ignores all events'''

    def __init__(self) -> None:
        logging.debug(f'creating {self.__class__.__name__} channel logger')
        self.timeout: int = 0

    def log(self, msg: ChannelMessage | None) -> None:
        '''Concrete classes provide their specific implementation of writing channel events'''
        pass

class Debug(ChannelLogger):
    '''Send channel events to the debug log (enable with --debug)'''

    def __init__(self, timeout: int) -> None:
        logging.debug(f'creating {self.__class__.__name__} channel logger')
        self.timeout = timeout

    def log(self, msg: ChannelMessage | None) -> None:
        if msg is None:
            return
        
        logging.debug(msg)
        
class FixedField(ChannelLogger):
    '''Send channel events to a file with fixed field length records'''

    def __init__(self, file_name: str, timeout: int) -> None:
        logging.debug(f'creating {self.__class__.__name__} channel logger (file: {file_name})')

        self.file_name = file_name
        self.timeout = timeout

    def log(self, msg: ChannelMessage | None) -> None:
        if msg is None:
            return
        
        now = datetime.datetime.now()
        with open(self.file_name, 'a') as file:
            file.write(f'{now.strftime("%Y-%m-%d, %H:%M:%S.%f")}: {msg.state:<4}{msg.frequency:<10}{msg.channel:<2}\n')

class JsonToServer(ChannelLogger):
    '''Send channels events as json messages to remote server'''

    def __init__(self, endpoint: str, timeout: int) -> None:
        logging.debug(f'creating {self.__class__.__name__} channel logger (file: {endpoint})')

        self.server = endpoint
        self.timeout = timeout

        self.requests = import_module('requests')

    def log(self, msg: ChannelMessage | None) -> None:
        if msg is None:
            return

        msg_dict = asdict(msg)
        logging.debug(f'{msg =}')

        try:
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
