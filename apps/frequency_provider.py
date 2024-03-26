'''
Provides a center frequency to Scanner.  If more than one frequency or
frequency range is used than the provided center frequency will step
through all the available frequencies

Initialize with command line parameters (frequencies or frequency ranges) and
timeouts for activity tracking.
'''
from dataclasses import dataclass, field
import logging
import asyncio
import typing

@dataclass(kw_only=True)
class FrequencyRangeParams:
    '''
    Represents a range of frequencies in Hz
    '''
    lower_freq: int
    upper_freq: int

    def __post_init__(self):
        if self.lower_freq >= self.upper_freq:
            raise ValueError('Upper frequency must be larger than lower frequency')

@dataclass(kw_only=True)
class FrequencySingleParams:
    '''
    Represents a single frequency in Hz
    '''
    freq: int

@dataclass(kw_only=True)
class FrequencyGroup:
    '''
    Stores all the frequency information and timeouts provided by the user.
    Also includes the callbacks used to notify the scanner and user interface.
    
    Frequencies are in Hz

    Note: The notify_interface callback is included here but used only
    by the scanner.
    '''
    ranges: list[FrequencyRangeParams] = field(default_factory=list)
    singles: list[FrequencySingleParams] = field(default_factory=list)
    sample_rate: int
    quiet_timeout: float = 10
    active_timeout: float = 16
    notify_scanner: typing.Callable = field(default=lambda: None)
    notify_interface: typing.Callable = field(default=lambda: None)

class FrequencyProvider():
    '''
    Determines the current center frequency and provides it to the scanner
    '''
  
    def __init__(self, params: FrequencyGroup) -> None:
        logging.debug('Creating frequency provider')

        self.center_freq: int
        self.step: int = 0
        self.step_task: asyncio.Task

        self.params = params

        self.steps = self._get_steps()

        if self.not_stepping():
            self.center_freq = self.steps[0]
            return

        # all remaining logic is to handle case where we are stepping
        # through more than one center frequency

        self.step = 0
        self.center_freq = self.steps[self.step]  # start out in the first step

        # kick off the process
        timeout = params.quiet_timeout
        self.step_task = asyncio.create_task(
            self.step_if_no_activity(timeout)
        )

    def not_stepping(self) -> bool:
      '''
      If there is only one step we are not stepping
      through a range
      '''
      if len(self.steps) == 1:
          return True
      else:
          return False

    async def step_if_no_activity(self, timeout: float) -> None:
        '''
        Stay on this center frequency for a bit then move to the next step

        This routine may be cancelled if the provider is informed
        of activity that should stop the advance to the next step.
        '''
        logging.debug(f'starting: {self.step=} {self.center_freq=}')

        await asyncio.sleep(timeout)

        # loop around if at the end of the steps
        if self.step == len(self.steps) - 1:
            self.step = 0
        else:
            self.step += 1

        self.center_freq = self.steps[self.step]
        self.params.notify_scanner()

        timeout = self.params.quiet_timeout
        self.step_task = asyncio.create_task(self.step_if_no_activity(timeout))

    async def interesting_activity(self) -> None:
        '''
        This routine is how the scanner will notify the provider that
        there is activity which will delay the advance to the next step.

        Stay on this center frequency based on the active timeout
        '''
        if self.not_stepping():
            return

        logging.debug('Got something of note, setting the active timer')

        was_cancelled = self.step_task.cancel()
        try:
            await self.step_task
        except asyncio.CancelledError: # This exception reflects that the coroutine addressed to the cancel
            logging.debug("cancelled step task in frequency_provider")

        if not was_cancelled:
            logging.error('Could not cancel logging task in frequency_provider')

        timeout = self.params.active_timeout
        self.step_task = asyncio.create_task(self.step_if_no_activity(timeout))
        

    def _get_steps(self) -> list[int]:
        '''
        Take the frequency singles/ranges and breaks them down into steps. Return a list
        of the center frequencies for each step.
        '''
        # Treat single frequencies as centers
        centers = [single.freq for single in self.params.singles]

        # For ranges, divide it up into a series of center frequencies.
        #
        # This approach defines first and last centers such that
        # the edge of the available hardware bandwidth aligns with the
        # edge of the range.  It then divides the remaining range up equally.
        #
        # Of course there are other approaches.  If no one is ideal than maybe
        # the approach could be user selectable.    
        for a_range in self.params.ranges:

            sample_rate = self.params.sample_rate

            # handle range less than sample rate
            if a_range.upper_freq - a_range.lower_freq <= sample_rate:
                center = a_range.lower_freq + int((a_range.upper_freq - a_range.lower_freq) / 2)
                centers.append(center)
                continue

            # break the range into pieces based on sample rate
            start_at = a_range.lower_freq + int(sample_rate/2)
            end_at = a_range.upper_freq - int(sample_rate/2)
            number_of_moves = int((end_at - start_at) / sample_rate) + 2

            logging.debug(f'{a_range=} {start_at=} {end_at=} {number_of_moves=}')

            distance = int((end_at - start_at) / (number_of_moves - 1))

            logging.debug(f'Range: {start_at}-{end_at} Steps: {number_of_moves} Distance: {distance}, Min time: {number_of_moves*self.params.quiet_timeout}')

            center = start_at
            for _ in range(number_of_moves):
                centers.append(center)
                center += distance

        return centers

async def main() -> None:

    # single frequency
    expected_center_freq = int(float(356E6))
    frequency_params =  FrequencySingleParams(freq=expected_center_freq)
    sample_rate: int = int(float(3E6))
    frequency_provider = FrequencyProvider(FrequencyGroup(singles=[frequency_params], sample_rate=sample_rate))

    center_freq = frequency_provider.center_freq
    if center_freq == expected_center_freq:
        print(f'Got the expected center frequency: {center_freq}')

    # invalid range
    try:
        range = FrequencyRangeParams(lower_freq=int(float(460.0E6)), upper_freq=int(float(450.0E6)))
    except ValueError:
        print('Got the expected error for upper frequency less than lower frequency')

    # small range need to figure out expected results
    range = FrequencyRangeParams(lower_freq=460000000, upper_freq=460000002)
    frequency_provider = FrequencyProvider(FrequencyGroup(ranges=[range], sample_rate=sample_rate))
    steps = frequency_provider.steps
    if len(steps) == 1:
        print(f'Got the expected single step for a small range: {steps}')
    else:
        raise Exception('Small range did not result in a single step')

    # range
    expected_lower_freq: int = int(float(450E6))
    expected_upper_freq: int = int(float(459E6))
    sample_rate = int(float(3E6))
    expected_center_freq = expected_lower_freq + int(sample_rate/2)
    # from range need to create a RangeParams and pass it in as well somehow?
    range = FrequencyRangeParams(lower_freq=expected_lower_freq, upper_freq=expected_upper_freq)
    frequency_provider = FrequencyProvider(FrequencyGroup(ranges=[range], sample_rate=sample_rate))

    if frequency_provider.center_freq == expected_center_freq:
        print(f'Got the expected center frequency: {expected_center_freq}')

if __name__ == '__main__':
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass

