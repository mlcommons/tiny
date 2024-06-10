"""
Questions:
    1. What changeable parameters should the power manager have?
    2. What is the output format?
    3. Do we poll the power manager, or rely on its own timers?
    4. How do we handle dropped samples?
"""

from abc import ABC, abstractmethod
from threading import Thread

class PowerManager(ABC):

    def __init__(self):

        # set the default timeout to 1 second
        self._timeout = 1.0

    """
    Properties
    """

    @property
    def timeout(self) -> float:
        """
        A getter method for the timeout attribute.
        The timeout for waiting for triggers.

        Returns:
            float: The timeout in seconds
        """
        return self._timeout

    @timeout.setter
    def timeout(self, value: float):
        """
        A setter method for the timeout attribute.
        The timeout for waiting for triggers.

        Args:
            value (float): The timeout in seconds
        """

        self._timeout = value

    @abstractmethod
    @property
    def sampling_rate(self) -> float:
        """
        A getter method for the sampling_rate attribute.
        This dictates the rate at which the power trace
        is sampled from the DUT.

        The sampling rate can be accessed as follows:

        ```
        power_manager = PowerManager()
        sampling_rate = power_manager.sampling_rate
        ```

        Returns:
            float: The sampling rate of the power trace,
            in Hz.
        """


    @abstractmethod
    @sampling_rate.setter
    def sampling_rate(self, value: float):
        """
        A setter method for the sampling_rate attribute.
        This dictates the rate at which the power trace
        is sampled from the DUT.

        The sampling rate can be set as follows:

        ```
        power_manager = PowerManager()
        power_manager.sampling_rate = 200
        ```
        """



    """
    Methods
    """

    # TODO: define an enumeration for common status between
    # power measuring devices
    # @abstractmethod
    # def status(self):
    #     """
    #     This method checks if the power manager is currently
    #     collecting power traces from the DUT.

    #     Returns:
    #         bool: True if the power manager is currently
    #         collecting power traces, False otherwise.
    #     """


    @abstractmethod
    def setup(self, *args):
        """
        A method that sets up the power manager.
        This is power measurement device specific, and the
        configuration should be loaded from a file, in a
        format that is relevant to the power measurement.

        Args:
            args: A list of arguments relevant to the
            device setup. Probably best to have a
        """


    @abstractmethod
    def start(self):
        """
        Start reading from the device.

        This should be a non-blocking method, where once
        it has started off the power manager, it should
        return immediately.
        """

    @abstractmethod
    def stop(self):
        """
        Stop reading from the device.

        This should be a blocking method, where it
        immediately ends the device's operation.
        """

    @abstractmethod
    def reset(self):
        """
        Reset the device and clear the sample buffer.
        """

    @abstractmethod
    def wait_for_trigger_start(self) -> bool:
        """
        Method to wait for the start trigger from the DUT.
        Blocks execution until this trigger happens.

        Raises:
            TimeoutError: If the trigger does not occur
            before `self.timeout` seconds after starting,
            this error is triggered.
        """

    @abstractmethod
    def wait_for_trigger_stop(self) -> bool:
        """
        Method to wait for the stop trigger from the DUT.
        Blocks execution until this trigger happens.

        Raises:
            TimeoutError: If the trigger does not occur
            before `self.timeout` seconds after starting,
            this error is triggered.
        """

    @abstractmethod
    def get_results(self) -> list[float]:
        """
        Get the list of power measurement results from
        the most recent execution of the device. This is
        a list of values stored either locally or on the
        device itself. It is expected that these samples
        were uniformly sampled.

        Returns:
            list[float]: A list of instantaneous power
            measurements, in mW.

        Raises:
            ValueError: If there is nothing in the buffer,
            either due to reset or nothing being run.
        """


    def autorun(self) -> list[float]:
        """
        The main method that will collect the power trace.

        1. Wait for the starting trigger, which is sent by
            the DUT.
        2. start collecting power measurements. This runs

        """

        # reset the device
        self.reset()

        # wait for the trigger to start
        self.wait_for_trigger_start()

        # start collecting the power trace
        self.start()

        # wait for the trigger to stop
        self.wait_for_trigger_stop()

        # stop collecting the power trace
        self.stop()

        # return the power trace
        return self.get_results()
