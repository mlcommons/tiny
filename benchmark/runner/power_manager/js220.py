import time

import joulescope
from joulescope.v1.stream_buffer import StreamBuffer

from power_manager import PowerManager

# There's a good example here of how to use the pyjoulescope API:
# https://github.com/jetperch/pyjoulescope_examples/blob/main/bin/read_by_callback.py

# Also, the API is well documented here:
# https://joulescope.readthedocs.io/en/latest/api/device.html

class JS220PowerManager(PowerManager):

    def __init__(self):

        # create the device
        self.device =joulescope.scan_require_one(config="auto")

    @property
    def sampling_rate(self) -> float:
        self.device.output_sampling_frequency

    @sampling_rate.setter
    def sampling_rate(self, value: float):
        self.device.output_sampling_frequency = value

    def is_running(self):
        pass

    def start(self):
        self.device.start()

    def stop(self):
        self.device.stop()

    def reset(self): pass

    def wait_for_trigger_start(self) -> bool:
        t_start = time.time()
        while self.device._query_gpi_value() == 0:
            if time.time() - t_start > self.timeout:
                raise TimeoutError(f"Timeout waiting for trigger start ({self.timeout} seconds)")
        return True

    def wait_for_trigger_stop(self, timeout: float = 1.0) -> bool:
        t_start = time.time()
        while self.device._query_gpi_value() == 1:
            if time.time() - t_start > timeout:
                raise TimeoutError(f"Timeout waiting for trigger stop ({self.timeout} seconds)")
        return True



