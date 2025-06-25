# power_manager.py

import re
import sys
from queue import Queue
from threading import Thread
from serial_device import SerialDevice
from .power_manager_lpm import LPMCommands
from .power_manager_js220 import JoulescopeCommands


class PowerManager:
    PROMPT = "PowerShield > "

    def __init__(self, device_type, port_device=None, baud_rate=None, voltage=3.3, echo=False, js_device=None, config=None):
        self._device_type = device_type
        self._voltage = voltage
        self._board_id = None
        self._version = None
        self._lcd = [None, None]
        self._in_prograss = False
        self._data_queue = Queue()
        self._message_queue = Queue()
        self._read_thread = None
        self._running = False

        if device_type == "lpm01a":
            self._port = SerialDevice(port_device, baud_rate, "ack|error", "\r\n", echo=echo)
            self._commands = LPMCommands(self, self._port)
        elif device_type == "js220":
            if js_device is None:
                from joulescope import scan_require_one
                js_device = scan_require_one(config="ignore")
            self._commands = JoulescopeCommands(self, js_device=js_device, config=config)
            self._port = self._commands.get_port()
        else:
            raise ValueError(f"Unsupported device type: {device_type}")

        self.__enter__()

    def __enter__(self):
        self._port.__enter__()
        self._start_read_thread()           # ✅ move this up
        self._commands.setup()              # ✅ now it's safe like the old code
        return self

    def __exit__(self, *args):
        self._commands.tear_down()
        self._stop_read_thread()
        self._port.__exit__(*args)

    def _start_read_thread(self):
        self._running = True
        self._read_thread = Thread(target=self._commands.read_loop)
        self._read_thread.start()

    def _stop_read_thread(self):
        self._running = False
        self._read_thread.join()

    def get_results(self):
        while not self._data_queue.empty():
            yield self._data_queue.get()
            
    def should_stop(self):
        return not self._running


    # Pass-throughs
    def power_on(self): return self._commands.power_on()
    def power_off(self): return self._commands.power_off()
    def start(self): return self._commands.start()
    def stop(self): return self._commands.stop()
    def get_status(self): return self._commands.get_status()
    def get_board_id(self): return self._commands.get_board_id()
    def get_version(self): return self._commands.get_version()
    def set_lcd(self, *args): return self._commands.set_lcd(*args)
