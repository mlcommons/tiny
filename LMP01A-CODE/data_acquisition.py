#!/bin/env python3

from src.LPM01A import LPM01A

try:
    lpm = LPM01A(port="COM19", baud_rate=3864000, print_info_every_ms=10_000)
    lpm.init_device(mode="ascii", voltage=3300, freq=1000, duration=0)
    lpm.start_capture()
    lpm.read_and_parse_data()
except KeyboardInterrupt:
    print("KeyboardInterrupt detected. Exiting...")
    lpm.stop_capture()
    lpm.deinit_capture()
    exit(0)
