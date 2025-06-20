import signal
import time
import numpy as np
from joulescope import scan_require_one

stop = False

def handle_sigint(sig, frame):
    global stop
    stop = True

def main():
    global stop
    signal.signal(signal.SIGINT, handle_sigint)

    print("[info] Connecting to Joulescope...")
    device = scan_require_one(config='ignore')

    with device:
        device.parameter_set('sensor_power', 'on')
        device.parameter_set('source', 'raw')
        device.parameter_set('sampling_frequency', 1000)  # 1 kHz
        device.parameter_set('io_voltage', '3.3V')
        device.parameter_set('current_lsb', 'gpi0')  # Route GPI0 into current_lsb

        print("[info] Reading GPI0 (as current_lsb)... Press Ctrl+C to stop.")
        device.start(stop_fn=None)
        last_id = None

        while not stop:
            sb = device.stream_buffer
            start_id, end_id = sb.sample_id_range
            if last_id is None or last_id < start_id:
                last_id = start_id

            if end_id > last_id:
                samples = sb.samples_get(last_id, end_id, fields='current_lsb')
                for val in samples:
                    gpi_val = int(val > 0)
                    print(gpi_val)
                last_id = end_id
            time.sleep(0.1)

        device.stop()
        print("[info] Stream stopped.")

if __name__ == '__main__':
    main()
