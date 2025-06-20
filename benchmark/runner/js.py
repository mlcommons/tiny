#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import signal
import argparse
from joulescope import scan


def get_parser():
    p = argparse.ArgumentParser(description="Dump every JS220 sample")
    p.add_argument("--out", "-o", help="Optional output file")
    return p


def main():
    args = get_parser().parse_args()
    stop_flag = False

    def signal_handler(*_):
        nonlocal stop_flag
        stop_flag = True

    signal.signal(signal.SIGINT, signal_handler)

    devices = scan(config="off")
    if not devices:
        print("❌ No Joulescope devices found.")
        return

    device = devices[0]
    device.open()
    device.parameter_set("source", "raw")
    device.parameter_set("sensor_power", "on")
    device.parameter_set("sampling_frequency", 2000000)
    device.parameter_set("trigger_source", "gpi0")
    device.start()  # ✅ Start streaming

    sb = device.stream_buffer
    last_sample_id = None

    print("✅ Dumping raw samples (press Ctrl+C to stop)...")
    if args.out:
        f = open(args.out, "w")
        f.write("timestamp,current_A,voltage_V\n")
    else:
        f = None

    try:
        while not stop_flag:
            sample_id_range = sb.sample_id_range
            if sample_id_range is None:
                time.sleep(0.001)
                continue

            start_id, end_id = sample_id_range
            if last_sample_id is None:
                last_sample_id = start_id

            if end_id <= last_sample_id:
                time.sleep(0.001)
                continue

            if end_id < last_sample_id:
                # Stream buffer reset
                last_sample_id = end_id
                continue

            try:
                data = sb.samples_get(last_sample_id, end_id, fields=["current", "voltage"])
            except ValueError:
                last_sample_id = None
                continue

            t0 = time.time()
            current = data["signals"]["current"]["value"]
            voltage = data["signals"]["voltage"]["value"]
            count = min(len(current), len(voltage))
            last_sample_id = end_id

            for i in range(count):
                timestamp = t0
                line = f"{timestamp:.6f},{current[i]:.9f},{voltage[i]:.6f}"
                print(line)
                if f:
                    f.write(line + "\n")

    finally:
        device.stop()
        device.close()
        if f:
            f.close()
        print("📁 Complete. Device closed.")


if __name__ == "__main__":
    main()
