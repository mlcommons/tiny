import time
from joulescope import scan


class JS220PortWrapper:
    def __init__(self, device):
        self._device = device

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self

    def close(self):
        return self._device.close()

    def write_line(self, line):
        pass

    def read_line(self, timeout=None):
        return None


class JoulescopeCommands:
    def __init__(self, manager, js_device):
        self.m = manager
        self._device = js_device

        # ✅ Configure and start streaming
        self._device.parameter_set("source", "raw")
        self._device.parameter_set("sensor_power", "on")
        self._device.parameter_set("sampling_frequency", 1000)
        self._device.parameter_set("io_voltage", "3.3V")
        self._device.parameter_set("trigger_source", "gpi0")
        self._device.parameter_set("current_lsb", "gpi0")

        try:
            self._device.start()
        except Exception as e:
            print(f"[JS220] Failed to start device: {e}")

    def get_port(self):
        return JS220PortWrapper(self._device)

    def setup(self):
        pass  # Setup handled in __init__

    def tear_down(self):
        print("[JS220] Shutting down stream...")
        self.m._running = False
        try:
            self._device.stop()
        except Exception:
            pass
        try:
            self._device.close()
        except Exception:
            pass
        print("[JS220] Shutdown complete.")

    def read_loop(self):
        sb = self._device.stream_buffer
        last_sample_id = None
        last_gpi = None

        print("[JS220] Stream reading started (raw)...")
        while self.m._running:
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
                last_sample_id = end_id
                continue

            try:
                data = sb.samples_get(last_sample_id, end_id, fields=["current", "voltage", "current_lsb"])
            except ValueError:
                last_sample_id = None
                continue

            t0 = time.time()
            current = data["signals"]["current"]["value"]
            voltage = data["signals"]["voltage"]["value"]
            gpi0_vals = data["signals"]["current_lsb"]["value"]
            count = min(len(current), len(voltage), len(gpi0_vals))
            last_sample_id = end_id

            for i in range(count):
                # Always put sample
                self.m._data_queue.put([t0, current[i], voltage[i]])

                # Edge detection logic
                gpi = int(gpi0_vals[i] > 0)
                if last_gpi is None:
                    last_gpi = gpi
                    continue

                if gpi != last_gpi:
                    if last_gpi == 1 and gpi == 0:  # Falling edge only
                        # Queue TimeStamp line
                        timestamp_line = f"TimeStamp: >000s {int((t0 % 1) * 1000):03}ms, buff 00%"
                        event_line = f"event {int(t0 * 1000)} ris"
                        print("[GPI0] Falling edge detected!")
                        self.m._data_queue.put(timestamp_line)
                        self.m._data_queue.put(event_line)
                    last_gpi = gpi
        print("[JS220] Stream reading loop exited.")


    def power_on(self):
        return True

    def power_off(self):
        return True

    def start(self):
        return True

    def stop(self):
        return True

    def get_board_id(self):
        return self._device.device_path()

    def get_version(self):
        try:
            return self._device.info().get("fw")
        except Exception:
            return None

    def get_status(self):
        return None

    def set_lcd(self, *args):
        return [None, None]


"""
################################################################################
#                               Joulescope Commands                            #
################################################################################
# Parameter: sensor_power
#   Options:
#     [0] ('off', 0, [])
#     [1] ('on', 1, [])

# Parameter: source
#   Options:
#     [0] ('off', 0, [])
#     [1] ('raw', 192, ['on'])
#     [2] ('pattern_usb', 9, [])
#     [3] ('pattern_control', 10, [])
#     [4] ('pattern_sensor', 175, [])

# Parameter: i_range
#   Options:
#     [0] ('auto', 128, ['on'])
#     [1] ('10 A', 1, ['0', 0])
#     [2] ('2 A', 2, ['1', 1])
#     [3] ('180 mA', 4, ['2', 2])
#     [4] ('18 mA', 8, ['3', 3])
#     [5] ('1.8 mA', 16, ['4', 4])
#     [6] ('180 µA', 32, ['5', 5])
#     [7] ('18 µA', 64, ['6', 6])
#     [8] ('off', 0, [])

# Parameter: v_range
#   Options:
#     [0] ('15V', 0, ['low', 0])
#     [1] ('5V', 1, ['high', 1])

# Parameter: ovr_to_lsb
#   Options:
#     [0] ('off', 0, [])
#     [1] ('on', 1, [])

# Parameter: trigger_source
#   Options:
#     [0] ('auto', 0, [])
#     [1] ('gpi0', 2, [])
#     [2] ('gpi1', 3, [])

# Parameter: io_voltage
#   Options:
#     [0] ('1.8V', 1800, [])
#     [1] ('2.1V', 2100, [])
#     [2] ('2.5V', 2500, [])
#     [3] ('2.7V', 2700, [])
#     [4] ('3.0V', 3000, [])
#     [5] ('3.3V', 3300, [])
#     [6] ('3.6V', 3600, [])
#     [7] ('5.0V', 5000, [])

# Parameter: gpo0
#   Options:
#     [0] ('0', 0, [0])
#     [1] ('1', 1, [1])

# Parameter: gpo1
#   Options:
#     [0] ('0', 0, [0])
#     [1] ('1', 1, [1])

# Parameter: current_lsb
#   Options:
#     [0] ('normal', 0, [])
#     [1] ('gpi0', 2, [])
#     [2] ('gpi1', 3, [])

# Parameter: voltage_lsb
#   Options:
#     [0] ('normal', 0, [])
#     [1] ('gpi0', 2, [])
#     [2] ('gpi1', 3, [])

# Parameter: control_test_mode
#   Options:
#     [0] ('normal', 3, [])
#     [1] ('usb', 129, [])
#     [2] ('fpga', 130, [])
#     [3] ('both', 131, [])

# Parameter: transfer_length
#   Options:
#     [0] ('1', 1, [])
#     [1] ('2', 2, [])
#     [2] ('4', 4, [])
#     [3] ('8', 8, [])
#     [4] ('16', 16, [])
#     [5] ('32', 32, [])
#     [6] ('64', 64, [])
#     [7] ('128', 128, [])
#     [8] ('256', 256, [])

# Parameter: transfer_outstanding
#   Options:
#     [0] ('1', 1, [])
#     [1] ('2', 2, [])
#     [2] ('4', 4, [])
#     [3] ('8', 8, [])

# Parameter: current_ranging
#   Current Value: interp_1_n_1

# Parameter: current_ranging_type
#   Options:
#     [0] ('off', 'off', [])
#     [1] ('mean', 'mean', [])
#     [2] ('interp', 'interp', ['interpolate'])
#     [3] ('NaN', 'nan', ['nan'])

# Parameter: current_ranging_samples_pre
#   Options:
#     [0] ('0', 0, [0])
#     [1] ('1', 1, [1])
#     [2] ('2', 2, [2])
#     [3] ('3', 3, [3])
#     [4] ('4', 4, [4])
#     [5] ('5', 5, [5])
#     [6] ('6', 6, [6])
#     [7] ('7', 7, [7])
#     [8] ('8', 8, [8])

# Parameter: current_ranging_samples_window
#   Options:
#     [0] ('m', 'm', [])
#     [1] ('n', 'n', [])
#     [2] ('0', 0, [0])
#     [3] ('1', 1, [1])
#     [4] ('2', 2, [2])
#     [5] ('3', 3, [3])
#     [6] ('4', 4, [4])
#     [7] ('5', 5, [5])
#     [8] ('6', 6, [6])
#     [9] ('7', 7, [7])
#     [10] ('8', 8, [8])
#     [11] ('9', 9, [9])
#     [12] ('10', 10, [10])
#     [13] ('11', 11, [11])
#     [14] ('12', 12, [12])

# Parameter: current_ranging_samples_post
#   Options:
#     [0] ('0', 0, [0])
#     [1] ('1', 1, [1])
#     [2] ('2', 2, [2])
#     [3] ('3', 3, [3])
#     [4] ('4', 4, [4])
#     [5] ('5', 5, [5])
#     [6] ('6', 6, [6])
#     [7] ('7', 7, [7])
#     [8] ('8', 8, [8])

# Parameter: buffer_duration
#   Options:
#     [0] ('15 seconds', 15, [15])
#     [1] ('30 seconds', 30, [30])
#     [2] ('1 minute', 60, [60])
#     [3] ('2 minutes', 120, [120])
#     [4] ('5 minutes', 300, [300])
#     [5] ('10 minutes', 600, [600])
#     [6] ('20 minutes', 1200, [1200])
#     [7] ('1 hour', 3600, [3600])
#     [8] ('2 hours', 7200, [7200])
#     [9] ('5 hours', 18000, [18000])
#     [10] ('10 hours', 36000, [36000])
#     [11] ('1 day', 86400, [86400])

# Parameter: reduction_frequency
#   Options:
#     [0] ('100 Hz', 100, [100])
#     [1] ('50 Hz', 50, [50])
#     [2] ('20 Hz', 20, [20])
#     [3] ('10 Hz', 10, [10])
#     [4] ('5 Hz', 5, [5])
#     [5] ('2 Hz', 2, [2])
#     [6] ('1 Hz', 1, [1])

# Parameter: sampling_frequency
#   Options:
#     [0] ('2 MHz', 2000000, [2000000, 'auto', None, 'default'])
#     [1] ('1 MHz', 1000000, [1000000])
#     [2] ('500 kHz', 500000, [500000])
#     [3] ('200 kHz', 200000, [200000])
#     [4] ('100 kHz', 100000, [100000])
#     [5] ('50 kHz', 50000, [50000])
#     [6] ('20 kHz', 20000, [20000])
#     [7] ('10 kHz', 10000, [10000])
#     [8] ('5 kHz', 5000, [5000])
#     [9] ('2 kHz', 2000, [2000])
#     [10] ('1 kHz', 1000, [1000])
#     [11] ('500 Hz', 500, [500])
#     [12] ('200 Hz', 200, [200])
#     [13] ('100 Hz', 100, [100])
#     [14] ('50 Hz', 50, [50])
#     [15] ('20 Hz', 20, [20])
#     [16] ('10 Hz', 10, [10])
"""