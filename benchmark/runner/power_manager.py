import re
import sys
from queue import Queue
from threading import Thread
from time import time
from serial_device import SerialDevice

class UnitConversions:
    """A utility class for performing unit conversions used in the PowerManager."""

    @staticmethod
    def A_to_uA(current_in_A: int) -> float:
        return current_in_A * 1e6  # 1 A = 1,000,000 µA

    @staticmethod
    def s_to_us(seconds: float) -> int:
        return int(seconds * 1e6)  # 1 second = 1,000,000 microseconds

    @staticmethod
    def us_to_ms(microseconds: int) -> int:
        return microseconds // 1000  # 1 millisecond = 1,000 milliseconds

class PowerManager:
    def __init__(self, port_device, baud_rate=3864000, print_info_every_ms: int = 10_000) -> None:
        # Use SerialDevice instead of SerialCommunication
        self._port = SerialDevice(port_device, baud_rate, "ack|error", "\r\n")
        
        # Initialize attributes
        self.data_storage = []  # Local storage for captured data
        self.uc = UnitConversions()
        self.print_info_every_ms = print_info_every_ms
        self.mode = None
        self.board_timestamp_ms = 0
        self.capture_start_us = 0
        self.num_of_captured_values = 0
        self.last_print_timestamp_ms = 0
        self.board_buffer_usage_percentage = 0
        self.sum_current_values_ua = 0
        self.number_of_current_values = 0

        # Clear metrics_log.txt
        with open("metrics_log.txt", "w") as log_file:
            log_file.write("Timestamp, Amps (A), Voltage (V), Power (W)\n")

    def _read_and_parse_ascii(self) -> None:
        """
        Reads and parses the data from the LPM01A device in ASCII mode.
        Writes the parsed data to the `metrics_log.txt` file in real-time with timestamps starting at 0.
        """
        self.capture_start_us = int(self.uc.s_to_us(time()))
        
        with open("metrics_log.txt", "a") as log_file:
            while True:
                response = self._port.read_line()  # Use SerialDevice read method
                if not response:
                    continue

                if "TimeStamp:" in response:
                    try:
                        match = re.search(r"TimeStamp: (\d+)s (\d+)ms, buff (\d+)%", response)
                        if match:
                            self.board_timestamp_ms = (
                                int(match.group(2)) + int(match.group(1)) * 1000
                            )
                            self.board_buffer_usage_percentage = int(match.group(3))
                    except:
                        continue  # Suppress errors silently
                    continue

                try:
                    split_response = response.split("-")
                    current = int(split_response[0])
                    exponent = int(split_response[1])
                    current = current * pow(10, -exponent)  # Convert using exponent
                    current = round(current, 4)  # Keep the current in Amps
                    voltage = 3.3  # Assuming a fixed voltage of 3.3V
                    power = round(current * voltage, 4)  # Power in Watts
                    relative_timestamp_us = (
                        int(self.uc.s_to_us(time())) - self.capture_start_us
                    )  # Calculate relative timestamp in microseconds

                    # Write to metrics_log.txt
                    log_file.write(f"{relative_timestamp_us}, {current}, {voltage}, {power}\n")
                    log_file.flush()
                except:
                    continue

    def start_background_parsing(self) -> None:
        """
        Starts the _read_and_parse_ascii method in a background thread.
        """
        thread = Thread(target=self._read_and_parse_ascii, daemon=True)
        thread.start()

    def send_command_wait_for_response(self, command: str, expected_response: str = None, timeout_s: int = 5) -> bool:
        """
        Sends a command to the LPM01A device and waits for a response.
        """
        self._port.write_line(command)  # Use SerialDevice send method
        tick_start = time()

        while time() - tick_start < timeout_s:
            response = self._port.read_line()
            if response == "":
                continue

            if expected_response:
                return response == expected_response

            if "ack" in response:
                return True

        return False

    def init_device(self, mode: str = "ascii", voltage: int = 3300, freq: int = 5000, duration: int = 0) -> None:
        """
        Initializes the LPM01A device.
        """
        self.mode = mode
        self.send_command_wait_for_response("htc")

        if self.mode == "ascii":
            self.send_command_wait_for_response(f"format ascii_dec")
        else:
            raise NotImplementedError

        self.send_command_wait_for_response(f"volt {voltage}m")
        self.send_command_wait_for_response(f"freq {freq}")
        self.send_command_wait_for_response(f"acqtime {duration}")

    def start_capture(self) -> None:
        """
        Starts the capture of the LPM01A device.
        """
        print(f"Starting capture, printing info every {self.print_info_every_ms} ms")
        self.send_command_wait_for_response("start")

    def stop_capture(self) -> None:
        """
        Stops the capture of the LPM01A device.
        """
        self.send_command_wait_for_response("stop", expected_response="PowerShield > Acquisition completed")
        self.send_command_wait_for_response("hrc")

    def deinit_capture(self) -> None:
        """
        Deinitializes the capture of the LPM01A device.
        """
        with open("captured_data.txt", "w") as file:
            file.write("Current (uA),rx timestamp (us),board timestamps (ms)\n")
            for entry in self.data_storage:
                file.write(f"{entry['current_uA']},{entry['local_timestamp_us']},{entry['board_timestamp_ms']}\n")

    def read_and_parse_data(self) -> None:
        """
        Reads and parses the data from the LPM01A device.
        """
        self.capture_start_us = int(self.uc.s_to_us(time()))
        self._read_and_parse_ascii()

    def getAmperage(self, start_time, end_time):
        """
        Calculates the average current (amperage) for the entries in the data storage.
        """
        relevant_data = [
            entry['current_uA']
            for entry in self.data_storage
            if start_time <= entry['local_timestamp_us'] <= end_time
        ]

        if not relevant_data:
            print("No data points found within the specified time range.")
            return 0.0

        return sum(relevant_data) / len(relevant_data)
