import re
import sys
from queue import Queue
from threading import Thread
from time import time
from serial_device import SerialDevice

class UnitConversions:
    """
    A utility class for performing unit conversions used in the PowerManager.
    """
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
    def __init__(self, port_device, baud_rate=3686400) -> None:
        """
        Initializes the LPM01A device with the given port and baud rate.
        """
        self.serial_comm = SerialDevice(port_device, baud_rate)

        # Initialize attributes
        self.data_storage = []  
        self.uc = UnitConversions()
        self.mode = None
        self.board_timestamp_ms = 0
        self.capture_start_us = 0
        self.num_of_captured_values = 0
        self.last_print_timestamp_ms = 0
        self.board_buffer_usage_percentage = 0
        self.sum_current_values_ua = 0
        self.number_of_current_values = 0

        # Clear metrics_log.txt
        # Initialize queue for storing power readings
        self._data_queue = Queue()

        self.__enter__()

    def __enter__(self):
        """ Ensures logging starts for voltage, power, and amperage when entering context. """
        self.init_device(mode="ascii", voltage=3300, freq=1000, duration=0)
        self.start_capture()  # Ensure data capture starts
        self.start()  # Start background logging
        return self

    def _read_and_parse_ascii(self) -> None:
        """
        Reads and parses the data from the LPM01A device in ASCII mode.
        Stores parsed data in the queue instead of writing to a file.
        """
        # Record the capture start time
        self.capture_start_us = int(self.uc.s_to_us(time()))

        while True:
            response = self.serial_comm.receive_data()
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

            if "-" in response:
                exponent_sign = "-"
            elif "+" in response:
                exponent_sign = "+"

            try:
                split_response = response.split("-")
                current = int(split_response[0][1:] if split_response[0][0] == '\x00' else split_response[0])
                exponent = int(split_response[1])

                if exponent_sign == "+":
                    current = current * pow(10, exponent)
                else:
                    current = current * pow(10, -exponent)

                current = round(current, 4)  # Keep the current in Amps
                voltage = 3.3  # Assuming a fixed voltage of 3.3V
                power = round(current * voltage, 4)  # Power in Watts
                relative_timestamp_us = (
                    int(self.uc.s_to_us(time())) - self.capture_start_us
                )  # Calculate relative timestamp in microseconds

                # Store parsed data in queue instead of writing to a file
                self._data_queue.put((relative_timestamp_us, current, voltage, power))

            except:
                continue  # Suppress errors silently
    def init_device(
        self,
        mode: str = "ascii",
        voltage: int = 3300,
        freq: int = 5000,
        duration: int = 0,
    ) -> None:
        """
        Initializes the LPM01A device with the given mode, voltage, frequency, and duration.

        Args:
            mode (str): The mode for the LPM01A device. Currently only supports "ascii".
            voltage (int): The voltage for the LPM01A device.
            freq (int): The frequency for the LPM01A device.
            duration (int): The duration for the LPM01A device.
        """

        self.mode = mode
        self.send_command_wait_for_response("htc")

        if self.mode == "ascii":
            self.send_command_wait_for_response(f"format ascii_dec")
        else:
            raise NotImplementedError
            self.send_command_wait_for_response(f"format bin_hexa")

        self.send_command_wait_for_response(f"volt {voltage}m")
        self.send_command_wait_for_response(f"freq {freq}")
        self.configure_trigger('inf', 0, 'd7')  # Set D7 as the trigger source
        # Confirm trigger source
        # Verify the trigger source
        response = self.send_command_wait_for_response("trigsrc?", timeout_s=2)

        print(f"DEBUG: Trigger source response: {response}")  # 🔍 Debug output

        if response != "trigsrc d7":
            print(f"ERROR: D7 trigger was not properly set. Got response: {response}")
        else:
            print("SUCCESS: D7 trigger source confirmed!")



    def configure_trigger(self, acquisition_time, trigger_delay, trigger_source):
        """
        Configures the trigger settings for acquisition.

        Args:
            acquisition_time (str): Duration of acquisition ('inf' for infinite).
            trigger_delay (int): Delay before acquisition starts (in seconds).
            trigger_source (str): Source of trigger ('d7' for external D7 trigger).

        Returns:
            None
        """
        self.send_command_wait_for_response(f"acqtime {acquisition_time}",
                        expected_response="ack",
                        timeout_s=2)
        self.send_command_wait_for_response(f"trigdelay {trigger_delay}",
                        expected_response="ack",
                        timeout_s=2)
        self.send_command_wait_for_response(f"trigsrc {trigger_source}",
                        expected_response="ack",
                        timeout_s=2)

    def get_results(self):
        """
        Retrieves power data stored in self._data_queue.

        Returns:
            Generator: Yields data from self._data_queue.
        """
        while not self._data_queue.empty():
            yield self._data_queue.get()

    def start(self) -> None:
        """
        Starts the _read_and_parse_ascii method in a background thread.
        """
        thread = Thread(target=self._read_and_parse_ascii, daemon=True)
        thread.start()

    def send_command_wait_for_response(
        self, command: str, expected_response: str = None, timeout_s: int = 5
    ) -> str:
        """
        Sends a command to the LPM01A device and waits for a response.

        Returns:
            str: The actual response from the device (instead of True/False).
        """
        tick_start = time()
        self.serial_comm.send_data(command)
        while time() - tick_start < timeout_s:
            response = self.serial_comm.receive_data()
            if response == "":
                continue

            print(f"DEBUG: Received response for {command}: {response}")  # 🔍 Debugging

            if expected_response:
                if response == expected_response:
                    return response  # Return actual response instead of True/False
                else:
                    return response  # Return response so we can debug

            response = response.split("PowerShield > ack ")
            try:
                if response[1] == command:
                    return response[1]  # Return command confirmation
            except IndexError:
                return response[0]  # Return whatever we got for debugging

        return "TIMEOUT"  # Indicate that we timed out



    def start_capture(self) -> None:
        """
        Starts the capture of the LPM01A device.
        """
        self.send_command_wait_for_response("start")

    def stop(self) -> None:
        """
        Stops the capture of the LPM01A device and prints the logged data.
        """
        # Stop the capture
        self.send_command_wait_for_response(
            "stop", expected_response="PowerShield > Acquisition completed"
        )
        self.send_command_wait_for_response("hrc")

        # Print the logged data
        print("Total raw data logged:")
        for entry in self.data_storage:
            print(f"Current (uA): {entry['current_uA']}, "
                f"Timestamp (us): {entry['local_timestamp_us']}, "
                f"Board Timestamp (ms): {entry['board_timestamp_ms']}")


    def deinit_capture(self) -> None:
        """
        Deinitializes the capture of the LPM01A device.
        """
        # Optionally save collected data to a file
        with open("captured_data.txt", "w") as file:
            file.write("Current (uA),rx timestamp (us),board timestamps (ms)\n")
            for entry in self.data_storage:
                file.write(
                    f"{entry['current_uA']},{entry['local_timestamp_us']},{entry['board_timestamp_ms']}\n"
                )
        self.serial_comm.close_serial()

    def read_and_parse_data(self) -> None:
        """
        Reads and parses the data from the LPM01A device.
        """
        self.capture_start_us = int(self.uc.s_to_us(time()))
        if self.mode == "ascii":
            self._read_and_parse_ascii()
        else:
            raise NotImplementedError
    def getAmperage(self, start_time, end_time):
        """
        Calculates the average current (amperage) for the entries in the data storage
        between the specified start and end timestamps.

        Args:
            start_time (int): The start timestamp (in microseconds).
            end_time (int): The end timestamp (in microseconds).

        Returns:
            float: The average current in microamperes (uA) between the start and end times.
                Returns 0 if no data points are within the specified range.
        """
        # Filter the data within the time range
        relevant_data = [
            entry['current_uA'] 
            for entry in self.data_storage 
            if start_time <= entry['local_timestamp_us'] <= end_time
        ]

        # Check if there is any relevant data
        if not relevant_data:
            print("No data points found within the specified time range.")
            return 0.0

        # Calculate the average current
        average_current = sum(relevant_data) / len(relevant_data)
        return average_current