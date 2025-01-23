import re
import sys
from queue import Queue
from threading import Thread
from time import time
from SerialCommunication import SerialCommunication

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
        return microseconds // 1000  # 1 millisecond = 1,000 microseconds

class PowerManager:
    def __init__(
        self, port: str, baud_rate: int, print_info_every_ms: int = 10_000
    ) -> None:
        """
        Initializes the LPM01A device with the given port and baud rate.

        Args:
            port (str): The port where the LPM01A device is connected.
            baud_rate (int): The baud rate for the serial communication.
            print_info_every_ms (int): The interval in ms to print the info.
        """
        self.serial_comm = SerialCommunication(port, baud_rate)
        self.serial_comm.open_serial()

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

    def _read_and_parse_ascii(self) -> None:
        """
        Reads and parses the data from the LPM01A device in ASCII mode.
        """
        while True:
            response = self.serial_comm.receive_data()
            if not response:
                continue

            if "TimeStamp:" in response:
                try:
                    match = re.search(
                        "TimeStamp: (\d+)s (\d+)ms, buff (\d+)%", response
                    )
                    if match:
                        self.board_timestamp_ms = (
                            int(match.group(2)) + int(match.group(1)) * 1000
                        )
                        self.board_buffer_usage_percentage = int(match.group(3))

                except:
                    print(f"Error parsing timestamp: {response}")
                continue

            if "-" in response:
                exponent_sign = "-"
            elif "+" in response:
                exponent_sign = "+"

            try:
                split_response = response.split("-")
                try:
                    current = int(split_response[0])  # Extract the raw current value
                except ValueError:
                    # When the TimeStamp is received,
                    # the next current values has \x00 in the beginning so I need to strip it
                    current = int(split_response[0][1:])

                exponent = int(split_response[1])  # Extract the exponent value

                # Apply the exponent with the correct sign
                if exponent_sign == "+":
                    current = current * pow(10, exponent)
                else:
                    current = current * pow(10, ((-1) * exponent))

                current = round(self.uc.A_to_uA(current), 4)

                local_timestamp_us = (
                    int(self.uc.s_to_us(time())) - self.capture_start_us
                )
                
                # Store data locally
                self.data_storage.append(
                    {
                        "current_uA": current,
                        "local_timestamp_us": local_timestamp_us,
                        "board_timestamp_ms": self.board_timestamp_ms
                    }
                )

                self.num_of_captured_values += 1

                self.sum_current_values_ua += current
                self.number_of_current_values += 1

                if (
                    self.uc.us_to_ms(local_timestamp_us) - self.last_print_timestamp_ms
                    > self.print_info_every_ms
                ):
                    average_current = (
                        self.sum_current_values_ua / self.number_of_current_values
                    )
                    average_current = round(average_current, 4)
                    self.sum_current_values_ua = 0
                    self.number_of_current_values = 0
                    print(
                        f"Average current for previous {self.print_info_every_ms} ms: {average_current} uA\n"
                        f"Local timestamp: {self.uc.us_to_ms(local_timestamp_us)} ms\n"
                        f"Num of received values: {self.num_of_captured_values}\n"
                        f"LPM01A buffer usage: {self.board_buffer_usage_percentage}%\n"
                    )
                    self.last_print_timestamp_ms = self.uc.us_to_ms(local_timestamp_us)
            except:
                print(f"Error parsing data: {response}")

    def send_command_wait_for_response(
        self, command: str, expected_response: str = None, timeout_s: int = 5
    ) -> bool:
        """
        Sends a command to the LPM01A device and waits for a response.

        Args:
            command (str): The command to send to the LPM01A device.
            expected_response (str): The expected response from the LPM01A device.
            timeout_s (int): The timeout in seconds to wait for a response.

        Returns:
            bool: True if the command was successful, False otherwise.
        """
        tick_start = time()
        self.serial_comm.send_data(command)
        while time() - tick_start < timeout_s:
            response = self.serial_comm.receive_data()
            if response == "":
                continue

            if expected_response:
                if response == expected_response:
                    return True
                else:
                    return False

            response = response.split("PowerShield > ack ")
            try:
                if response[1] == command:
                    return True
            except IndexError:
                return False

        return False

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
        self.send_command_wait_for_response(
            "stop", expected_response="PowerShield > Acquisition completed"
        )
        self.send_command_wait_for_response("hrc")

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
