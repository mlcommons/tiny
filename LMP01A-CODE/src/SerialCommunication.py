import serial


class SerialCommunication:
    def __init__(self, port: str, baud_rate: int) -> None:
        """Initializes the SerialCommunication with the given port and baud rate.

        Args:
            port (str): The port where the device is connected.
            baud_rate (int): The baud rate for the serial communication.
        """

        self.serial_port = port
        self.baud_rate = baud_rate
        self.ser = None

    def open_serial(self) -> None:
        """Opens the serial communication."""
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            print(
                f"Serial communication established on {self.serial_port} with baud rate {self.baud_rate}"
            )
        except serial.SerialException as e:
            print(f"Error: {e}")
            exit(1)

    def close_serial(self) -> None:
        """Closes the serial communication."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed.")

    def send_data(self, data: str) -> None:
        """Sends the given data to the device."""
        self.ser.write((data + "\n").encode())

    def receive_data(self) -> str:
        """Receives data from the device.

        Returns:
            str: The received data from the device.
        """
        response = self.ser.readline().decode().strip()
        return response

    def receive_data_raw(self, num_bytes: int) -> bytes:
        """Receives raw data from the device.

        Args:
            num_bytes (int): The number of bytes to receive.

        Returns:
            bytes: The received raw data from the device.
        """
        response = self.ser.read(num_bytes)
        return response
