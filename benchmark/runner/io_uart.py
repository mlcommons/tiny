from typing import Optional, Union, List
from interface_device import InterfaceDevice
from serial_device import SerialDevice


class IOUart(InterfaceDevice):
    """Represents an UART interface for communication over a serial device.

    This class provides a high-level interface for sending commands to and 
    receiving responses from a DUT.
    """

    def __init__(self, port_device: str, baud_rate: int = 115200) -> None:
        """Initializes the IOUart interface.

        Args:
            port_device: The device port.
            baud_rate: The baud rate for communication.

        Raises:
            RuntimeError: If the initialization of the SerialDevice fails.
        """
        try:
            self.port: SerialDevice = SerialDevice(port_device, baud_rate, "m-ready", '%')
            self.is_port_open = False
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SerialDevice on port {port_device}: {e}")

    def __enter__(self) -> "IOUart":
        """Enters the context for the IOUart interface.

        Opens the underlying serial port if it is not already open.

        Returns:
            The IOUart instance for use in the context.

        Raises:
            RuntimeError: If the port could not be opened successfully.
        """
        if not self.is_port_open:
            try:
                self.port.__enter__()
                self.is_port_open = True
            except Exception as e:
                raise RuntimeError(f"Failed to open the port: {e}") from e
        return self

    def __exit__(self, *args) -> None:
        """Exits the context for the IOUart interface.

        Closes the underlying serial port if it is open.

        Args:
            *args: Arguments passed to the `__exit__` method.
        """
        if self.is_port_open:
            try:
                self.port.__exit__(*args)
            except Exception as e:
                print(f"Error while closing the port: {e}")
                raise
            finally:
                self.is_port_open = False

    def send_command(
        self,
        command: str
    ) -> Union[str, List[str]]:
        """Sends a command to the serial device and retrieves the response.

        Args:
            command: The command string to send.

        Returns:
            str | list[str]: The response(s) from the serial device.

        Raises:
            ValueError: If no response is received for the command.
            RuntimeError: If an error occurs during communication with the serial device.
        """
        try:
            resp: List[str] = self.port.send_command(command)
            if not resp:
                raise ValueError(f"IOUart: No response received for command: {command}")
            return resp if len(resp) != 1 else resp[0]
        except Exception as e:
            raise RuntimeError(f"IOUart: Failed to send command '{command}': {e}")
