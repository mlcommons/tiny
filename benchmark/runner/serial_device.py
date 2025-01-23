import sys
from queue import Empty, Queue
from threading import Thread

import serial


class SerialDevice:
  def __init__(self, port_device, baud_rate, end_of_response="", delimiter="\n", echo=False):
    self._port = serial.Serial(port_device, baud_rate, timeout=0.1)
    self._delimiter = delimiter
    self._end_of_response = end_of_response
    self._message_queue = Queue()
    self._read_thread = None
    self._running = False
    self._echo = echo
    self._timeout = 5.0
    self.serial_port = port_device
    self.baud_rate = baud_rate
    self.ser = None

  def __enter__(self):
    self._port.__enter__()
    self._start_read_thread()
    return self

  def __exit__(self, *args):
    self._stop_read_thread()
    self._port.__exit__(*args)

  def _read_loop(self):
    msg = ""
    while self._running:
      char = self._port.read(1).decode()
      if char and char not in '\n\r\0':
        msg = msg + char
      elif char == '\n':
        if self._echo: print(f"RX:{msg}")
        self._message_queue.put(msg)
        msg = ""
    else:
      if msg:
        print(f"Lost bytes: {msg}")

  def _start_read_thread(self):
    self._running = True
    self._read_thread = Thread(target=self._read_loop)
    self._read_thread.start()

  def _stop_read_thread(self):
    self._running = False
    self._read_thread.join()

  def write(self, text):
    self._port.write(text.encode())

  def write_line(self, text):
    self.write(text)
    self.write(self._delimiter)

  def read_line(self, timeout=None):
    """
    Read from the serial port.  If nothing is available within timeout seconds, returns None.
    """
    result = None
    if timeout is None:
      timeout = self._timeout
    try:
      result = self._message_queue.get(timeout=timeout)
    except Empty:
      pass
    return result

  def send_command(self, command, end=None, echo=False):
    if echo or self._echo: print(command + (self._delimiter if '\n' not in self._delimiter else ''))
    self.write_line(command)
    lines = []

    while True:
      resp = self.read_line()
      if resp is None:
        raise RuntimeError(f"No response to command {command}")
      end_of_resp = (end if end is not None else self._end_of_response) in resp
      if resp:
        lines.append(resp)
      if end_of_resp:
        break

    return lines if len(lines) != 1 else lines[0]
  
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