import sys
from queue import Empty, Queue
from threading import Thread

import serial, time
 
class SerialDevice:
  def __init__(self, port_device, baud_rate, end_of_response="", delimiter="\n", echo=False):
    print(f"Initializing SerialDevice on port: {port_device} at {baud_rate} baud")  # Debug print
    self._port = serial.Serial(port_device, baud_rate, timeout=0.1)
    self._delimiter = delimiter
    self._end_of_response = end_of_response
    self._message_queue = Queue()
    self._read_thread = None
    self._running = False
    self._echo = echo
    self._timeout = 5.0
    self.full_debug = False
    self._entry_count = 0
    # self.reset_port()

  def __enter__(self):
    print(f"SerialDevice entering port {self._port.port}, entry_count (before increment) = {self._entry_count}")
    if not self._entry_count:
      self._port.__enter__()
      self._start_read_thread()
    self._entry_count += 1
    return self

  def __exit__(self, *args):
    print(f"SerialDevice exiting port {self._port.port}, entry_count (before decrement) = {self._entry_count}")
    self._entry_count -= 1    
    if not self._entry_count:
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

  def send_command(self, command, end=None, echo=False, timeout=None):
    if echo or self._echo: print(command + (self._delimiter if '\n' not in self._delimiter else ''))
    self.write_line(command)
    lines = []

    while True:
      resp = self.read_line(timeout=timeout)
      if resp is None:
        raise RuntimeError(f"No response to command {command}")
      end_of_resp = (end if end is not None else self._end_of_response) in resp
      if resp:
        lines.append(resp)
      if end_of_resp:
        break

    return lines if len(lines) != 1 else lines[0]

  def close_serial(self) -> None:
        """Closes the serial communication."""
        if self._port and self._port.is_open:
            self._port.close()
            print("Serial connection closed.")

  def send_data(self, data: str) -> None:
        """Sends the given data to the device."""
        self._port.write((data + "\n").encode())

  def receive_data(self) -> str:
    """Receives data from the device."""
    if not self._port.is_open:
        print("Error: Serial port is closed. Attempting to reopen...")
        self._port.open()  # Reopen the port if needed

    try:
        response = self._port.readline().decode().strip()
        return response
    except serial.SerialException as e:
        print(f"SerialException: {e}")
        return ""  # Return an empty response instead of crashing
    
  def reset_port(self, timeout=5):
    t0 = time.time()
    self._port.reset_input_buffer()  # flush buffers on host
    self._port.reset_output_buffer()
    # flush any partial command in progress on DUT, don't compare txd/rxd cmd

    self._port.write(self._delimiter.encode())
    print(f"Flushing port: {self._port}")
    while(True):
      b0 = self._port.read(1)
      if b0 == b'': # nothing more to read
          break
      else:
         print(b0.decode(), end="")
      if timeout is not None and time.time() - t0 > timeout:
          break  