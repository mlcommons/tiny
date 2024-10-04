import sys
from queue import Empty, Queue
from threading import Thread

import serial


class SerialDevice:
  def __init__(self, port_device, baud_rate, end_of_response="", delimiter="\n"):
    self._port = serial.Serial(port_device, baud_rate, timeout=0.1)
    self._delimiter = delimiter
    self._end_of_response = end_of_response
    self._message_queue = Queue()
    self._read_thread = None
    self._running = False
    self._echo = True
    self._timeout = 1.0

  def __enter__(self):
    print(f"Entering SerialDevice.__enter(), port={self._port}.")
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

  def send_command(self, command, end=None, echo=True):
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
