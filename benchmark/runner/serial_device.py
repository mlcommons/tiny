import serial
import time


class SerialDevice:
  def __init__(self, port_device, baud_rate, end_of_response="", delimiter="\n"):
    self.port = serial.Serial(port_device, baud_rate, timeout=0.1)
    self.delimiter = delimiter
    self.end_of_response = end_of_response

  def __enter__(self):
    self.port.__enter__()
    return self

  def __exit__(self, *args):
    self.port.__exit__(*args)

  def write(self, text, echo=False):
    self.port.write(text.encode())
    if echo: print(text, end='')

  def write_line(self, text, echo=False):
    self.write(text, echo)
    self.write(self.delimiter, echo=False)
    if echo: print()

  def read_line(self, timeout=0):
    result = ""
    txt = None
    start_time = round(time.time() * 1000)
    while txt != "\n" and (not timeout or round(time.time() * 1000) - start_time < timeout):
        txt = self.port.read(1).decode()
        if txt and txt != "\n":
            result = result + txt
    else:
      if txt != "\n":
        return None
    return result.replace('\r', '').replace('\0', '')

  def send_command(self, command, end=None, echo=False):
    self.write_line(command, echo)
    lines = []

    while True:
      resp = self.read_line()
      end_of_resp = (end if end is not None else self.end_of_response) in resp
      if resp:
        lines.append(resp)
      if end_of_resp:
        break

    return lines if len(lines) != 1 else lines[0]
