import re
import sys

from interface_device import InterfaceDevice
from serial_device import SerialDevice


class DUT:
  def __init__(self, port_device, baud_rate=115200, power_manager=None):
    interface = port_device
    if not isinstance(port_device, InterfaceDevice):
      interface = SerialDevice(port_device, baud_rate, "m-ready", '%')
    self._port = interface
    self.power_manager = power_manager
    self._profile = None
    self._model = None
    self._name = None
    self._max_bytes = 26 if power_manager else 31

  def __enter__(self):
    if self.power_manager:
      self.power_manager.__enter__()
    self._port.__enter__()
    return self

  def __exit__(self, *args):
    self._port.__exit__(*args)
    if self.power_manager:
      self.power_manager.__exit__(*args)

  def _get_name(self):
    name_retrieved = False
    for l in self._port.send_command("name"):
      match = re.match(r'^m-(name)-dut-\[([^]]+)]$', l)
      if match:
        self.__setattr__(f"_{match.group(1)}", match.group(2))
        name_retrieved = True
    if not name_retrieved:
      print(f"WARNING: Failed to get name.")

  def get_name(self):
    if self._name is None:
      self._get_name()
    return self._name

  def _get_profile(self):
    for l in self._port.send_command("profile"):
      match = re.match(r'^m-(model|profile)-\[([^]]+)]$', l)
      if match:
        self.__setattr__(f"_{match.group(1)}", match.group(2))

  def get_model(self):
    if self._model is None:
      self._get_profile()
    return self._model

  def get_profile(self):
    if self._profile is None:
      self._get_profile()
    return self._profile

  def timestamp(self):
    return self._port.send_command("timestamp")

  def send_data(self, data):
    size = len(data)
    pass

  def load(self, data):
    self._port.send_command(f"db load {len(data)}")
    i = 0
    while i < len(data):
      cmd = f"db {''.join(f'{d:02x}' for d in data[i:i+self._max_bytes])}"
      result = self._port.send_command(cmd)
      i += self._max_bytes
    return result

  def infer(self, number, warmups):
    command = f"infer {number}"
    if warmups:
      command += f" {warmups}"
    if self.power_manager:
      print(self.power_manager.start())
    result = self._port.send_command(command)
    if self.power_manager:
      print(self.power_manager.stop())
    return result

  def get_help(self):
    return self._port.send_command("help")

  """
  ULPMark for tinyML Firmware V0.0.1

  help         : Print this information
  name         : Print the name of the device
  timestsamp   : Generate a timetsamp
  db SUBCMD    : Manipulate a generic byte buffer
    load N     : Allocate N bytes and set load counter
    db HH[HH]* : Load 8-bit hex byte(s) until N bytes
    print [N=16] [offset=0]
               : Print N bytes at offset as hex
  infer N [W=0]: Load input, execute N inferences after W warmup loops
  results      : Return the result fp32 vector
  """
