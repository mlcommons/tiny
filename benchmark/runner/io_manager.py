from interface_device import InterfaceDevice
from serial_device import SerialDevice


class IOManager(InterfaceDevice):
  def __init__(self, port_device, baud_rate=115200):
    self.port = SerialDevice(port_device, baud_rate, "m-ready", '%')
    self.entry_count = 0

  def __enter__(self):
    if not self.entry_count:
      self.port.__enter__()
      self.get_name()
    self.entry_count += 1
    return self

  def __exit__(self, *args):
    self.entry_count -= 1
    if not self.entry_count:
      self.port.__exit__(*args)

  def get_name(self):
    return self.port.send_command("name")

  def timestamp(self):
    return self.port.send_command("timestamp")

  def get_help(self):
    return self.port.send_command("help")

  def send_data(self, data):
    size = len(data)
    pass

  def read_line(self):
    resp = self.port.read_line()
    resp = resp.replace("[dut]: ", "")
    return resp

  def send_command(self, command, end=None, echo=False):
    resp = self.port.send_command(f"dut {command}")
    if len(resp) != 2 or resp[1] != "m-ready":
      return None
    resp = None
    lines = []
    while resp != 'm-ready':
      resp = self.read_line()
      lines.append(resp)
    return lines if len(lines) != 1 else lines[0]

  """
   help
   name              device name
   res               Reset polarity 0*,1
   enable-timer      Enable timer ISR
   disable-timer     Disable ISR
   dut               DUT passthrough
   i2c-enable        N : Enable I2C slave-mode to send N bytes (mod16)
   i2c-disable       Disable above
   load-vdata        Load 16 random bytes into vdata
   show-vdata        Show vdata
   et                Start Emon
   tm                0=fall/1*=change
   tp                [res]
   version           firmware version
  """