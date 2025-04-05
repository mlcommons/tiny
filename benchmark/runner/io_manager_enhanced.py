from io_manager import IOManager


class IOManagerEnhanced(IOManager):
  def __init__(self, port_device, baud_rate=115200, echo=None):
    kw_args = {"baud_rate":baud_rate,
               "echo":echo
               }
    IOManager.__init__(self, port_device, **kw_args)

  def get_files(self):
    return self.port.send_command('ls')

  def get_waves(self):
    return [x for x in self.get_files() if x.lower().endswith("wav") or x == "m-ready"]

  def play_wave(self, filename=None, timeout=10):
    command = "play" + (f" {filename}" if filename else "")
    return self.port.send_command(command, timeout=timeout) # need to set this timeout longer than file length

  def record_detections(self):
    command = "record_detections"
    return self.port.send_command(command)

  def print_detections(self):
    command = "print_detections"
    return self.port.send_command(command)
