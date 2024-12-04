from io_manager import IOManager


class IOManagerEnhanced(IOManager):
  def __init__(self, port_device, baud_rate=921600):
    IOManager.__init__(self, port_device, baud_rate)

  def get_files(self):
    return self.port.send_command('ls')

  def get_waves(self):
    return [x for x in self.get_files() if x.lower().endswith("wav") or x == "m-ready"]

  def play_wave(self, filename=None):
    command = "play" + (f" {filename}" if filename else "")
    return self.port.send_command(command)