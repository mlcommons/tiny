from io_manager import IOManager
import re
import time

class IOManagerEnhanced(IOManager):
  def __init__(self, port_device, baud_rate=115200):
    kw_args = {"baud_rate":baud_rate,
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
  
  def _sync_baud(self, baud):
    response = self.port.send_command("checkbaud")
    print(response)
    lines = [response] if isinstance(response, str) else response

    for line in lines:
        print(f"DEBUG checkbaud line: {line}")
        if "baud is:" in line:
            try:
                current_baud = int(line.split(":")[-1].strip())
                if current_baud != baud:
                    print(f"[INFO] Baud mismatch: current={current_baud}, desired={baud}. Updating...")
                    self.port.send_command(f"setbaud {baud}", end="m-ready")
                    
                    # Reset and reopen port
                    self.port._port.flush()
                    self.port._port.close()
                    time.sleep(1.0)
                    self.port._port.baudrate = baud
                    self.port._port.open()
                    print("[INFO] Baud updated and port reopened.")
                    return f"Baud updated to {baud}"
                else:
                    return f"Baud already {baud}, no change needed."
            except Exception as e:
                raise ValueError(f"Could not parse baud from line: '{line}'") from e
