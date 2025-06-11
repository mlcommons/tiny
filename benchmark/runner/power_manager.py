import re
import sys
from queue import Queue
from serial_device import SerialDevice
from threading import Thread
from abc import ABC, abstractmethod


class PowerManager(SerialDevice):
  # device prefaces output with PROMPT. This will always be stripped out of responses
  PROMPT = "" 

  def __init__(self, port_device, baud_rate=None, voltage=3.3, echo=False):
    self._voltage = voltage
    self._board_id = None
    self._version = None
    self._in_progress = False
    self._data_queue = Queue()
    self._message_queue = Queue()
    self._read_thread = None
    self._running = False
    # sub-classes should construct _port (SerialDevice) and any 
    # display (e.g. _lcd in lpm01a) and call self.__enter__()

  def __enter__(self):
    self._port.__enter__()
    self._start_read_thread()
    self._setup()
    return self

  def __exit__(self, *args):
    self._tear_down()
    self._stop_read_thread()
    self._port.__exit__(*args)

  @abstractmethod
  def extract_current_values(line):
    """ Must be implemented by subclass.  Takes a line of text from
    energy monitor device and returns a list of energy samples.
    """
    pass
    
  def _read_loop(self):
    # Leave the basic loop in the abstract class, but then call a method
    # to process the line when there is one.  So something like this, where
    # process_line() is an abstract function that each subclass has to implement.
    # while self._running:
    #   line = self._port.read_line(timeout=0.250)
    #   if line:
    #     self.process_line(line)
    while self._running:
      line = self._port.read_line(timeout=0.250)
      if not line:
        continue
      if line.startswith("TimeStamp"):
        self._data_queue.put(line)
      elif re.match("\d\d\d\d[+-]\d\d", line):
        values = self.extract_current_values(line)
        for v in values: 
          self._data_queue.put(v)
      elif re.match(r"event (\d+) ris", line): # event markers indicate an occurence (D7 falling edge here)
        self._data_queue.put(line)
      else:
        self._message_queue.put(line)

  def _start_read_thread(self):
    self._running = True
    self._read_thread = Thread(target=self._read_loop)
    self._read_thread.start()

  def _stop_read_thread(self):
    self._running = False
    self._read_thread.join()

  def _setup(self):
    # make this an abstract class that each subclass must implement

    # verify board
    self._claim_remote_control()
    print(f"LPM01A Power Monitor Board", file=sys.stderr)
    print(f"BoardID: {self.get_board_id()}", file=sys.stderr)
    print(f"Version: {self.get_version()}", file=sys.stderr)
    print(f"Status: {self.get_status()}", file=sys.stderr)
    self.set_lcd("MLPerf Tiny", "     monitor    ")
    self.power_off()
    # Acquire infinitely
    # self.configure_trigger('inf', 0, 'd7')
    # trigger='sw' => start measuring on command; ='d7'=>wait for d7 then start
    self.configure_trigger('inf', 0, 'sw') 

    self.configure_output('energy', 'ascii_dec', '1k')
    # a little redundant.  sets the internal variable self._voltage and sends the command to the device
    self.configure_voltage(self._voltage) 

  def _tear_down(self):
    # this can maybe all be handled in the base class. Then if a subclass needs to do something 
    # extra or different, it can either override, or override and call superclass._tear_down(), 
    # like init() does.
    self.stop()
    self.power_off()
    self._release_remote_control()

  def _claim_remote_control(self):
    # make this an abstract class that each subclass must implement
    self._send_command("htc")

  def _release_remote_control(self):
    # make this an abstract class that each subclass must implement
    self._send_command("hrc")

  def get_board_id(self):
    # make this an abstract class that each subclass must implement
    if not self._board_id:
      result, output = self._send_command("powershield", err_message="Error getting BoardId")
      self._board_id = output if result else None
    return self._board_id

  def get_version(self):
    # make this an abstract class that each subclass must implement
    if not self._version:
      result, output = self._send_command("version", err_message="Error getting version")
      self._version = output[1] if result else None
    return self._version

  def get_lcd(self):
    # make this an abstract class that each subclass must implement
    return self._lcd

  def set_lcd(self, *args):
    # make this an abstract class that each subclass must implement.  If the 
    # device doesn't have an LCD, it can just pass.  If you think there's a cleaner 
    # way to handle this, that's fine too.
    for i in range(len(args)):
      if args[i] and args[i] != self._lcd[i]:
        result, _ = self._send_command(f'lcd {i+1} "{args[i]}"',
                                       err_message=f"Error setting LCD line {i+1} to \"{args[i]}\"")
        self._lcd[i] = args[i] if result else self._lcd[i]
    return self._lcd

  def configure_trigger(self, acquisition_time, trigger_delay, trigger_source):
    # probably make this an abstract class that each subclass must implement
    # this may have to be refactored since the arguments (acquisition_time, 
    # trigger_delay, trigger_source) are LPM-specific
    self._send_command(f"acqtime {acquisition_time}",
                       err_message=f"Error setting acquisition_time to {acquisition_time}")
    self._send_command(f"trigdelay {trigger_delay}",
                       err_message="Error setting trigger_delay to {trigger_delay}")
    self._send_command(f"trigsrc {trigger_source}",
                       err_message=f"Error setting trigger_source to {trigger_source}")
    self._send_command(f"eventsrc d7 fal",
                        err_message=f"Error setting event source to D7 Falling edge")

  def configure_output(self, output_type, output_format, samples_per_second):
    # probably make this an abstract class that each subclass must implement
    # this may have to be refactored since the arguments are LPM-specific    
    self._send_command(f"output {output_type}", err_message=f"Error setting output_type to {output_type}")
    self._send_command(f"format {output_format}", err_message=f"Error setting output_format to {output_format}")
    self._send_command(f"freq {samples_per_second}",
                       err_message=f"Error setting samples_per_second to {samples_per_second}")

  def power_on(self, show_status=False):
    # make this an abstract class that each subclass must implement
    self.set_lcd(None, f"{self._voltage : >14}V ")
    self._send_command(f"pwr on {'' if show_status else 'no'}status", err_message=f"Error turning on power")

  def power_off(self):
    #  make this an abstract class that each subclass must implement
    self._send_command("pwr off", err_message=f"Error turning off power")

  def configure_voltage(self, voltage):
    # make this an abstract class that each subclass must implement
    self._voltage = voltage
    voltage_str = f"{int(1000*(voltage))}m" # express in mV eg 1.1V => "1100m"
    self._send_command(f"volt {voltage_str}", err_message=f"Error setting voltage to {voltage_str}V")

  def start(self):
    #  make this an abstract class that each subclass must implement
    return self._send_command("start")

  def stop(self):
    # make this an abstract class that each subclass must implement
    # stop does not ack
    print("PM: stop")
    self._port.write_line("stop")
    lines = []
    line = None
    while not line or "Acquisition completed" not in line:
      line = self._message_queue.get()
      if line:
        lines.append(line)
    print("Done with PM.stop()")
    return lines

  def get_results(self):
    # this can probably stay in the base class
    while not self._data_queue.empty():
      yield self._data_queue.get()

  def get_status(self):
    # make this an abstract class that each subclass must implement
    result, output = self._send_command("status", err_message="Error getting status")
    return output if result else None

  def get_help(self):
    #  make this an abstract class that each subclass must implement
    result, output = self._send_command("help", True, err_message="Error getting help")
    return output if result else None

  def _read_response(self, command):
    # split kind of like _read_loop

    # should probably add a timeout to this function
    out_lines = []
    while True:
      line = self._message_queue.get()

      temp = line.replace(self.PROMPT, "").strip()
      if temp and command in temp and (temp.startswith('ack') or temp.startswith('error')):
        out_lines.extend(r for r in temp.replace(command, "").split(" ", 2) if r)
        break
      elif temp and not temp.startswith("ack") and not temp.startswith("error"):
        out_lines.append(temp)
       
      print(f"pwr mgr rxd: {line} => {temp}")
    return out_lines

  def _read_error_output(self):
    # split kind of like _read_loop
    while True:
      line = self._message_queue.get()
      line = line.replace(self.PROMPT, "").strip()
      if line.startswith("Error detail"):
        return [line.replace("Error detail:", "").strip()]

  def _read_output(self):
    # not sure if this be kept in the base class like this or not.
    while True:
      line = self._message_queue.get()
      # print(f"OUT: {line}")
      if line == self.PROMPT:
        return
      line = line.replace(self.PROMPT, "").strip()
      if line:
        yield line

  def _purge_messages(self):
    # can probably stay in the base class
    while not self._message_queue.empty():
      line = self._message_queue.get()
      if line:
        print(f"Dropping message: {line}", file=sys.stderr)

  def _send_command(self, command, expect_output=False, err_message=None):
    # probably split into some generic structure and move specific commands to 
    # sub-class, a little like _read_loop()
    self._purge_messages()
    print(f"pwr {command}")
    self._port.write_line(command)
    lines = self._read_response(command)
    result = lines and lines[0] == 'ack'
    output = lines[1:] if lines and len(lines) > 1 else []
    if not result:
      output = self._read_error_output()
      if err_message is not None:
        print(f"{err_message}: {output[0]}", file=sys.stderr)
    elif expect_output:
      output = [l for l in self._read_output()]
    return result, output if not output or len(output) != 1 else output[0]

  