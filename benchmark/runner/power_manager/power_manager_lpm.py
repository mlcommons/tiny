# power_manager_lpm.py

import re
import sys


class LPMCommands:
    def __init__(self, manager, port):
        self.m = manager
        self._port = port

    def setup(self):
        self._send_command("htc")
        self.set_lcd("MLPerf Tiny", "     monitor    ")
        self.power_off()
        self.configure_trigger("inf", 0, "sw")
        self.configure_output("energy", "ascii_dec", "1k")
        self.configure_voltage(self.m._voltage)

    def tear_down(self):
        self.stop()
        self.power_off()
        self._send_command("hrc")

    def read_loop(self):
        while self.m._running:
            line = self._port.read_line(timeout=0.25)
            if not line:
                continue
            if line.startswith("TimeStamp"):
                self.m._data_queue.put(line)
            elif re.match(r"\d{4}[+-]\d{2}", line):
                values = self._extract_current_values(line)
                for v in values:
                    self.m._data_queue.put(v)
            elif re.match(r"event \d+ ris", line):
                self.m._data_queue.put(line)
            else:
                self.m._message_queue.put(line)

    def configure_trigger(self, acqtime, trigdelay, trigsrc):
        self._send_command(f"acqtime {acqtime}")
        self._send_command(f"trigdelay {trigdelay}")
        self._send_command(f"trigsrc {trigsrc}")
        self._send_command("eventsrc d7 fal")

    def configure_output(self, output_type, output_format, freq):
        self._send_command(f"output {output_type}")
        self._send_command(f"format {output_format}")
        self._send_command(f"freq {freq}")

    def configure_voltage(self, voltage):
        self.m._voltage = voltage
        self._send_command(f"volt {int(voltage * 1000)}m")

    def power_on(self):
        self.set_lcd(None, f"{self.m._voltage: >14}V ")
        return self._send_command("pwr on nostatus")

    def power_off(self):
        return self._send_command("pwr off")

    def start(self):
        return self._send_command("start")

    def stop(self):
        self._port.write_line("stop")
        while True:
            line = self.m._message_queue.get()
            if "Acquisition completed" in line:
                break
        return True

    def get_board_id(self):
        if not self.m._board_id:
            result, output = self._send_command("powershield")
            self.m._board_id = output if result else None
        return self.m._board_id

    def get_version(self):
        if not self.m._version:
            result, output = self._send_command("version")
            self.m._version = output[1] if result else None
        return self.m._version

    def get_status(self):
        result, output = self._send_command("status")
        return output if result else None

    def set_lcd(self, *args):
        for i in range(len(args)):
            if args[i] and args[i] != self.m._lcd[i]:
                result, _ = self._send_command(f'lcd {i+1} "{args[i]}"')
                if result:
                    self.m._lcd[i] = args[i]
        return self.m._lcd

    # Internal utilities moved from PowerManager

    def _send_command(self, command, expect_output=False, err_message=None):
        self._purge_messages()
        self._port.write_line(command)
        lines = self._read_response(command)
        result = lines and lines[0] == 'ack'
        output = lines[1:] if lines and len(lines) > 1 else []
        if not result:
            print(f"Power Manager did not acknowledge.  PM Response:")
            print(lines)
            output = self._read_error_output()
            if err_message:
                print(f"{err_message}: {output[0]}", file=sys.stderr)
        elif expect_output:
            output = list(self._read_output())
        return result, output if not output or len(output) != 1 else output[0]

    def _purge_messages(self):
        while not self.m._message_queue.empty():
            self.m._message_queue.get()

    def _read_response(self, command):
        out_lines = []
        while True:
            line = self.m._message_queue.get()
            temp = line.replace(self.m.PROMPT, "").strip()
            if temp and command in temp and (temp.startswith("ack") or temp.startswith("error")):
                out_lines.extend(r for r in temp.replace(command, "").split(" ", 2) if r)
                break
            elif temp and not temp.startswith("ack") and not temp.startswith("error"):
                out_lines.append(temp)
        return out_lines

    def _read_output(self):
        while True:
            line = self.m._message_queue.get()
            if line == self.m.PROMPT:
                return
            yield line.replace(self.m.PROMPT, "").strip()

    def _read_error_output(self):
        while True:
            line = self.m._message_queue.get()
            line = line.replace(self.m.PROMPT, "").strip()
            if line.startswith("Error detail"):
                return [line.replace("Error detail:", "").strip()]

    def _extract_current_values(self, line):
        pattern = r'\d{4}[-+]\d{2}'
        matches = list(re.finditer(pattern, line))
        results = []
        if matches:
            if matches[0].start() > 0:
                results.append("nan")
            results.extend(match.group() for match in matches)
            if matches[-1].end() < len(line):
                results.append("nan")
        return [float(s.replace("+", "e+").replace("-", "e-")) for s in results]



"""
################################################################################
#                             PowerShield commands                             #
################################################################################
#   Command    #                          Description                          #
################################################################################
# Common operation                                                             #
################################################################################
# help         # Displays list of commands.                                    #
#              #                                                               #
# echo <arg1>  # Loopback to check functionality of communication Rx and Tx.   #
#              # <arg1>: String of characters                                  #
#              #                                                               #
# powershield  # Check PowerShield device availability, can be used to scan    #
#              # on which serial port is connected the PowerShield.            #
#              # Response: 'PowerShield present' with board unique ID          #
#              #                                                               #
# version      # Get PowerShield FW revision.                                  #
#              # Response: '<main>.<sub1>.<sub2>'                              #
#              #                                                               #
# range        # Get PowerShield current measurement range.                    #
#              # Response: <current min> <current max>                         #
#              #                                                               #
# status       # Get PowerShield status.                                       #
#              # Response: 'ok' or 'error: <error description>'                #
#              #                                                               #
# htc          # Host take control (go from mode 'standalone'                  #
#              # to mode 'controlled by host')                                 #
# hrc          # Host release control (go from mode 'controlled by host'       #
#              # to mode 'standalone')                                         #
#              #                                                               #
# lcd          # Display a custom string on LCD display when PowerShield is    #
# <arg1>       # controlled by host.                                           #
# <arg2>       # <arg1>: LCD line. Numerical value among list: {1, 2}          #
#              # <arg2>: String to be displayed, surrounded by double quotes   #
#              #         and with 16 characters maximum.                       #
#              # Example: lcd 1 "  custom display"                             #
#              #                                                               #
# psrst        # Reset PowerShield (hardware reset,                            #
#              # host communication will have to be restored).                 #
#              #                                                               #
################################################################################
# Measurement acquisition configuration                                        #
################################################################################
# volt <arg1>  # Set or get power supply voltage level, unit: V.               #
#              # <arg1>: Set voltage: Numerical value in range [1800m; 3300m]  #
#              #                      Default value: 3300m                     #
#              #         Get voltage: String 'get'                             #
#              #                                                               #
# freq <arg1>  # Set sampling frequency, unit: Hz.                             #
#              # <arg1>: Numerical value among list:                           #
#              #         {100k, 50k, 20k, 10k, 5k, 2k, 1k, 500, 200            #
#              #          100, 50, 20, 10, 5, 2, 1}                            #
#              #         Default value: 100Hz                                  #
#              #                                                               #
# acqtime      # Set acquisition time, unit: s.                                #
# <arg1>       # <arg1>: For limited acquisition duration:                     #
#              #           Numerical value in range: [100u; 100]               #
#              #         For infinite acquisition duration:                    #
#              #           Numerical value '0' or string 'inf'                 #
#              #         Caution: Maximum acquisition time depends on other    #
#              #                  parameters. Refer to table below.            #
#              #         Default value: 10s                                    #
#              #                                                               #
# acqmode      # Set acquisition mode: dynamic or static.                      #
# <arg1>       #   dynamic: current can vary, range [100nA; 10mA]              #
#              #   static: current must be constant, range [2nA; 200mA]        #
#              # <arg1>: String among list: {'dyn', 'stat'}                    #
#              #         Default value: 'dyn'                                  #
#              #                                                               #
# funcmode     # Set optimization of acquisition mode dynamic (applicable      #
# <arg1>       # only with command 'output' set to parameter 'current'):       #
#              #   optim: priority on current resolution (100nA-10mA),         #
#              #          max sampling frequency at 100KHz.                    #
#              #   high:  high current (30uA-10mA), high sampling frequency    #
#              #          (50-100KHz), high resolution.                        #
#              # <arg1>: String among list: {'optim', 'high'}                  #
#              #         Default value: 'optim'                                #
#              #                                                               #
# output       # Set output type.                                              #
# <arg1>       #   current: instantaneous current                              #
#              #   energy:  integrated energy, reset after each sample sent    #
#              #            (integration time set by param 'freq',             #
#              #            limited at 10kHz max (<=> 100us min)).             #
#              # <arg1>: String among list: {'current', 'energy'}              #
#              #         Default value: 'current'                              #
#              #                                                               #
# format <arg1># Set measurement data format.                                  #
#              # Data format 1: ASCII, decimal basis.                          #
#              #                Format readable directly, but sampling         #
#              #                frequency limited to 10kHz.                    #
#              #                Decoding: 6409-07 <=> 6409 x 10^-7 = 640.9uA   #
#              # Data format 2: Binary, hexadecimal basis.                     #
#              #                Format optimized data stream size.             #
#              #                Decoding: 52A0 <=> (2A0)16 x 16^-5 = 640.9uA   #
#              # <arg1>: String among list: {'ascii_dec', 'bin_hexa'}          #
#              #         Caution: Data format depends on other                 #
#              #                  parameters. Refer to table below.            #
#              #         Default value: 'ascii_dec'                            #
#              #                                                               #
# trigsrc      # Set trigger source to start measurement acquisition:          #
# <arg1>       # trigger source SW (immediate trig after SW start),            #
#              # trigger from external signal rising or falling edge on        #
#              # Arduino connector D7 (via solder bridge).                     #
#              # <arg1>: String among list: {'sw', 'd7'}                       #
#              #         Default value: 'sw'                                   #
#              #                                                               #
# trigdelay    # Set trigger delay between target power-up and start           #
# <arg1>       # measurement acquisition, unit: s, resolution: 1ms.            #
#              # <arg1>: Numerical value in range [0; 30]                      #
#              #         Default value: 1m                                     #
#              #                                                               #
# currthres    # Set current threshold to trig an event, unit: A.              #
# <arg1>       # Event triggered when threshold exceeded: signal generated     #
#              # on Arduino connector D2 or D3 (via solder bridge)             #
#              # and LED4 (blue) turned on.                                    #
#              # <arg1>: Numerical value in range [100n; 50m]                  #
#              #         or value '0' for threshold disable                    #
#              #         Default value: 1m                                     #
#              #                                                               #
# pwr          # Set target power supply connection.                           #
# <arg1>       # Automatic: On first run, power-on when acquisition start.     #
#(<arg2>)      #            Then, power state depends on command 'pwrend'.     #
#              # Manual: Force power state.                                    #
#              #         Note: Can be used during acquisition. To perform      #
#              #               successive power off and on, it is preferable   #
#              #               to use command 'targrst'.                       #
#              # Optionally, connection status can be sent at the beginning    #
#              # and end of acquisition data stream.                           #
#              # <arg1>: Set pwr: String among list: {'auto', 'on', 'off'}     #
#              #                  Default value: 'auto'                        #
#              #         Get pwr: String 'get' (response: state 'on' or 'off') #
#              # <arg2>: Optional, string among list: {'nostatus', 'status'}   #
#              #         Default value: 'nostatus'                             #
#              #                                                               #
# pwrend       # Set target power supply connection to be applied after        #
# <arg1>       # acquisition: power-on or power-off.                           #
#              # <arg1>: String among list: {on, off}                          #
#              #         Default value: 'on'                                   #
#              #                                                               #
#              #                                                               #
################################################################################
# Measurement acquisition operation                                            #
################################################################################
# start        # Start acquisition (measurement of current or energy           #
#              # depending on configuration).                                  #
#              #                                                               #
# stop         # Stop acquisition. If acquisition is set to a finite duration, #
#              # it will stop when reaching the target duration.               #
#              #                                                               #
# targrst      # Reset target by disconnecting power supply during             #
# <arg1>       # a configurable duration, unit: s.                             #
#              # Note: Can be performed during acquisition to monitor target   #
#              #       transient current consumption during its power-up.      #
#              # <arg1>: Numerical value in range [10m; 1]                     #
#              #         or value '0' to let target powered-down               #
#              #                                                               #
# temp         # Get temperature from temperature sensor on PowerShield board, #
# <arg1>       # on unit: Degree Celsius or Fahrenheit.                        #
#              # <arg1>: String among list: {degc, degf}                       #
#              #         Default value: 'degc'                                 #
#              #                                                               #
################################################################################
# Board state operation                                                        #
################################################################################
# autotest     # Perform board autotest and display autotest results.          #
#(<arg1>)      # Note: Autotest is done at PowerShield power-up.               #
#(<arg2>)      # <arg1>: Optional: string among list: {start, status}          #
#              #                   or no argument equivalent to value: 'start' #
#              # <arg2>: Optional: results status all or minimal (ok - not ok) #
#              #                   string among list: {stat_all, stat_min}     #
#              #                                                               #
# calib        # Perform board self-calibration.                               #
#              # Note: New calibration should be performed when temperature    #
#              #       shifts of more than 5 degC since the previous           #
#              #       calibration.                                            #
#              # Note: Calibration is done at PowerShield power-up             #
#              #       and after each update of power supply voltage level.    #
#              #                                                               #
################################################################################
# Note: Numerical values of arguments can be formatted either:                 #
#       - Numerical characters only (possible when numbers are >= 1)           #
#       - Numerical characters with unit characters 'u', 'm', 'k', 'm'         #
#       - Numerical characters with power of ten '-xx' or '+xx'                #
#         (xx: number on 2 digits maximum)                                     #
#       Example: Value '2 milliseconds' can be entered with: '2m' or '2-3'     #
################################################################################
# Table of maximum acquisition time possible in function of                    #
# sampling frequency and data format,                                          #
# for a baudrate of 3686400 bauds:                                             #
#    ______________________________________________________________________    #
#    |  format        freq    |   acqtime max    (corresp. nb of samples) |    #
#    |____________________________________________________________________|    #
#    | ascii_dec  | <=  5k    |    unlimited     (unlimited)              |    #
#    | ascii_dec  |    10k    |        1s        (   10 000)              |    #
#    | ascii_dec  |    20k    |      500ms       (    5 000)              |    #
#    | bin_hexa   | <=100k    |    unlimited     (unlimited)              |    #
#    |____________________________________________________________________|    #
################################################################################
"""