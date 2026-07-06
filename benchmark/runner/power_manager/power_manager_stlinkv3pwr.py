# power_manager_stlinkv3pwr.py

import re
import sys
from .power_manager_lpm import LPMCommands


class STLinkV3PWRCommands(LPMCommands):
    PROMPT = "stlp > "

    def __init__(self, manager, port):
        super().__init__(manager, port)

    def setup(self):
        self._send_command("htc")
        self.power_off()
        self.configure_trigger("inf", 0, "sw")
        self.configure_output("energy", "ascii_dec", "1k")
        self.configure_voltage(self.m._voltage)


    def read_loop(self):
        in_summary = False

        while self.m._running:
            line = self._port.read_line(timeout=0.25)
            if line is None:
                continue
            if not line:
                continue

            temp = self._strip_prompt(line)
            if not temp:
                continue

            if temp == "summary beg":
                in_summary = True
                self.m._message_queue.put(temp)
                continue

            if in_summary:
                self.m._message_queue.put(temp)
                if temp == "summary end":
                    in_summary = False
                continue

            if re.fullmatch(r"\d{4}[+-]\d{2}", temp):
                value = self._decode_ascii_dec_value(temp)
                self.m._data_queue.put(value)
            elif re.fullmatch(r"event \d+ (ris|fal)", temp):
                self.m._data_queue.put(temp)
            elif temp == "end":
                self.m._message_queue.put("Acquisition completed")
            else:
                self.m._message_queue.put(temp)


    def power_on(self):
        return self._send_command("pwr on nostatus")

    def get_board_id(self):
        if not self.m._board_id:
            result, output = self._send_command("whoami")
            self.m._board_id = output if result else None
        return self.m._board_id


    def stop(self):
        self._port.write_line("stop")
        while True:
            line = self.m._message_queue.get()
            temp = self._strip_prompt(line)
            if temp == "Acquisition completed" or temp == "end":
                break
        return True

    def _strip_prompt(self, line):
        line = line.strip()
        if line.startswith(self.PROMPT):
            line = line[len(self.PROMPT):].strip()
        return line


    def _read_response(self, command):
        out_lines = []

        while True:
            line = self.m._message_queue.get()
            temp = self._strip_prompt(line)

            if not temp:
                continue

            if temp.startswith("ack"):
                remainder = temp[3:].strip()
                out_lines.append("ack")
                if remainder:
                    if remainder.startswith(command):
                        tail = remainder[len(command):].strip()
                        if tail.startswith(":"):
                            tail = tail[1:].strip()
                        if tail:
                            out_lines.append(tail)
                    else:
                        out_lines.append(remainder)
                break

            elif temp.startswith("err"):
                remainder = temp[3:].strip()
                out_lines.append("err")
                if remainder:
                    out_lines.append(remainder)
                break

            else:
                out_lines.append(temp)

        return out_lines

    def _read_output(self):
        while True:
            line = self.m._message_queue.get()
            temp = self._strip_prompt(line)
            if temp == "":
                return
            yield temp

    def _read_error_output(self):
        errors = []
        while not self.m._message_queue.empty():
            line = self.m._message_queue.get()
            temp = self._strip_prompt(line)
            if temp:
                errors.append(temp)
        return errors if errors else ["Unknown STLINK-V3PWR error"]

    def _decode_ascii_dec_value(self, s):
        return float(s.replace("+", "e+").replace("-", "e-"))
