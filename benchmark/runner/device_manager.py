import yaml
from serial.tools import list_ports
import serial, re
import usb.core

from device_under_test import DUT
from io_manager import IOManager
from io_manager_enhanced import IOManagerEnhanced
from power_manager.power_manager import PowerManager
from runner_utils import get_baud_rate

def precheck_device_name(dev_cfg, serial_device, mode):
    """
    Called before constructing a device to check if a potential match provides
    the correct name.  Mainly used to disambiguate multiple devices with a common
    USB vid and pid.  If the dev_cfg does not have a "check_name" property, 
    return True.  If the device on <serial_device> does not respond to the
    "name%" command, or responds but the name does not match check_name return
    False. If the response matches check_name, return True.
    Note that this function uses the 'check_name' property, not 'name', which
    is mostly arbitrary
    ** Arguments:
    - dev_cfg: device configuration dict from devices.yaml
    - serial_device: serial port name (e.g. "/dev/cu.usbmodel1103" on a Mac)
    - mode: Test mode, "a", "p", or "e".  Used to determine correct baud rate.
    """
    if "check_name" not in dev_cfg:
         # no check_name specified, so skip this test.
         return True
    baud = get_baud_rate(dev_cfg, mode=mode)
    with serial.Serial(serial_device, baud, timeout=1) as port:
        port.write("name%".encode())
        response = ""
        while True:
            # read until there is nothing available or until an undecodable character is received.
            raw_char = port.read(1)
            if raw_char is None or raw_char == b'':
                # nothing available, so break
                break
            try:
                char = raw_char.decode()
            except UnicodeDecodeError as e:
                print("WARNING: Character decoding error while pre-checking name on {serial_device}")
                print(str(e))
                break
            response += char
    match = re.search(r"m-name-dut-\[(.*)\]", response)
    if match:
        received_name = match.group(1)
    else:
        print(f"Device on {serial_device} matches pid/vid to {dev_cfg['name']} but failed to respond to 'name%' command.")
        print(f"Will not associate this device. If this is unexpected, check baud rate and connection.")
        return False
    
    if dev_cfg["check_name"] == received_name:
        return True
    else:
        print(f"Device on {serial_device} matches pid/vid to {dev_cfg['name']} but gave non-matching check_name {received_name} (expected {dev_cfg['check_name']}). Will not associate.")
        return False


class DeviceManager:
    """Detects and identifies available devices attached to the host."""

    def __init__(self, device_defs, desired_baud, mode):
        self._device_defs = device_defs
        self._desired_baud = desired_baud
        self.mode = mode

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def get(self, item, default=None):
        return self.__dict__.get(item, default)

    def values(self):
        return (a for a in self.__dict__.values() if isinstance(a, dict))

    def _add_device(self, dev, definition, non_serial=False):
        dev_type = definition.get("type")
        if dev_type and (
            not self.__dict__.get(dev_type)
            or definition.get("preference", 0) > self.__dict__[dev_type].get("preference", 0)
        ):
            self.__dict__[dev_type] = {k: v for k, v in definition.items()}
            if not non_serial:
                self.__dict__[dev_type]["port"] = dev.device
            else:
                self.__dict__[dev_type]["port"] = None  # USB-only devices like JS220

    def _instantiate(self, definition):
        args = {}
        if "port" in definition:
            args["port_device"] = definition["port"]
        if definition.get("baud"):
            if definition.get("type") == "dut":
                args["baud_rate"] = get_baud_rate(definition, self.mode)
            else:
                args["baud_rate"] = definition.get("baud")
        if definition.get("echo"):
            args["echo"] = definition["echo"]
        if definition.get("voltage"):
            args["voltage"] = definition["voltage"]

        dev_type = definition.get("type")
        if dev_type == "interface":
            if self._desired_baud:
                args["dut_baud_rate"] = self._desired_baud
            definition["instance"] = (
                IOManagerEnhanced(**args) if definition.get("name") == "stm32h573i-dk" else IOManager(**args)
            )
        elif dev_type == "power":
            if definition.get("name") == "js220":
                from joulescope import scan
                devices = scan()
                if not devices:
                    raise RuntimeError("No JS220 found during instantiation")
                js_device = devices[0].open()
                args["js_device"] = js_device  # 👈 Pass to JS220Commands via PowerManager
            definition["instance"] = PowerManager(definition["name"], config=definition, **args)
        elif dev_type == "dut":
            definition["instance"] = DUT(**args)


    def scan(self):
        """Scan for both serial and USB-only devices and initialize them."""
        pending_serial = [p for p in list_ports.comports(True) if p.vid]
        matched = []

        for p in pending_serial:
            for d in self._device_defs:
                found = False
                for vid, pids in d.get("usb", {}).items():
                    for pid in (pids if isinstance(pids, list) else [pids]):
                        if pid == p.pid and vid == p.vid and precheck_device_name(d, p.device, self.mode):
                            self._add_device(p, d)
                            matched.append(p)
                            found = True
                            break
                if found:
                    break

        for p in (a for a in pending_serial if a not in matched):
            for d in (d1 for d1 in self._device_defs if d1.get("usb_description", "zZzZzZzZ") in p.description):
                self._add_device(p, d)
                matched.append(p)
                break

        # Additional scan for USB-only devices (non-serial)
        all_usb = usb.core.find(find_all=True)
        for dev in all_usb:
            vid = dev.idVendor
            pid = dev.idProduct

            for d in self._device_defs:
                if d.get("interface", "") != "direct_usb":
                    # this association logic is only for direct (non-serial) devices, like the JS-220.
                    # so skip it if interface is unspecified or not "direct_usb"
                    # Without this block, a VID/PID match that has been previously rejected based on
                    # "name" mismatch can be incorrectly associated here.
                    continue
                for k, v in d.get("usb", {}).items():
                    if isinstance(v, list):
                        if pid in v and vid == k:
                            self._add_device(dev, d, non_serial=True)
                    elif pid == v and vid == k:
                        self._add_device(dev, d, non_serial=True)

        for d in self.values():
            self._instantiate(d)
