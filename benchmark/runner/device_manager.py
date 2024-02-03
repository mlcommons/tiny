import yaml

from serial.tools import list_ports

from device_under_test import DUT
from io_manager import IOManager
from io_manager_enhanced import IOManagerEnhanced
from power_manager import PowerManager


class DeviceManager:
  def __init__(self, devices):
    self._device_defs = devices

  def __getitem__(self, item):
    return self.__dict__[item]

  def __setitem__(self, key, value):
    self.__dict__[key] = value

  def get(self, item, default=None):
    return self.__dict__.get(item, default)

  def values(self):
    return (a for a in self.__dict__.values() if isinstance(a, dict))

  def _add_device(self, usb, definition):
    type = definition.get("type")
    if type and (not self.__dict__.get(type)
                 or definition.get("preference", 0) > self.__dict__[type].get("preference", 0)):
      self.__dict__[type] = {k: v for k, v in definition.items()}
      self.__dict__[type]["port"] = usb.device

  def _instantiate(self, definition):
    args = {
      "port_device": definition.get("port")
    }
    if definition.get("baud"):
      args["baud_rate"] = definition.get("baud")
    if definition.get("type") == "interface":
      definition["instance"] = IOManagerEnhanced(**args) if definition.get("name") == "stm32h573i-dk" \
        else IOManager(**args)
    elif definition.get("type") == "power":
      definition["instance"] = PowerManager(**args)
    elif definition.get("type") == "dut":
      definition["instance"] = DUT(**args)

  def scan(self):
    """Scan fpr USB serial devices
    This scans the connected usb devices.  It compares the device definitions
    based on the USB vid and pid primarily and then by the text of the description.
    """
    pending = [p for p in list_ports.comports(True) if p.vid]
    matched = []
    for p in pending:
      for d in self._device_defs:
        found = False
        for vid, pids in d.get("usb", {}).items():
          for pid in (pids if isinstance(pids, list) else [pids]):
            if pid == p.pid and vid == p.vid:
              self._add_device(p, d)
              matched.append(p)
              found = True
              break
        if found: break
    for p in (a for a in pending if a not in matched):
      for d in (d1 for d1 in self._device_defs if d1.get("usb_description", "zZzZzZzZ") in p.description):
        self._add_device(p, d)
        matched.append(p)
        break

    for d in self.values():
      self._instantiate(d)
