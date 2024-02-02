import argparse

from device_under_test import DUT

from io_manager_enhanced import IOManagerEnhanced
from power_manager import PowerManager
from io_manager import IOManager

from serial.tools import list_ports
import yaml


def test_power(device):
  with device as power:
      for line in power.get_help():
          print(line)
      # power.acquire(500)


def run_dut_test():
  """
  power on
  m-ready
  -mode energy
  imit-done
  m-ready
  :return:
  dut passthrough: profile
  """

def test_dut(device):
    with device as dut:
        print(dut.get_name())
        print(dut.get_model())
        print(dut.get_profile())
        print(dut.timestamp())
        for l in dut.get_help():
            print(l)


def test_io_manager(device):
  with device as io:
      print(io.get_name())
      for l in io.get_help():
          print(l)


def test_io_manager_enhanced(device):
    with device as io:
        print(io.get_name())
        print(io.get_waves())
        print(io.play_wave())
        print(io.play_wave("spaceship.wav"))
        # waves = io.list_waves()
        # io.play_wave();
        # for w in waves:
        #     print(w)
        #     # io.play_wave(w)


def scan_devices(devices=None):
    pending = [p for p in list_ports.comports(True) if p.vid]
    matched = []
    for p in pending:
        pass
        for d in (d1 for d1 in devices if d1.get("usb_description", "zZzZzZzZ") in p.description):
            yield p, d
            matched.append(p)
            break
    for p in (a for a in pending if a not in matched):
        for d in devices:
            found = False
            for vid, pids in d.get("usb", {}).items():
                for pid in (pids if isinstance(pids, list) else [pids]):
                    if pid == p.pid and vid == p.vid:
                        yield p, d
                        found =True
                        break
            if found: break


def identify_dut(tools):
  interface = tools.get("interface", {}).get("instance")
  power = tools.get("power", {}).get("instance")
  if not tools.get("dut") and interface and power:
    dut = DUT(interface)
    tools["dut"] = {
        "instance": dut
    }
    # power.on()
  else:
    dut = tools.get("dut", {}).get("instance")
  if dut:
    with dut:
      dut.get_name()
      dut.get_model()
      dut.get_profile()



def instantiate(device):
    args = {
        "port_device": device.get("port")
    }
    if device.get("baud"):
        args["baud_rate"] = device.get("baud")
    if device.get("type") == "interface":
        device["instance"] = IOManagerEnhanced(**args) if device.get("name") == "stm32h573i-dk" \
                                                       else IOManager(**args)
    elif device.get("type") == "power":
        device["instance"] = PowerManager(**args)
    elif device.get("type") == "dut":
        device["instance"] = DUT(**args)


def build_tools(devices):
    tools = { 'power': None, 'interface': None, 'dut': None }
    for port, device in scan_devices(devices):
        type = device.get("type")
        if type and (not tools.get(type) or device.get("preference", 0) > tools.get(type).get("preference", 0)):
            tools[type] = {k: v for k, v in device.items()}
            tools[type]["port"] = port.device

    for d in (a for a in tools.values() if a):
        instantiate(d)
    return tools


def run_test(device_list, dut, test_script):
    with open(device_list) as dev_file:
        devices = yaml.load(dev_file, Loader=yaml.CLoader)
    tools = build_tools(devices)
    identify_dut(tools)
    for i in (t.get("instance") for t in tools.values() if t and t.get("instance")):
        if isinstance(i, PowerManager):
            test_power(i)
        elif isinstance(i, IOManagerEnhanced):
            test_io_manager_enhanced(i)
        elif isinstance(i, IOManager):
            test_io_manager(i)
        elif isinstance(i, DUT):
            test_dut(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="TestRunner")
    parser.add_argument("-d", "--device_list", default="devices.yaml")
    parser.add_argument("-u", "--dut", default="dut.yaml")
    parser.add_argument("-t", "--test_script", default="test.yaml")
    args = parser.parse_args()
    run_test(**args.__dict__)
