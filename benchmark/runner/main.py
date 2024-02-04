import argparse
import csv
import time

from device_manager import DeviceManager
from device_under_test import DUT

from io_manager_enhanced import IOManagerEnhanced
from power_manager import PowerManager
from io_manager import IOManager

from contextlib import nullcontext, ExitStack
from serial.tools import list_ports
import yaml


def test_power(device):
  with device as power:
    pass


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
  if device:
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


def init_dut(device):
  with device as dut:
    time.sleep(2)
    dut.get_name()
    dut.get_model()
    dut.get_profile()

def identify_dut(manager):
  interface = manager.get("interface", {}).get("instance")
  power = manager.get("power", {}).get("instance")
  if not manager.get("dut") and interface and power:
    dut = DUT(interface, power_manager=power)
    manager["dut"] = {
        "instance": dut
    }
  else:
    dut = manager.get("dut", {}).get("instance")
  init_dut(dut)


def run_test(devices_config, dut_config, test_script):
  manager = DeviceManager(devices_config)
  manager.scan()
  if manager.get("power", {}).get("instance") and dut_config and dut_config.get("voltage"):
    manager["power"]["instance"].configure_voltage(dut_config["voltage"])
  identify_dut(manager)
  dut = manager.get("dut", {}).get("instance")
  test = test_script.get(dut.get_model())
  with open("/Users/sreckamp/eembc/runner/benchmarks/ulp-mlperf/datasets/ic01/y_labels.csv") as file:
    reader = csv.DictReader(file, fieldnames=["file", "classes", "class"])
    file_name = reader.__next__()["file"]
  with open(f"/Users/sreckamp/eembc/runner/benchmarks/ulp-mlperf/datasets/ic01/{file_name}", mode="rb") as file:
    data = file.read()
  with dut:
    dut.load(data)
    print(dut.infer(10, 1))
  # for _def, i in ((t, t.get("instance")) for t in manager.values() if t and t.get("instance")):
  #   if isinstance(i, PowerManager):
  #     test_power(i)
  #   elif isinstance(i, IOManagerEnhanced):
  #     test_io_manager_enhanced(i)
  #   elif isinstance(i, IOManager):
  #     test_io_manager(i)
  #   elif isinstance(i, DUT):
  #     test_dut(i)


def parse_device_config(device_list, device_yaml):
  if device_yaml:
    return yaml.load(device_yaml)
  else:
    with open(device_list) as dev_file:
      return yaml.load(dev_file, Loader=yaml.CLoader)


def parse_dut_config(dut, dut_voltage, dut_baud):
  config = {}
  if dut:
    with open(dut) as dut_file:
      dut_config = yaml.load(dut_file, Loader=yaml.CLoader)
      config.update(**dut_config)
  if dut_voltage:
    config.update(voltage=dut_voltage)
  if dut_baud:
    config.update(baud=dut_baud)
  return config


def parse_test_script(test_script):
  with open(test_script) as test_file:
    return yaml.load(test_file, Loader=yaml.CLoader)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog="TestRunner")
  parser.add_argument("-d", "--device_list", default="devices.yaml")
  parser.add_argument("-y", "--device_yaml", required=False)
  parser.add_argument("-u", "--dut", required=False)
  parser.add_argument("-v", "--dut_voltage", required=False)
  parser.add_argument("-b", "--dut_baud", required=False)
  parser.add_argument("-t", "--test_script", default="tests.yaml")
  args = parser.parse_args()
  config = {
    "devices_config": parse_device_config(args.device_list, args.device_yaml),
    "dut_config": parse_dut_config(args.dut, args.dut_voltage, args.dut_baud),
    "test_script": parse_test_script(args.test_script)
  }
  run_test(**config)
