import argparse
import os
import time
import yaml

from datasets import DataSet
from device_manager import DeviceManager
from device_under_test import DUT
from script import Script


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


def run_test(devices_config, dut_config, test_script, dataset_path):
  manager = DeviceManager(devices_config)
  manager.scan()
  power = manager.get("power", {}).get("instance")
  if power and dut_config and dut_config.get("voltage"):
    power.configure_voltage(dut_config["voltage"])
  identify_dut(manager)

  dut = manager.get("dut", {}).get("instance")
  io = manager.get("interface", {}).get("instance")

  script = Script(test_script.get(dut.get_model()))
  set = DataSet(os.path.join(dataset_path, script.model), script.truth)

  return script.run(io, dut, set)


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
  parser.add_argument("-s", "--dataset_path", default="datasets")
  args = parser.parse_args()
  config = {
    "devices_config": parse_device_config(args.device_list, args.device_yaml),
    "dut_config": parse_dut_config(args.dut, args.dut_voltage, args.dut_baud),
    "test_script": parse_test_script(args.test_script),
    "dataset_path": args.dataset_path
  }
  print(run_test(**config))
