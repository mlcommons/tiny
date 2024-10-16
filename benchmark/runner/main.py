import argparse
import os
import time
import yaml

from datasets import DataSet
from device_manager import DeviceManager
from device_under_test import DUT
from script import Script

"""
Application to execute test scripts to measure power consumption, turn on amd off power, send commands to a device
under test.
"""


def init_dut(device):
  if device:
    with device as dut:
      time.sleep(2)
      dut.get_name()
      dut.get_model()
      dut.get_profile()

def identify_dut(manager):
  interface = manager.get("interface", {}).get("instance")
  power = manager.get("power", {}).get("instance")
  if not manager.get("dut") and interface: # removed and power:
    dut = DUT(interface, power_manager=power)
    manager["dut"] = {
        "instance": dut
    }
  else:
    dut = manager.get("dut", {}).get("instance")
  init_dut(dut)


def run_test(devices_config, dut_config, test_script, dataset_path):
  """Run the test

  :param devices_config:
  :param dut_config:
  :param test_script:
  :param dataset_path:
  """
  manager = DeviceManager(devices_config)
  manager.scan()
  power = manager.get("power", {}).get("instance")
  if power and dut_config and dut_config.get("voltage"):
    power.configure_voltage(dut_config["voltage"])
  identify_dut(manager) # hangs in identify_dut()=>init_dut()=>time.sleep()

  dut = manager.get("dut", {}).get("instance")
  io = manager.get("interface", {}).get("instance")

  # with io:
  #   start_time = time.time()
  #   io.play_wave("cd16m.wav")
  #   elapsed = time.time() - start_time
  # create a Script object from the dict that was read from the tests yaml file.
  script = Script(test_script.get(dut.get_model()))
  data_set = DataSet(os.path.join(dataset_path, script.model), script.truth)

  return script.run(io, dut, data_set)


def parse_device_config(device_list_file, device_yaml):
  """Parsee the device discovery configuration

  :param device_list: device discovery configuration file
  :param device_yaml: device description as raw yaml
  """
  if device_yaml:
    return yaml.load(device_yaml)
  else:
    with open(device_list_file) as dev_file:
      return yaml.load(dev_file, Loader=yaml.CLoader)


def parse_dut_config(dut_cfg_file, dut_voltage, dut_baud):
  """ Parse the dut configuration file and override values

  :param dut: path to device config file
  :param dut_voltage: dut voltage in mV
  :param dut_baud: dut baud rate
  """
  config = {}
  if dut_cfg_file:
    with open(dut_cfg_file) as dut_file:
      dut_config = yaml.load(dut_file, Loader=yaml.CLoader)
      config.update(**dut_config)
  if dut_voltage:
    config.update(voltage=dut_voltage)
  if dut_baud:
    config.update(baud=dut_baud)
  return config


def parse_test_script(test_script):
  """Load the test script

  :param test_script: The path to the test script definition
  """
  with open(test_script) as test_file:
    return yaml.load(test_file, Loader=yaml.CLoader)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog="TestRunner", description=__doc__)
  parser.add_argument("-d", "--device_list", default="devices.yaml", help="Device definition YAML file")
  parser.add_argument("-y", "--device_yaml", required=False, help="Raw YAML to interpret as the target device")
  parser.add_argument("-u", "--dut_config", required=False, help="Target device")
  parser.add_argument("-v", "--dut_voltage", required=False, help="Voltage set during test")
  parser.add_argument("-b", "--dut_baud", required=False, help="Baud rate for device under test")
  parser.add_argument("-t", "--test_script", default="tests.yaml", help="File containing test scripts")
  parser.add_argument("-s", "--dataset_path", default="datasets")
  args = parser.parse_args()
  config = {
    "devices_config": parse_device_config(args.device_list, args.device_yaml),
    "dut_config": parse_dut_config(args.dut_config, args.dut_voltage, args.dut_baud),
    "test_script": parse_test_script(args.test_script),
    "dataset_path": args.dataset_path
  }
  result = run_test(**config)
  print(result)
