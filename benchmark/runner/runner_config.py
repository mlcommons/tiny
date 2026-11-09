"""Parse runner configuration without importing hardware drivers."""
import yaml


def parse_device_config(device_list_file, device_yaml):
    """Parse the device discovery configuration

    :param device_list: device discovery configuration file
    :param device_yaml: device description as raw yaml
    """
    if device_yaml:
        return yaml.safe_load(device_yaml)
    else:
        with open(device_list_file) as dev_file:
            return yaml.safe_load(dev_file)

def parse_test_script(test_script):
    """Load the test script

    :param test_script: The path to the test script definition
    """
    with open(test_script) as test_file:
        return yaml.safe_load(test_file)
