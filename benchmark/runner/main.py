import argparse
import os
import time
import yaml
import numpy as np
from sklearn.metrics import roc_auc_score

from datasets import DataSet
from device_manager import DeviceManager
from device_under_test import DUT
from script import Script

"""
Application to execute test scripts to measure power consumption, turn on and off power, send commands to a device
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


def run_test(devices_config, dut_config, test_script, dataset_path, mode):
    """Run the test

    :param devices_config:
    :param dut_config:
    :param test_script:
    :param dataset_path:
    :param mode: Test mode (energy, performance, accuracy)
    """
    manager = DeviceManager(devices_config)
    manager.scan()
    power = manager.get("power", {}).get("instance")
    if power and dut_config and dut_config.get("voltage"):
        power.configure_voltage(dut_config["voltage"])
    identify_dut(manager)  # hangs in identify_dut()=>init_dut()=>time.sleep()

    dut = manager.get("dut", {}).get("instance")
    io = manager.get("interface", {}).get("instance")

    # Create a Script object from the dict that was read from the tests yaml file.
    script = Script(test_script.get(dut.get_model()))

    # Run the test script with the mode passed in
    data_set = DataSet(os.path.join(dataset_path, script.model), script.truth)
    return script.run(io, dut, data_set, mode)  # Pass mode to the run method


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

def normalize_probabilities(probabilities):
    """Normalize probabilities to ensure they sum to 1.0, no matter the initial sum."""
    probabilities = np.array(probabilities)
    sum_probs = np.sum(probabilities)
    
    # Ensure probabilities sum exactly to 1
    if sum_probs == 0:
        raise ValueError("Sum of probabilities is zero, cannot normalize.")
    
    probabilities = probabilities / sum_probs  # Normalize probabilities to make sum = 1
    
    return probabilities

def summarize_result(result):
    num_correct = 0  # Initialize the counter for correct predictions
    
    # Store true labels and predicted probabilities for AUC calculation
    true_labels = []
    predicted_probabilities = []

    for r in result:
        # Normalize the predicted probabilities
        infer_results = r['infer']['results']
        infer_results = normalize_probabilities(infer_results)  # Ensure this is a valid probability distribution

        # Add to the list for AUC calculation
        true_labels.append(int(r['class']))
        predicted_probabilities.append(infer_results)

        # Compare predicted class with the true class
        if np.argmax(infer_results) == int(r['class']):
            num_correct += 1  # Increment if the prediction matches
    
    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    predicted_probabilities = np.array(predicted_probabilities)

    # Check if binary classification (i.e., only two unique classes)
    if len(np.unique(true_labels)) == 2:
        # For binary classification, use the probability of class 1
        predicted_probabilities = predicted_probabilities[:, 1].reshape(-1, 1)  # Use class 1 probability

        # Calculate accuracy
        accuracy = num_correct / len(result)
        print(f"Accuracy = {num_correct}/{len(result)} = {100*accuracy:4.2f}%")   

        # Compute AUC for binary classification (class 1)
        try:
            auc_score = roc_auc_score(true_labels, predicted_probabilities)
            print(f"AUC: {auc_score:.4f}")
        except ValueError as e:
            print(f"AUC calculation failed: {e}")
    
    else:
        # For multi-class classification
        # Dynamically handle the number of classes based on the unique values in true_labels
        unique_classes = np.unique(true_labels)
        num_classes = len(unique_classes)

        # Adjust the probabilities to match the number of classes in true_labels
        predicted_probabilities = np.array([prob[:num_classes] for prob in predicted_probabilities])

        # Calculate accuracy
        accuracy = num_correct / len(result)
        print(f"Accuracy = {num_correct}/{len(result)} = {100*accuracy:4.2f}%")   

        # Compute AUC for multi-class classification using one-vs-rest (macro-average AUC)
        try:
            auc_score = roc_auc_score(true_labels, predicted_probabilities, multi_class="ovr", average="macro")
            print(f"Macro-average AUC: {auc_score:.4f}")
        except ValueError as e:
            print(f"AUC calculation failed: {e}")
              
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="TestRunner", description=__doc__)
    parser.add_argument("-d", "--device_list", default="devices.yaml", help="Device definition YAML file")
    parser.add_argument("-y", "--device_yaml", required=False, help="Raw YAML to interpret as the target device")
    parser.add_argument("-u", "--dut_config", required=False, help="Target device")
    parser.add_argument("-v", "--dut_voltage", required=False, help="Voltage set during test")
    parser.add_argument("-b", "--dut_baud", required=False, help="Baud rate for device under test")
    parser.add_argument("-t", "--test_script", default="tests.yaml", help="File containing test scripts")
    parser.add_argument("-s", "--dataset_path", default="datasets")
    parser.add_argument("-m", "--mode", choices=["e", "p", "a"], default="a", help="Test mode (energy (e), performance (p), accuracy (a))")
    args = parser.parse_args()
    config = {
        "devices_config": parse_device_config(args.device_list, args.device_yaml),
        "dut_config": parse_dut_config(args.dut_config, args.dut_voltage, args.dut_baud),
        "test_script": parse_test_script(args.test_script),
        "dataset_path": args.dataset_path,
        "mode": args.mode
    }
    result = run_test(**config)
    summarize_result(result)