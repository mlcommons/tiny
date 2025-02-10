import argparse
import os
import time
import yaml
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import Counter

from power_manager import PowerManager
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
    lpm = None
    if mode == 'e':
        lpm = PowerManager(port="COM19", baud_rate=3864000, print_info_every_ms=1_000)
        lpm.init_device(mode="ascii", voltage=3300, freq=1000, duration=0)
        lpm.start_capture()
        lpm.start_background_parsing()  # Start parsing in the background

    manager = DeviceManager(devices_config)
    manager.scan()
    power = manager.get("power", {}).get("instance")
    print(f"Power instance: {power}")
    
    if power and dut_config and dut_config.get("voltage"):
        power.configure_voltage(dut_config["voltage"])
    identify_dut(manager)

    dut = manager.get("dut", {}).get("instance")
    io = manager.get("interface", {}).get("instance")

    # Pass PowerManager instance to Script
    script = Script(test_script.get(dut.get_model()))

    data_set = DataSet(os.path.join(dataset_path, script.model), script.truth)
    result = script.run(io, dut, data_set, mode)

    return result, lpm


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

# Function to normalize probabilities
def normalize_probabilities(scores):
    total = sum(scores)
    return [s / total for s in scores]

# Function to calculate accuracy
def calculate_accuracy(y_pred, labels):
    # Normalize y_pred
    # Check if y_pred has only one value per instance and transform it
    if y_pred.shape[1] == 1:
        y_pred = (y_pred - y_pred.min(axis=0)) / (y_pred.max(axis=0) - y_pred.min(axis=0) + 1e-10)
        y_pred = np.array([[value[0], 1 - value[0]] for value in y_pred])

    # Get predicted labels and calculate accuracy
    y_pred_label = np.argmax(y_pred, axis=1)
    correct = np.sum(labels == y_pred_label)
    accuracy = 100 * correct / len(y_pred)

    print(f"Overall accuracy = {accuracy:2.1f}")
    return accuracy

# Function to calculate AUC
def calculate_auc(y_pred, labels, n_classes):
    # Normalize y_pred for each class
    if y_pred.shape[1] == 1:
        y_pred = np.array([[value, 1- value] for value in y_pred])
        y_pred = (y_pred - y_pred.min(axis=0)) / (y_pred.max(axis=0) - y_pred.min(axis=0) + 1e-10)
    thresholds = np.arange(0.0, 1.01, 0.01)
    fpr = np.zeros([n_classes, len(thresholds)])
    tpr = np.zeros([n_classes, len(thresholds)])
    roc_auc = np.zeros(n_classes)
    
    for class_item in range(n_classes):
        all_positives = sum(labels == class_item)
        all_negatives = len(labels) - all_positives

        for threshold_item in range(1, len(thresholds)):
            threshold = thresholds[threshold_item]
            false_positives = 0
            true_positives = 0
            for i in range(len(y_pred)):
                if y_pred[i, class_item] > threshold:
                    if labels[i] == class_item:
                        true_positives += 1
                    else:
                        false_positives += 1
            fpr[class_item, threshold_item] = false_positives / float(all_negatives)
            tpr[class_item, threshold_item] = true_positives / float(all_positives)

        fpr[class_item, 0] = 1
        tpr[class_item, 0] = 1
        for threshold_item in range(len(thresholds) - 1):
            roc_auc[class_item] += 0.5 * (tpr[class_item, threshold_item] + tpr[class_item, threshold_item + 1]) * (
                        fpr[class_item, threshold_item] - fpr[class_item, threshold_item + 1])

    roc_auc_avg = np.mean(roc_auc)
    print(f"Simplified average ROC AUC = {roc_auc_avg:.3f}")
    return roc_auc


# Summarize results
def summarize_result(result, mode, lpm=None):
    num_correct_files = 0
    total_files = 0
    y_pred = []
    y_true = []

    file_infer_results = {}

    # Extract all unique classes
    all_classes = sorted(list(set(int(r['class']) for r in result)))
    n_classes = len(all_classes)

    for r in result:
        infer_results = r['infer']['results']
        file_name = r['file']
        true_class = int(r['class'])

        if file_name not in file_infer_results:
            file_infer_results[file_name] = {'true_class': true_class, 'results': []}

        if len(infer_results) == 1:
            file_infer_results[file_name]['results'].append(infer_results)
        else:
            infer_results = normalize_probabilities(infer_results)
            file_infer_results[file_name]['results'].append(infer_results)

    if mode != 'e':
        for file_name, data in file_infer_results.items():
            true_class = data['true_class']
            results = data['results']

            class_counts = Counter([np.argmax(res) for res in results])
            majority_class = class_counts.most_common(1)[0][0]

            if majority_class == true_class:
                num_correct_files += 1

            averaged_result = np.mean(results, axis=0)
            y_pred.append(averaged_result)
            y_true.append(true_class)

            total_files += 1

        calculate_accuracy(np.array(y_pred), np.array(y_true))
        calculate_auc(np.array(y_pred), np.array(y_true), n_classes)
    else:
        # Stop the PowerManager if provided
        if lpm:
            lpm.stop_capture()
            print("Power measurement stopped.")

    return

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
    result, lpm = run_test(**config)
    summarize_result(result, args.mode, lpm=lpm)