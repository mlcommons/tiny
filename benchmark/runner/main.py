import argparse
import os
import time
import yaml
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import Counter

from power_manager import PowerManager
from datasets import DataSet, StreamingDataSet

from device_manager import DeviceManager
from device_under_test import DUT
from script import Script
import streaming_ww_utils as sww_util

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
    power = manager.get("power", {}).get("instance")
    interface = manager.get("interface", {}).get("instance")
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
    print(f"Power instance: {power}")

    if power:
        mode = "Energy"
    else:
        # Ask user for mode when no power board is detected
        while True:
            user_input = input("No power board detected. Enter mode [P]erformance or [A]ccuracy: ").strip().upper()
            if user_input == "P":
                mode = "Performance"
                break
            elif user_input == "A":
                mode = "Accuracy"
                break
            else:
                print("Invalid input. Please enter 'P' or 'A'.")

    print(f"Running in {mode} mode")
    if power and dut_config and dut_config.get("voltage"):
        power.configure_voltage(dut_config["voltage"])
    identify_dut(manager)
    
    dut = manager.get("dut", {}).get("instance")
    dut_config['model'] = dut.get_model()
    io = manager.get("interface", {}).get("instance")

    # with io:
    #   start_time = time.time()
    #   io.play_wave("cd16m.wav")
    #   elapsed = time.time() - start_time

    script = Script(test_script.get(dut.get_model()))
    if script.model[:3] == 'sww':
        set = StreamingDataSet(os.path.join(dataset_path, script.model), script.truth)
    else:
        set = DataSet(os.path.join(dataset_path, script.model), script.truth)
    result = script.run(io, dut, set, mode)
    return result, power

def parse_device_config(device_list_file, device_yaml):
    """Parse the device discovery configuration

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

def calculate_accuracy(y_pred, labels):
    # Check if y_pred has only one value per instance (Anomaly Detection case)
    if y_pred.shape[1] == 1:
        thresholds = np.amin(y_pred) + np.arange(0.0, 1.0, .01) * (np.amax(y_pred) - np.amin(y_pred))
        accuracy = 0
        n_normal = np.sum(labels == 0)  # Assuming normal instances are labeled as 0

        for threshold in thresholds:
            y_pred_binary = (y_pred > threshold).astype(int)  # Binarization

            true_negative = np.sum((y_pred_binary[labels == 0] == 0))
            false_positive = np.sum((y_pred_binary[labels == 0] == 1))
            true_positive = np.sum((y_pred_binary[labels == 1] == 1))
            false_negative = np.sum((y_pred_binary[labels == 1] == 0))

            precision = true_positive / (true_positive + false_positive + 1e-8)  # Avoid division by zero
            recall = true_positive / (true_positive + false_negative + 1e-8)

            accuracy_tmp = 100 * (precision + recall) / 2  # F1-like accuracy estimation
            accuracy = max(accuracy, accuracy_tmp)
        return accuracy

    # Normal Classification Case
    y_pred_label = np.argmax(y_pred, axis=1)
    correct = np.sum(labels == y_pred_label)
    accuracy = 100 * correct / len(y_pred)
    return accuracy


def calculate_auc(y_pred, labels, n_classes):
    # Check if y_pred has only one value per instance (Anomaly Detection case)
    if y_pred.shape[1] == 1:
        thresholds = np.amin(y_pred) + np.arange(0.0, 1.01, .01) * (np.amax(y_pred) - np.amin(y_pred))
        roc_auc = 0

        n_normal = np.sum(labels == 0)  # Assuming normal instances are labeled as 0
        tpr = np.zeros(len(thresholds))
        fpr = np.zeros(len(thresholds))

        for threshold_item in range(1, len(thresholds)):
            threshold = thresholds[threshold_item]
            y_pred_binary = (y_pred > threshold).astype(int)

            tpr[threshold_item] = np.sum(y_pred_binary[labels == 1]) / float(np.sum(labels == 1))
            fpr[threshold_item] = np.sum(y_pred_binary[labels == 0]) / float(n_normal)

        # Force boundary condition
        fpr[0] = 1
        tpr[0] = 1

        # Compute AUC using trapezoidal rule
        for threshold_item in range(len(thresholds) - 1):
            roc_auc += 0.5 * (tpr[threshold_item] + tpr[threshold_item + 1]) * (
                        fpr[threshold_item] - fpr[threshold_item + 1])
        return roc_auc

    # Multiclass Case (Existing Logic)
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
            if all_negatives == 0:
                fpr[class_item, threshold_item] = 0
            else:
                fpr[class_item, threshold_item] = false_positives / float(all_negatives)
            if all_positives == 0:
                tpr[class_item, threshold_item] = 0
            else:                    
                tpr[class_item, threshold_item] = true_positives / float(all_positives)

        fpr[class_item, 0] = 1
        tpr[class_item, 0] = 1
        for threshold_item in range(len(thresholds) - 1):
            roc_auc[class_item] += 0.5 * (tpr[class_item, threshold_item] + tpr[class_item, threshold_item + 1]) * (
                        fpr[class_item, threshold_item] - fpr[class_item, threshold_item + 1])

    roc_auc_avg = np.mean(roc_auc)
    return roc_auc_avg

# Summarize results
def summarize_result(result, power):
    num_correct_files = 0
    total_files = 0
    y_pred = []
    y_true = []
    throughput_values = []

    file_infer_results = {}
    current_time = datetime.now()
    formatted_time = current_time.strftime("%m%d.%H%M%S ") 

    # Extract all unique classes
    all_classes = sorted(list(set(int(r['class']) for r in result if 'class' in r)))
    n_classes = len(all_classes)

    for r in result:
        if 'infer' not in r or 'class' not in r or 'file' not in r:
            continue  # Skip malformed or error-only entries
        infer_data = r['infer']
        infer_results = infer_data['results']
        file_name = r['file']
        true_class = int(r['class'])
        for r in result:
            errors = r.get('error')  # ✅ Safe access
            if errors:
                continue  # Skip error entries entirely here
        
        if 'throughput' in infer_data:
            throughput_values.append(infer_data['throughput'])

        if file_name not in file_infer_results:
            file_infer_results[file_name] = {'true_class': true_class, 'results': []}

        if len(infer_results) == 1:
            file_infer_results[file_name]['results'].append(infer_results)
        else:
            infer_results = normalize_probabilities(infer_results)
            file_infer_results[file_name]['results'].append(infer_results)

    if power is not None:  # If power is present, we're in energy mode
        print("Power Edition Output")
        # power.stop()  # Stop power capture
        # power.send_command_wait_for_response("pwr off")
        # print("after stop")
        power.power_off()
        # print("after power_off")
        power.__exit__() # fix this so it only looks for 'ack stop'
        print("after __exit__")
    elif throughput_values:  # <-- NEW: Performance mode detected
        has_error_1 = any(r.get("error") == "error 1" for r in result)
        has_error_2 = any(r.get("error") == "error 2" for r in result)
        if has_error_1:
            print(f"{formatted_time}ulp-mlperf: ERROR 1 – loop_count was not exactly 5.")
        elif has_error_2:
            print(f"{formatted_time}ulp-mlperf: ERROR 2 – loop exited before 10 seconds elapsed.")
        else:
            median_throughput = np.median(throughput_values)
            print(f"{formatted_time}ulp-mlperf: ---------------------------------------------------------")
            print(f"{formatted_time}ulp-mlperf: Median throughput is {median_throughput:>10.3f} inf./sec.")
            print(f"{formatted_time}ulp-mlperf: ---------------------------------------------------------")

    else:
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

        auc = calculate_accuracy(np.array(y_pred), np.array(y_true))
        accuracy = calculate_auc(np.array(y_pred), np.array(y_true), n_classes)
        current_time = datetime.now()
        formatted_time = current_time.strftime("%m%d.%H%M%S ") 
        print(f"{formatted_time}ulp-mlperf: Top 1% = {accuracy:2.1f}")
        print(f"{formatted_time}ulp-mlperf: AUC = {auc:.3f}")
    
        

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
        "dataset_path": args.dataset_path,
    }
    result, power = run_test(**config)  # Unpack power from run_test
    if config['dut_config']['model'] == 'sww01':
        sww_util.summarize_sww_result(result, power)  # Pass only power
    else:
        summarize_result(result, power)  # Remove mode parameter

    