import argparse, os, time, yaml, json, io
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import Counter
import matplotlib.pyplot as plt

from power_manager.power_manager import PowerManager
from datasets import DataSet, StreamingDataSet

from device_manager import DeviceManager
from device_under_test import DUT
from script import Script, log_filename
import streaming_ww_utils as sww_util
from runner_utils import get_baud_rate, print_tee

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

def identify_dut(manager, desired_baud):
    power = manager.get("power", {}).get("instance")
    interface = manager.get("interface", {}).get("instance")

    # Step 2: Instantiate DUT and initialize it
    if not manager.get("dut") and interface:
        dut = DUT(interface, baud_rate = desired_baud, power_manager=power)
        manager["dut"] = {"instance": dut}
    else:
        dut = manager.get("dut", {}).get("instance")

    init_dut(dut)


def run_test(devices_config, dut_config, test_script, dataset_path,mode):
    """Run the test

    :param devices_config:
    :param dut_config:
    :param test_script:
    :param dataset_path:
    """
    
    desired_dut_baud = get_baud_rate(dut_config, mode)
    manager = DeviceManager(devices_config, desired_dut_baud, mode)
    manager.scan()
    
    if not any([d['type'] == 'interface' for d in manager.values()]) and not any([d['type'] == 'dut' for d in manager.values()]):
       raise RuntimeError("No interface or DUT detected. One must be present to run test")

    power = manager.get("power", {}).get("instance")
    if mode == "e" and power is None:
        raise RuntimeError("ERROR: Energy mode selected but no power board was found.")
    
    if power and dut_config and dut_config.get("voltage"):
        power.configure_voltage(dut_config["voltage"])
    if power:
        power.power_on()
        time.sleep(1) # let the DUT boot up
        power.start() # start recording current measurements

    io = manager.get("interface", {}).get("instance")
    if io:
        io.__enter__()
        io.sync_baud(desired_dut_baud)
    
    identify_dut(manager, desired_dut_baud)
    dut = manager.get("dut", {}).get("instance")
    dut_config['model'] = dut.get_model()
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
    if io:
        io.__exit__()

    if not isinstance(result, list):
        # if the test does not contain a loop, then result will just be a dict
        # rather than a list of dicts.  So make it a list
        result = [result]  

    for r in result:
        if 'power' in r:
            r['power']['voltage'] = power._voltage
    
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


def print_energy_results(l_results, energy_sampling_freq=1000, req_cycles=5, results_file=None):
    # Make sure it has a line that matches this regex
    #   m = re.match(r".* Median energy cost is ([\d\.]+) uJ/inf\..*", line)
    if results_file:
        results_dir = os.path.dirname(results_file)
    else:
        results_dir = os.getcwd()

    inf_energies = np.nan*np.zeros(len(l_results))
    inf_times = np.nan*np.zeros(len(l_results))
    

    for inf_num,res in enumerate(l_results):
        plt.clf()
        energy_samples = res['power']['samples'] # assume 'output type' is set to energy
        
        # timestamps (or 'events' per LPM01a wording are recorded as 
        # (counter, num_samples) where
        # counter: 0,1,.. for the 1st, 2nd, etc, to test if you lost a timestamp
        # num_samples: number of samples captured when the timestamp happened
        ts_counters = [int(ts[0]) for ts in res['power']['timestamps']]
        ts_samp_nums = np.array([int(ts[1]) for ts in res['power']['timestamps']])
        ts_seconds = ts_samp_nums/energy_sampling_freq
        
        if np.any(np.diff(ts_counters) != 1):
            raise RuntimeError(f"Timestamps are not consecutive:\n{res['power']['timestamps']}")

        t_implicit = np.arange(len(energy_samples))/energy_sampling_freq
        plt.plot(t_implicit, energy_samples, 'b')
        y_min = np.min(energy_samples)
        y_max = np.max(energy_samples)
        
        for ts in ts_seconds:  
            plt.plot([ts, ts], [y_min, y_max], 'r')
            plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"energy_inf_{inf_num:03d}.png"))
        
        # There is sometimes some extra activity on the timestamp pin at the 
        # beginning, so take the last two
        t_start = ts_seconds[-2]
        t_stop = ts_seconds[-1]
        idx_start = ts_samp_nums[-2]
        idx_stop = ts_samp_nums[-1]

        elapsed_time = t_stop-t_start
        inference_energy_samples = np.array(energy_samples[idx_start:idx_stop])
        total_inference_energy = np.sum(inference_energy_samples)
        num_inferences = res['infer']['iterations']
        energy_per_inf = total_inference_energy / num_inferences
        latency_per_inf = elapsed_time / num_inferences
        inf_energies[inf_num] = energy_per_inf
        inf_times[inf_num] = elapsed_time
        
        time_warning = "" if elapsed_time >= 10.0 else "  <<<< below minimum duration"
        print_tee(f"Energy data for trial {inf_num}", outfile=results_file)
        print_tee(f"  Elapsed time  : {elapsed_time}{time_warning}", outfile=results_file)
        print_tee(f"  Inferences    : {num_inferences}", outfile=results_file)
        print_tee(f"  Power         : {1e3*total_inference_energy/elapsed_time:5.4f} mW.", outfile=results_file)
        print_tee(f"  Total Energy  : {total_inference_energy*1e6:.3f} uJ", outfile=results_file)
        print_tee(f"  Energy/Inf    : {energy_per_inf*1e6:.3f} uJ/inf", outfile=results_file)
    
    print_tee("---------------------------------------------------------", outfile=results_file)
    print_tee(f"Median energy cost is {1e6*np.median(inf_energies):5.4f} uJ/inf.", outfile=results_file)
    print_tee("---------------------------------------------------------", outfile=results_file)

    if np.any(inf_times<10.0):
        print_tee(f"ERROR: Not valid for submission.  All inference times must be at least 10 seconds.")
    if len(l_results) != req_cycles:
        print_tee(f"ERROR: Not valid for submission.  Energy mode must include exactly {req_cycles} measurements.")

# Summarize results
def summarize_result(result, power, mode, results_file=None):
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


    if power is not None:  # If power is present, turn it off.  
        # this should really be somewhere else
        power.power_off()
        power.__exit__() # fix this so it only looks for 'ack stop', in case sampling has already stopped

    
    if mode == "e":
        print_tee("Power Edition Output", outfile=results_file)
        print_energy_results(result, energy_sampling_freq=1000, results_file=results_file)
        return

    for r in result:
        if 'infer' not in r or 'class' not in r or 'file' not in r:
            continue  # Skip malformed or error-only entries
        infer_data = r['infer']
        infer_results = infer_data['results']
        file_name = r['file']
        true_class = int(r['class'])

        errors = r.get('error')  # Safe access
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

    if throughput_values:  # <-- NEW: Performance mode detected
        has_error_1 = any(r.get("error") == "error 1" for r in result)
        has_error_2 = any(r.get("error") == "error 2" for r in result)
        if has_error_1:
            print_tee(f"{formatted_time}ulp-mlperf: ERROR 1 - loop_count was not exactly 5.", outfile=results_file)
        elif has_error_2:
            print_tee(f"{formatted_time}ulp-mlperf: ERROR 2 - loop exited before 10 seconds elapsed.", outfile=results_file)
        else:
            median_throughput = np.median(throughput_values)
            print_tee(f"{formatted_time}ulp-mlperf: ---------------------------------------------------------", outfile=results_file)
            print_tee(f"{formatted_time}ulp-mlperf: Median throughput is {median_throughput:>10.3f} inf./sec.", outfile=results_file)
            print_tee(f"{formatted_time}ulp-mlperf: ---------------------------------------------------------", outfile=results_file)

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

        accuracy = calculate_accuracy(np.array(y_pred), np.array(y_true))

        if np.array(y_pred).shape[1] == 2:
            auc =roc_auc_score(np.array(y_true), np.array(y_pred)[:,1])
        else:
            auc =roc_auc_score(np.array(y_true), np.array(y_pred), multi_class='ovr')
        
        current_time = datetime.now()
        formatted_time = current_time.strftime("%m%d.%H%M%S ") 
        print_tee(f"{formatted_time}ulp-mlperf: Top 1% = {accuracy:2.1f}", outfile=results_file)
        print_tee(f"{formatted_time}ulp-mlperf: AUC = {auc:.3f}", outfile=results_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="TestRunner", description=__doc__)
    parser.add_argument("-d", "--device_list", default="devices.yaml", help="Device definition YAML file")
    parser.add_argument("-y", "--device_yaml", required=False, help="Raw YAML to interpret as the target device")
    parser.add_argument("-t", "--test_script", default=None, help="File containing test scripts")
    parser.add_argument("-s", "--dataset_path", default="datasets")
    parser.add_argument("-m", "--mode", choices=["e", "p", "a"], default="a", help="Test mode (energy (e), performance (p), accuracy (a))")
    args = parser.parse_args()
    if args.test_script is not None:
        test_script_file=args.test_script
    else:
        if args.mode == "a":
            test_script_file = "tests_accuracy.yaml"  # Accuracy test script
        elif args.mode == "p":
            test_script_file = "tests_performance.yaml"  # Performance test script
        elif args.mode == "e":
            test_script_file = "tests_energy.yaml"  # Energy test script
    config = {
        "devices_config": parse_device_config(args.device_list, args.device_yaml),
        "test_script": parse_test_script(test_script_file),
        "dataset_path": args.dataset_path,
        "mode": args.mode
    }
    # copy any DUT from the devices_config into the dut_config.  The last DUT listed will 
    # be the one we use.  This means that internally we have the DUT info in two 
    # places (config["devices_config"][i] for some i, and dut_config), but at least
    # externally the DUT info is only in one place
    for dev in config["devices_config"]:
        if dev["type"] == "dut":
            config["dut_config"] = dev

    result, power = run_test(**config)  # Unpack power from run_test
    if isinstance(result, dict): # this is a hack. make the run_test outputs consistent
        result = [result]
    for r in result: 
        r["mode"] = config["mode"]
    results_file = os.path.join(os.path.dirname(log_filename), "results.txt")
    if config['dut_config']['model'] == 'sww01':
        if power is not None:  # If power is present, turn it off.  
            # this should really be somewhere else
            power.power_off()
            power.__exit__() # fix this so it only looks for 'ack stop', in case sampling has already stopped
        if config["mode"] == "e":
            print_tee("Power Edition Output", outfile=results_file)
            print_energy_results(result, energy_sampling_freq=1000, req_cycles=1, results_file=results_file)
        sww_util.summarize_sww_result(result, power, results_file=results_file)
    else:
        summarize_result(result, power, mode=config["mode"], results_file=results_file)

    print(f"Session logged in file {log_filename}")
    results_data_file = os.path.join(os.path.dirname(log_filename), "results.json")
    with open(results_data_file, 'w') as fpo:
        json.dump(result, fpo)