import numpy as np
import re

from runner_utils import print_tee


def array_from_strings(raw_info, header_str, end_str='m-ready', data_type=None):

    ## Combine the strings into one long string then split bracketed substrings
    raw_info = ''.join(raw_info)
    pattern = re.escape(header_str) + r'\s*(\[.*?\])\s*' + re.escape(end_str)
    match = re.search(pattern, raw_info, re.DOTALL)
    if not match:
        raise ValueError(f"Start ({header_str}) or end marker  ({end_str}) expected but not found.")
    raw_info = match.group(1)
    
    # split the long string into substrings based on brackets [].
    bracket_matches = re.findall(r'\[([^\]]+)\]', raw_info)
    array_strings = bracket_matches if bracket_matches else [raw_info]


    ## Figure out what data type is requested
    if data_type is None: # guess based on first value
        first_value = array_strings[0].split(',')[0]    
        if '.' in first_value or 'e' in first_value:
            data_type = converter = float
        else:
            data_type = converter = int
    elif data_type in [int, float]:
        converter = data_type
    elif isinstance(data_type, np.dtype):
        if np.issubdtype(dtype, np.integer):
            converter = int
        else:
            converter = float
    else:
        raise ValueError(f"data_type is {data_type}. Must be either None, int, float, or a numpy dtype")
  

    ## Now extract the data
    number_lists = []
    for s in array_strings:
        number_lists.append([converter(val) for val in s.split(',')])
    
    if len(number_lists) == 1:
        number_lists = number_lists[0] # only 1 array, don't make it 2D

    return number_lists

def process_timestamps(raw_result):
    if raw_result[0] != 'Detection Timestamps (ms)':
        raise ValueError(f"List must start with 'Detection Timestamps (ms)'. First element is {raw_result[0]}")
    if raw_result[-1] != 'm-ready':
        raise ValueError(f"List must end with 'm-ready'. Last element is {raw_result[-1]}")

    try:
        timestamps = [int(value.strip(',')) for value in raw_result[1:-1]]
    except ValueError:
        raise ValueError("One or more values in the list cannot be converted to an integer")
    print(timestamps)
    return timestamps # leave as list so we can dump to json later


def process_dutycycle(raw_result):
    if raw_result[0].find('Duty cycle start times (s)') < 0:
        raise ValueError(f"Duty cycle response must start with 'Duty cycle start times (s)'. First element is {raw_result[0]}")
    if raw_result[-1].find('m-ready') < 0:
        raise ValueError(f"List must end with 'm-ready'. Last element is {raw_result[-1]}")

    proc_start_times = []
    proc_stop_times = []

    target_list = proc_start_times
    # start populating proc_start_times, until we see stop-times message, then populate proc_stop_times
    endstr_found = False
    for i,line in enumerate(raw_result[1:]):
        if endstr_found:
            print(f"Warning: Duty cycle response continues beyond 'm-ready'.  Ignoring extra content")
            break
        if line.find('Duty cycle stop times (s)') >= 0:
            # end of start-times, beginning of stop-times.
            target_list = proc_stop_times
            continue
        if line.find('m-ready') > 0:
            line = line.replace('m-ready', '')
            endstr_found = True
            
        target_list += [int(numstr) for numstr in line.strip().split(',') if numstr]
        
    proc_start_times = np.array(proc_start_times)*10e-6
    proc_stop_times = np.array(proc_stop_times)*10e-6

    if len(proc_stop_times) != len(proc_start_times):
        err_str  = f"Number of start times ({len(proc_start_times)}) and number of "
        err_str += f"stop times ({len(proc_stop_times)}) should be equal"
        raise RuntimeError(err_str)
    on_times = proc_stop_times - proc_start_times
    periods = np.diff(proc_start_times)
    periods_fractional_var = (np.max(periods) - np.min(periods))/np.mean(periods)
    if periods_fractional_var > 0.02: # > 2% variation
        print(f"WARNING: Frame period variation exceeds 2%.", 
              f"Min period = {np.min(periods*1e3):.3f} ms. ", 
              f"Max period = {np.max(periods*1e3):.3f} ms")
    avg_period = np.mean(periods)
    avg_on_time = np.mean(on_times)
    avg_duty_cycle = avg_on_time / avg_period

    results = {'duty_cycle':avg_duty_cycle, 
               'period':avg_period,
               'processing_time':avg_on_time,
               'start_times_s':list(proc_start_times),
               'stop_times_s':list(proc_stop_times)
              }

    print(f"Duty cycle = {avg_duty_cycle:.3f}, Period = {avg_period*1e3:.3} ms")
    return results    

def calc_detection_stats(measured_timestamps_ms, true_det_windows,
                         false_pos_suppresion_delay_sec=1.0,
                         debounce_time_sec=1.0
                         ):
    
    if len(measured_timestamps_ms) == 0:
        raise RuntimeError("No detections captured.  There should be at least one to mark the"
                           "beginning of the run"
                           )
    measured_timestamps_ms = np.array(measured_timestamps_ms)
    detection_start = measured_timestamps_ms[0]
    fp_timestamps_ms = measured_timestamps_ms[1:] - detection_start

    true_positives = []
    false_negatives = []

    for i, (t_start, t_stop) in enumerate(true_det_windows):
        t_stop += false_pos_suppresion_delay_sec
        tp_indices = np.where((fp_timestamps_ms/1000 >= t_start) & 
                            (fp_timestamps_ms/1000 <= t_stop+false_pos_suppresion_delay_sec))[0]
        if len(tp_indices) > 0:
            true_positives.append(t_start)
            fp_timestamps_ms = np.delete(fp_timestamps_ms, tp_indices)
        else:
            false_negatives.append(t_start)

    i=0 # use while here b/c each loop we remove elements from fp_timestamps_ms
    while i < len(fp_timestamps_ms):
        current_fp = fp_timestamps_ms[i]

        fp_idxs_to_suppress = np.where((fp_timestamps_ms > current_fp) & 
                            (fp_timestamps_ms <= current_fp+int(debounce_time_sec*1000)))[0]    
        if len(fp_idxs_to_suppress) > 0:
            fp_timestamps_ms = np.delete(fp_timestamps_ms, fp_idxs_to_suppress)
        i += 1
    return true_positives, false_negatives, fp_timestamps_ms/1000

def replace_env_vars(str_in, env_dict=None):
    """
    Replaces any sub-strings enclosed by curly braces in str_in with the value 
    of the corresponding variable from env_dict.
    env_dict: a dict (or dict-like) of the form {'VAR_NAME':'value'}
    """
    if env_dict is None:
        env_dict = os.environ

    matches = re.findall(r"{(\w+)}", str_in)
    new_str = str_in
    for env_var_name in matches:
        if env_var_name in env_dict:
            env_var_value = env_dict[env_var_name]
            # Replace the enclosed string and curly braces with the value of the environment variable
            new_str = new_str.replace("{" + env_var_name + "}", env_var_value)
        else:
            raise ValueError(f"Environment variable {env_var_name} not found")
            
    return new_str
 
def summarize_sww_result(results_list, power, results_file=None):
    for res in results_list:
        inf_res = res["infer"]
        true_pos_sec, false_neg_sec, false_pos_sec = calc_detection_stats(
            inf_res["detections"], inf_res["detection_windows"])
        print(f"== File {inf_res['wav_file']} ({inf_res['length_sec']:2.1f} s) == ")
        with np.printoptions(precision=3):
            print_tee(f"    True positives: {true_pos_sec}", outfile=results_file)
            print_tee(f"    False negatives: {false_neg_sec}", outfile=results_file)
            print_tee(f"    False positives: {false_pos_sec}", outfile=results_file)
            print_tee(f"{len(true_pos_sec)} True positives, {len(false_neg_sec)} False negatives, {len(false_pos_sec)} False positives", outfile=results_file)
