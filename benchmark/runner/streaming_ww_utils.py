import numpy as np


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
    return np.array(timestamps)

def calc_detection_stats(measured_timestamps_ms, true_det_windows,
                         false_pos_suppresion_delay_sec=1.0,
                         debounce_time_sec=1.0
                         ):
    
    if len(measured_timestamps_ms) == 0:
        raise RuntimeError("No detections captured.  There should be at least one to mark the"
                           "beginning of the run"
                           )
    detection_start = measured_timestamps_ms[0]
    fp_timestamps_ms = measured_timestamps_ms[1:] - detection_start

    true_positives = []
    false_negatives = []

    for i, (t_start, t_stop) in enumerate(true_det_windows):
        t_stop += false_pos_suppresion_delay_sec
        print(f"{t_start}, {t_stop}")
        tp_indices = np.where((fp_timestamps_ms/1000 >= t_start) & 
                            (fp_timestamps_ms/1000 <= t_stop))[0]
        if len(tp_indices) > 0:
            true_positives.append(t_start)
            fp_timestamps_ms = np.delete(fp_timestamps_ms, tp_indices)
        else:
            false_negatives.append(t_start)

    i=0 # use while here b/c each loop we remove elements from fp_timestamps_ms
    while i < len(fp_timestamps_ms):
        current_fp = fp_timestamps_ms[i]
        print(f"FP at {current_fp/1000}")

        fp_idxs_to_suppress = np.where((fp_timestamps_ms >= current_fp) & 
                            (fp_timestamps_ms/1000 <= current_fp+int(debounce_time_sec*1000)))[0]    
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



def summarize_sww_result(result, mode, power):  # Pass power to summarize_result
    true_pos_sec, false_neg_sec, false_pos_sec = calc_detection_stats(
        result["detections"], result["detection_windows"])
    print(f"True Positives: {true_pos_sec}")
    print(f"False negatives: {false_neg_sec}")
    print(f"False Positives: {false_pos_sec}")