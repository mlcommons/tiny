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

    return np.array(timestamps)

def summarize_sww_result(result, mode, power):  # Pass power to summarize_result
    pass