import re
import numpy as np
from datetime import datetime
from device_under_test import DUT  # Import DUT class
global_loop_count = None  # This will store the loop count globally
performance_timed_out = False  # True when 10 seconds have passed
file_processed = False
import logging
import sys
import concurrent.futures
import time
import threading

import streaming_ww_utils as sww_util

watchdog_flag = {"timeout": False}
watchdog_lock = threading.Lock()
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{current_time}.log"

# Setup logging to file and console with NO extra formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Removes timestamp and log level
    handlers=[
        logging.FileHandler(log_filename, mode='w'),  # Log to file
        logging.StreamHandler(sys.stdout)  # Print to console
    ]
)

# Redirect print statements to logging
class LoggerWriter:
    """Custom writer to redirect stdout/stderr to logging."""
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip():  # Avoid logging empty messages
            self.level(message.strip())

    def flush(self):
        """Ensure real-time logging output."""
        for handler in logging.getLogger().handlers:
            handler.flush()

# Redirect all standard outputs to logging
sys.stdout = LoggerWriter(logging.info)  # Capture stdout
sys.stderr = LoggerWriter(logging.error)  # Capture stderr for errors

print(f"Logging initialized. Writing output to {log_filename}")


class _ScriptStep:
    """Base class for script steps"""
    def run(self, io, dut, dataset, mode):
        return None

class _ScriptDownloadStep(_ScriptStep):
    """Step to download a file to the DUT"""
    def __init__(self, index=None, model=None):
        self._index = None if index is None else int(index)
        self.model = model
        self._segments = None
        self._current_index = 0
        self._current_segment_index = 0
        self.total_length = 0

    def run(self, io, dut, dataset, mode):
        # Fetch the file and data
        if(self.model == "ad01" and self.total_length != 0):
            if (self._current_segment_index == self.total_length): 
                self._current_index += 1
                self._current_segment_index = 0
                self._segments = None
            file_truth, data = dataset.get_file_by_index(self._current_index)
        else:
            file_truth, data = dataset.get_file_by_index(self._index)
        
        # Define the current time for formatted output
        current_time = datetime.now()
        formatted_time = current_time.strftime("%m%d.%H%M%S")
        
        if self._segments is None and self.model == "ad01":  
            total_size = len(data)
            segment_size = int(file_truth.get('bytes_to_send'))  # Unified variable
            stride = int(file_truth.get('stride'))
            max_size = total_size
            self._segments = []
            start = 0
            while start + segment_size <= max_size:
                end = start + segment_size
                self._segments.append(data[start:end])
                start = start + stride  # Move start by stride to create overlap
                self.total_length = len(self._segments)
            self._current_segment_index = 0
            
        # Conditional print statements based on 'mode'
        if data:
            if mode == "a":
                print(f"{formatted_time} ulp-mlperf: Runtime requirements have been met.")
            elif mode == "p":
                print(f"{formatted_time} ulp-mlperf: Running Performance Metrics")
            if(self.model == "ad01"):
                segment = self._segments[self._current_segment_index]
                dut.load(segment)
                self._current_segment_index += 1
            else:
                dut.load(data)
        else:
            print(f"WARNING: No data returned from dataset read. Script index = {self._index}, Dataset index = {dataset._current_index}")
        file_truth['total_length'] = self.total_length if self.model == "ad01" else None
        return file_truth


class _ScriptLoopStep(_ScriptStep):
    """Step that implements a loop of nested steps"""
    def __init__(self, commands, loop_count=None, model=None):
        self._commands = [c for c in commands]
        self._loop_count = None if loop_count is None else int(loop_count)
        self.model = model

    def run(self, io, dut, dataset, mode):
        global global_loop_count
        i = 0
        result = None if self._loop_count == 1 else []
        start_time = time.time() if mode == "p" else None

        while (self._loop_count is None or i < self._loop_count):
            loop_res = {}
            with watchdog_lock:
                if watchdog_flag["timeout"]:
                    print("[WATCHDOG] Loop timeout. Restarting current loop iteration.")
                    watchdog_flag["timeout"] = False  # reset for next iteration
                    r = cmd.run(io, dut, dataset, mode)
            for cmd in self._commands:
                r = cmd.run(io, dut, dataset, mode)
                if r is not None:
                    loop_res.update(**r)
            if self.model == "ad01":
                total_length = loop_res.get('total_length', None)
                if total_length is not None and i == 0:
                    self._loop_count *= total_length
            global_loop_count = self._loop_count  # Store in global variable
            i += 1
            if self._loop_count != 1:
                result.append(loop_res)
            else:
                result = loop_res
        if mode == "p":
            # Error 1: Loop count not exactly 5
            if self._loop_count != 5:
                result.append({"error": "error 1"})

            # Error 2: Loop finished before 10 seconds
            duration = time.time() - start_time
            if duration < 10.0 and i >= self._loop_count:
                result.append({"error": "error 2"})
        return result

class _ScriptInferStep(_ScriptStep):
    """Step to execute infer on the DUT"""
    def __init__(self, iterations=1, warmups=0, skip_time=0, loop_count=None):
        self._iterations = int(iterations)
        self._warmups = int(warmups)
        self.skip_time = float(skip_time)
        self._infer_results = None
        self._power_samples = []
        self._power_timestamps = []
        self.throughput_values = []
        self._loop_count = loop_count


    def run(self, io, dut, dataset, mode):  # mode passed to run
        raw_result = dut.infer(self._iterations, self._warmups)
        infer_results = _ScriptInferStep._gather_infer_results(raw_result, mode)
        infer_results["iterations"] = self._iterations
        infer_results["warmups"] = self._warmups

        result = dict(infer=infer_results)

        if mode == "e":
            timestamps, samples = _ScriptInferStep._gather_power_results(dut.power_manager)
            print(f"samples:{len(samples)} timestamps:{len(timestamps)}")

            # Read power values from the log file using timestamps
            power_values = None

            result.update(power=dict(samples=samples,
                                     timestamps=timestamps,
                                     extracted_power=power_values  # <-- NEW: Power values from the file
                                    )
                         )
        else:
            self._print_AP_results(infer_results,mode)
        
        return result
    @staticmethod
    def _gather_power_results(power):
        # this method should really be done inside the power manager, since
        # other emon devices will structure the raw data different
        samples = []
        timeStamps = [] # this is what the EEMBC runner calls "timestamps"
        clock_ticks = [] # this is what the LPM01a calls "timestamps"
        if power:
            for x in power.get_results():
                if isinstance(x, str):
                    if x.startswith("TimeStamp"):
                        match = re.match(r"^TimeStamp: ([0-9]{3})s ([0-9]{3})ms, buff [0-9]{2}%$", x)                    
                        ts = float(f"{match.group(1)}.{match.group(2)}")
                        clock_ticks.append((ts, len(samples)))
                        if len(clock_ticks) > 1 and clock_ticks[-1][1]-clock_ticks[-2][1] != 1000:
                            expected_samples = clock_ticks[-2][1] + 1000
                            print(f"At time {ts}: expected {expected_samples}, but we have {clock_ticks[-1][1]}.")
                    elif x.startswith("event"):
                        match = re.match(r"event (\d+) ris", x)
                        event_num = match.group(1)
                        timeStamps.append((event_num, len(samples)))
                else:
                    samples.append(x)
        return timeStamps, samples

    @staticmethod
    def _gather_infer_results(cmd_results, mode):
        result = {}
        total_inferences = 0
        for res in cmd_results:
            match = re.match(r'^m-results-\[([^]]+)\]$', res)
            if match:
                try:
                    results = [float(x) for x in match.group(1).split(',') if x.strip()]
                    result["results"] = results
                    total_inferences += 1 # this results vector is the output of 1 inference, not a collection of multiple inferences
                except ValueError as e:
                    print(f"ERROR: Failed to parse infer results: {e}. Data: {match.group(1)}")
                    result["results"] = []
                continue
            match = re.match(r'^m-lap-us-([0-9]+)$', res)
            if match:
                key = "end_time" if "start_time" in result else "start_time"
                result[key] = int(match.group(1))
        if mode != "e" and "start_time" in result and "end_time" in result:
            result["elapsed_time"] = result["end_time"] - result["start_time"]
        elif mode != "e":
            print("ERROR: Incomplete time data, missing start_time or end_time.")
        result["total_inferences"] = total_inferences
        return result

    def _print_AP_results(self, infer_results,mode):
        """
        Accumulates throughput values for all loop iterations and calculates median throughput 
        at the end of all iterations.
        """
        global global_loop_count
        # Use the total_inferences from _gather_infer_results results
        num_inferences = self._iterations
        
        # Retrieve elapsed time (in microseconds)
        elapsed_time_us = infer_results.get("elapsed_time", 0)

        # Get the current time in the desired format
        current_time = datetime.now()
        formatted_time = current_time.strftime("%m%d.%H%M%S")

        # Calculate throughput
        if elapsed_time_us > 0:
            elapsed_time_sec = elapsed_time_us / 1_000_000  # Convert to seconds
            throughput = num_inferences / elapsed_time_sec  # Calculate throughput
        else:
            elapsed_time_sec = 0
            throughput = 0
        if mode == "p":
            infer_results["throughput"] = throughput
        # Add throughput to the list of throughput values
        self.throughput_values.append(throughput)

        # Print the old format performance results for each inference (every loop)
        print(f"{formatted_time} ulp-mlperf: Performance results for window {len(self.throughput_values)}:")
        print(f"{formatted_time} ulp-mlperf:   # Inferences : {num_inferences:>13}")
        print(f"{formatted_time} ulp-mlperf:   Runtime      : {elapsed_time_sec:>10.3f} sec.")
        print(f"{formatted_time} ulp-mlperf:   Throughput   : {throughput:>10.3f} inf./sec.")
        print(f"{formatted_time} ulp-mlperf:   m-results    : {infer_results['results']}")

class _ScriptStreamStep(_ScriptStep):
    """Step to stream audio from an enhanced interface board"""
    def __init__(self, index=None):        
        self._current_index = 0
        self._index = None if index is None else int(index)

    def run(self, io, dut, dataset, mode):
        file_truth = dataset.get_file_by_index(self._index)
        dut.stop_detecting()   # in case it wasn't stopped earlier
        dut.start_detecting()  # instruct DUT to pulse GPIO when wakeword detected
        io.record_detections() # intfc starts recording timestamp of GPIO pulses
        print(f"Playing {file_truth['wav_file']} ... ", end="")
        # play_wave is a blocking call, will pause here until wav finishes
        io.play_wave(file_truth['wav_file'], timeout=file_truth["length_sec"]+10.0) 
        # not sure why this is needed, but apparently the DUT is still occupied with the 
        # detection task, because w/o this sleep the next DUT command times out.  Could be shorter
        time.sleep(5)
        print(" ... done")
        dut.stop_detecting()   # DUT stops pulsing GPIO on WW.
        detected_timestamps = io.print_detections() # intfc prints out WW detection timestamps
        detected_timestamps = sww_util.process_timestamps(detected_timestamps)
        results = {}
        results.update(file_truth)        
        results["detections"] = detected_timestamps
        return results

class _WatchdogStep(_ScriptStep):
    def __init__(self, timeout):
        self.timeout = float(timeout)
        self._timer_thread = None
        self._last_reset_time = time.time()

    def _watchdog_thread(self):
        while True:
            time.sleep(0.5)
            with watchdog_lock:
                elapsed = time.time() - self._last_reset_time
                if elapsed >= self.timeout:
                    watchdog_flag["timeout"] = True
                    print(f"[WATCHDOG] Timeout reached after {elapsed:.2f}s")
                    break

    def run(self, io, dut, dataset, mode):
        with watchdog_lock:
            self._last_reset_time = time.time()
            watchdog_flag["timeout"] = False

        if self._timer_thread is None or not self._timer_thread.is_alive():
            self._timer_thread = threading.Thread(target=self._watchdog_thread, daemon=True)
            self._timer_thread.start()
        return {}

class Script:
    """Script that executes a test"""
    def __init__(self, script):
        self.name = script.get("name")
        self.model = script.get("model")
        self.truth = script.get("truth_file")
        self._commands = [s for s in self._parse_steps(script.get("script", []))]

    def _parse_steps(self, steps):
        for step in steps:
            contents = None
            if isinstance(step, dict):
                step, contents = list(step.items())[0]
            yield self._create_step(step, contents)

    def _create_step(self, step, contents):
        parts = step.split(' ')
        cmd = parts[0]
        args = parts[1:]
        if cmd == 'download':
            # Pass the model into the download step
            return _ScriptDownloadStep(*args, model=self.model)
        if cmd == 'loop':
            # Pass the model into the loop step and its commands
            loop_count = int(args[0]) if args else None
            return _ScriptLoopStep(self._parse_steps(contents), loop_count, model=self.model)
        if cmd == 'infer':
            # Pass the loop_count to the infer step
            loop_count = args[-1] if args else None  # Assuming loop_count is passed as last argument
            return _ScriptInferStep(*args, loop_count=loop_count)
        if cmd == 'reset':
            #resets the watchdog timer
            return _WatchdogStep(*args)
        if cmd == 'stream':
            # Pass the model into the download step
            return _ScriptStreamStep(*args)

    def run(self, io, dut, dataset, mode):
        result = None
        if io != None:
            with io:
                with dut:
                    result = None
                    for cmd in self._commands:
                        r = cmd.run(io, dut, dataset, mode)  # Pass boolean flag
                        if result is None:
                            result = r
                        elif isinstance(r, dict):
                            if isinstance(result, dict):
                                result.update(**r)
                            else:
                                result.append(r)
                        else:
                            if isinstance(result, dict):
                                result.update(res=r)
                            else:
                                result.extend(r)
        elif dut != None:  # Accuracy mode
            with dut:
                for cmd in self._commands:
                    r = cmd.run(io, dut, dataset, mode)
                    result = self._merge_results(result, r)
        else:
            raise RuntimeError("Missing required device instance.")

        return result

    def _merge_results(self, current, new):
        if current is None:
            return new
        if isinstance(new, dict):
            if isinstance(current, dict):
                current.update(**new)
            else:
                current.append(new)
        else:
            if isinstance(current, dict):
                current.update(res=new)
            else:
                current.extend(new)
        return current

