import re
import numpy as np
from datetime import datetime
from device_under_test import DUT  # Import DUT class
from power_manager import PowerManager

file_processed = False
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
                print(f"Loading file {file_truth.get('file'):30}, true class = {int(file_truth.get('class')):2}")
            elif mode == "p":
                print(f"{formatted_time} ulp-mlperf: Runtime requirements have been met.")
            elif mode == "e":
                pass  # Do nothing for energy mode
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
        
        i = 0
        result = None if self._loop_count == 1 else []

        while self._loop_count is None or i < self._loop_count:
            loop_res = {}
            for cmd in self._commands:
                r = cmd.run(io, dut, dataset, mode)
                if r is not None:
                    loop_res.update(**r)
            if self.model == "ad01":
                total_length = loop_res.get('total_length', None)
                if total_length is not None and i == 0:
                    self._loop_count *= total_length
            i += 1
            if self._loop_count != 1:
                result.append(loop_res)
            else:
                result = loop_res

        return result

class _ScriptInferStep(_ScriptStep):
    """Step to execute infer on the DUT"""
    def __init__(self, iterations=1, warmups=0, loop_count=None):
        self._iterations = int(iterations)
        self._warmups = int(warmups)
        self._infer_results = None
        self._power_samples = []
        self._power_timestamps = []
        self.throughput_values = []
        self._loop_count = loop_count  # Store loop_count passed to this step

    def run(self, io, dut, dataset, mode):  # mode passed to run
        result = dut.infer(self._iterations, self._warmups)

        infer_results = _ScriptInferStep._gather_infer_results(result)

        result = dict(infer=infer_results)
        if dut.power_manager:
            timestamps, samples = _ScriptInferStep._gather_power_results(dut.power_manager)
            print(f"samples:{len(samples)} timestamps:{len(timestamps)}")
            result.update(power=dict(samples=samples,
                                      timestamps=timestamps))

        if mode == "a":
            self._print_accuracy_results(infer_results)
        elif mode == "e":
            self._print_energy_results(infer_results)
        elif mode == "p":
            self._print_performance_results(infer_results)

        return result

    @staticmethod
    def _gather_infer_results(cmd_results):
        result = {}
        total_inferences = 0
        for res in cmd_results:
            match = re.match(r'^m-results-\[([^]]+)\]$', res)
            if match:
                try:
                    results = [float(x) for x in match.group(1).split(',') if x.strip()]
                    result["results"] = results
                    total_inferences += len(results)
                except ValueError as e:
                    print(f"ERROR: Failed to parse infer results: {e}. Data: {match.group(1)}")
                    result["results"] = []
                continue
            match = re.match(r'^m-lap-us-([0-9]+)$', res)
            if match:
                key = "end_time" if "start_time" in result else "start_time"
                result[key] = int(match.group(1))
        if "start_time" in result and "end_time" in result:
            result["elapsed_time"] = result["end_time"] - result["start_time"]
        else:
            print("ERROR: Incomplete time data, missing start_time or end_time.")
        result["total_inferences"] = total_inferences
        return result


    @staticmethod
    def _gather_power_results(power):
        samples = []
        timeStamps = []
        if power:
            for x in power.get_results():
                if isinstance(x, str):
                    match = re.match(r"^TimeStamp: ([0-9]{3})s ([0-9]{3})ms, buff [0-9]{2}%$", x)
                    ts = float(f"{match.group(1)}.{match.group(2)}")
                    timeStamps.append((ts, len(samples)))
                else:
                    samples.append(x)
        return timeStamps, samples

    def _print_accuracy_results(self, infer_results):
        print(f"    Results = {infer_results['results']}, time={infer_results['elapsed_time']} us")

    def _print_energy_results(self, infer_results):
        """
        Accumulates energy values for all loop iterations and calculates median energy per inference 
        at the end of all iterations.
        This function interacts with the PowerManager class to gather and process power data.
        """
        num_inferences = self._iterations

        # Get the current time in the desired format
        current_time = datetime.now()
        formatted_time = current_time.strftime("%m%d.%H%M%S")

        # Initialize the PowerManager with the correct port and baud rate
        port_device = "COM19"  # Replace with your serial port
        power_manager = PowerManager(port_device)

        # Use the PowerManager to gather energy and power results
        with power_manager:
            timestamps, power_samples = [], []

            # Start collecting data from PowerManager
            power_manager.start()
            for result in power_manager.get_results():
                if isinstance(result, str) and result.startswith("TimeStamp"):
                    timestamps.append(float(result.split()[-1]))
                elif isinstance(result, float):
                    power_samples.append(result)
            power_manager.stop()

            if len(timestamps) > 1 and len(power_samples) > 0:
                # Calculate total energy (sum of power * time intervals)
                total_energy = sum(
                    [power_samples[i] * (timestamps[i + 1] - timestamps[i]) for i in range(len(timestamps) - 1)]
                )

                # Calculate average power (mean of the recorded power samples)
                average_power = np.mean(power_samples)

                # Calculate energy per inference (total energy divided by number of inferences)
                energy_per_inference = total_energy / num_inferences if num_inferences > 0 else 0

                # Print energy results for each window
                print(f"{formatted_time} ulp-ml: Energy data for window {len(self.throughput_values)} at time {timestamps[-1]:.2f} for {timestamps[-1] - timestamps[0]:.2f} sec.:")
                print(f"{formatted_time} ulp-ml:   Energy       : {total_energy:>13.3f} uJ")
                print(f"{formatted_time} ulp-ml:   Power        : {average_power:>13.3f} uW")
                print(f"{formatted_time} ulp-ml:   Energy/Inf.  : {energy_per_inference:>13.3f} uJ/inf.")

                # Store energy values for calculating median
                self.energy_values.append(energy_per_inference)

        # Check if all iterations are complete
        if len(self.energy_values) == self._loop_count:
            # Calculate the median energy per inference after all loop iterations
            total_median_energy = np.median(self.energy_values)

            # Store the result for later use
            self.median_energy = total_median_energy

            # Print the formatted output with median energy per inference
            print(f"{formatted_time} ulp-ml: ---------------------------------------------------------")
            print(f"{formatted_time} ulp-ml: Median energy cost is {self.median_energy:>10.3f} uJ/inf.")
            print(f"{formatted_time} ulp-ml: ---------------------------------------------------------")


    def _print_performance_results(self, infer_results):
        """
        Accumulates throughput values for all loop iterations and calculates median throughput 
        at the end of all iterations.
        """
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

        # Add throughput to the list of throughput values
        self.throughput_values.append(throughput)

        # Print the old format performance results for each inference (every loop)
        print(f"{formatted_time} ulp-mlperf: Performance results for window {len(self.throughput_values)}:")
        print(f"{formatted_time} ulp-mlperf:   # Inferences : {num_inferences:>13}")
        print(f"{formatted_time} ulp-mlperf:   Runtime      : {elapsed_time_sec:>10.3f} sec.")
        print(f"{formatted_time} ulp-mlperf:   Throughput   : {throughput:>10.3f} inf./sec.")
        print(f"{formatted_time} ulp-mlperf:   m-results    : {infer_results['results']}")

        # Check if we've completed all loop iterations
        if len(self.throughput_values) == self._loop_count:
            # Calculate the median throughput after all loop iterations
            total_median_throughput = np.median(self.throughput_values)

            # Store the result for later use
            self.median_throughput = total_median_throughput

            # Print the new formatted output with median throughput
            print(f"{formatted_time} ulp-mlperf: ---------------------------------------------------------")
            print(f"{formatted_time} ulp-mlperf: Median throughput is {self.median_throughput:>10.3f} inf./sec.")
            print(f"{formatted_time} ulp-mlperf: ---------------------------------------------------------")


class _ScriptStreamStep(_ScriptStep):
    """Step to stream audio from an enhanced interface board"""
    def __init__(self, file_name=None):
        self._file_name = file_name

    def run(self, io, dut, dataset, mode):
        io.play_wave(self._file_name)
        return dict(audio_file=self._file_name)


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

    def run(self, io, dut, dataset, mode):
        with io:
            with dut:
                result = None
                for cmd in self._commands:
                    r = cmd.run(io, dut, dataset, mode)
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
                return result