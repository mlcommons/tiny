import re


class _ScriptStep:
  """Base class for script steps
  """
  def run(self, io, dut, dataset):
    return None

class _ScriptDownloadStep(_ScriptStep):
  """Step to download a file to the DUT
  """
  def __init__(self, index=None):
    self._index = None if index is None else int(index)

  def run(self, io, dut, dataset):
    file_truth, data = dataset.get_file_by_index(self._index)
    if data:
      print(f"Loading file {file_truth.get('file'):30}, true class = {int(file_truth.get('class')):2}")
      dut.load(data)
    else:
      print(f"WARNING: No data returned from dataset read.  Script index = {self._index}, Dataset index = {dataset._current_index}")
    return file_truth

class _ScriptLoopStep(_ScriptStep):
  """Step that implements a loop of nested steps
  """
  def __init__(self, commands, loop_count=None):
    self._commands = [c for c in commands]
    self._loop_count = None if loop_count is None else int(loop_count)

  def run(self, io, dut, dataset):
    i = 0
    result = None if self._loop_count == 1 else []

    while self._loop_count is None or i < self._loop_count:
      loop_res = {}
      for cmd in self._commands:
        r = cmd.run(io, dut, dataset)
        if r is not None:
          loop_res.update(**r)
      i += 1
      if self._loop_count != 1:
        result.append(loop_res)
      else:
        result = loop_res

    return result


class _ScriptInferStep(_ScriptStep):
  """Step to execute infer on that dut
  """
  def __init__(self, iterations=1, warmups=0):
    self._iterations = int(iterations)
    self._warmups = int(warmups)
    self._infer_results = None
    self._power_samples = []
    self._power_timestamps = []

  def run(self, io, dut, dataset):
    result = dut.infer(self._iterations, self._warmups)

    infer_results = _ScriptInferStep._gather_infer_results(result)

    if not 'results' in infer_results:
      print(f"Output of dut.infer:\n{result}")
      raise RuntimeError("No 'results' found in the inference results.  Likely failure in dut.infer")

    result = dict(infer=infer_results)
    if dut.power_manager:
      timestamps, samples = _ScriptInferStep._gather_power_results(dut.power_manager)
      print(f"samples:{len(samples)} timestamps:{len(timestamps)}")
      result.update(power=dict(samples=samples,
                               timestamps=timestamps
                               )
                    )
    # Indent to connect it visually the preceding 'loading' printout
    print(f"    Results = {infer_results['results']}, time={infer_results['elapsed_time']} us")
    return result

  @staticmethod
  def _gather_infer_results(cmd_results):
    result = {}
    for res in cmd_results:
      match = re.match(r'^m-results-\[([^]]+)\]$', res)
      if match:
        result["results"] = [float(x) for x in match.group(1).split(',')]
        continue
      match = re.match(r'^m-lap-us-([0-9]+)$', res)
      if match:
        key = "end_time" if "start_time" in result else "start_time"
        result[key] = int(match.group(1))
    if "start_time" in result and "end_time" in result:
      result["elapsed_time"] = result["end_time"] - result["start_time"]
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


class _ScriptStreamStep(_ScriptStep):
  """Step to stream audio from an enhanced interface board
  """
  def __init__(self, file_name=None):
    self._file_name = file_name

  def run(self, io, dut, dataset):
    io.play_wave(self._file_name)
    return dict(audio_file=self._file_name)


class Script:
  """Script that executes a test
  """
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
      return _ScriptDownloadStep(*args)
    if cmd == 'loop':
      return _ScriptLoopStep(self._parse_steps(contents), *args)
    if cmd == 'infer':
      return _ScriptInferStep(*args)

  def run(self, io, dut, dataset):
    with io:
      with dut:
        result = None
        for cmd in self._commands:
          r = cmd.run(io, dut, dataset)
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
