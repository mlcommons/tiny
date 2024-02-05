import re

class _ScriptStep:
  def run(self, it, dut, dataset):
    return None


class _ScriptDownloadStep(_ScriptStep):
  def __init__(self, index = None):
    self._index = None if index is None else int(index)

  def run(self, it, dut, dataset):
    file_truth, data = dataset.get_file_by_index(self._index)
    if data:
      dut.load(data)
    return file_truth


class _ScriptLoopStep(_ScriptStep):
  def __init__(self, commands, loop_count=None):
    self._commands = [c for c in commands]
    self._loop_count = None if loop_count is None else int(loop_count)

  def run(self, it, dut, dataset):
    i = 0
    result = None if self._loop_count == 1 else []

    while self._loop_count is None or i < self._loop_count:
      loop_res = {}
      for cmd in self._commands:
        r = cmd.run(it, dut, dataset)
        if r is not None:
          loop_res.update(**r)
      i += 1
      if self._loop_count != 1:
        result.append(loop_res)
      else:
        result = loop_res

    return result


class _ScriptInferStep(_ScriptStep):
  def __init__(self, iterations=1, warmups=0):
    self._iterations = int(iterations)
    self._warmups = int(warmups)
    self._infer_results = None
    self._power_samples = []
    self._power_timestamps = []

  def run(self, it, dut, dataset):
    result = dut.infer(self._iterations, self._warmups)

    infer_results = self._gather_infer_results(result)

    result = dict(infer=infer_results)
    if dut.power_manager:
      timestamps, samples = self._gather_power_results(dut.power_manager)
      print(f"samples:{len(samples)} timestamps:{len(timestamps)}")
      result.update(power=dict(samples=samples,
                               timestamps=timestamps
                               )
                    )
    return result

  def _gather_infer_results(self, cmd_results):
    for res in cmd_results:
      match = re.match(r'^m-results-\[([^]]+)\]$', res)
      if match:
        return [float(x) for x in match.group(1).split(',')]

  def _gather_power_results(self, power):
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


class Script:
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
