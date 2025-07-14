import argparse
import json
import logging
import os
import re
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


MODEL_CONFIG = {
    "v1.0": {
        "models": ["ad", "ic", "kws", "vww"],
        "required-scenarios": {
            # anything goes
        },
        "optional-scenarios": {
            # anything goes
        },
        "accuracy-target": {
            "ad": ("auc", 0.85),
            "ic": ("top-1", 85),
            "kws": ("top-1", 90),
            "vww": ("top-1", 80),
        },
        "model_mapping": {
            # map model names to the official mlperf model class
            "anomaly_detection": "ad",
            "image_classification": "ic",
            "keyword_spotting": "kws",
            "visual_wake_words": "vww",
            "streaming_wakeword_detection": "sww",
        },
    },
    "v1.1": {
        "models": ["ad", "ic", "kws", "vww"],  
        "required-scenarios": {
            # anything goes
        },
        "optional-scenarios": {
            # anything goes
        },
        "accuracy-target": {
            "ad": ("auc", 0.85),
            "ic": ("top-1", 85),
            "kws": ("top-1", 90),
            "vww": ("top-1", 80),
        },
        "model_mapping": {
        },
    },
    "v1.2": {
        "models": ["ad", "ic", "kws", "vww"],  
        "required-scenarios": {
            # anything goes
        },
        "optional-scenarios": {
            # anything goes
        },
        "accuracy-target": {
            "ad": ("auc", 0.85),
            "ic": ("top-1", 85),
            "kws": ("top-1", 90),
            "vww": ("top-1", 80),
        },
        "model_mapping": {
        },
    },
    "v1.3_eembc": {
        "models": ["ad", "ic", "kws", "vww"],  
        "required-scenarios": {
            # anything goes
        },
        "optional-scenarios": {
            # anything goes
        },
        "accuracy-target": {
            "ad": ("auc", 0.85),
            "ic": ("top-1", 85),
            "kws": ("top-1", 90),
            "vww": ("top-1", 80),
        },
        "model_mapping": {
        },
        "required_tests": {
          "ad": ["accuracy", "performance"],
          "ic": ["accuracy", "performance"],
          "kws": ["accuracy", "performance"],
          "vww": ["accuracy", "performance"],
          "sww": ["energy"]
        },
        "optional_tests": {
          "ad": ["energy"],
          "ic": ["energy"],
          "kws": ["energy"],
          "vww": ["energy"],
          "sww": []
        },
    },
    "v1.3": {
        "models": ["ad", "ic", "kws", "vww", "sww"],  
        "required-scenarios": {
            # anything goes
        },
        "optional-scenarios": {
            # anything goes
        },
        "accuracy-target": {
            "ad": ("auc", 0.85),
            "ic": ("top-1", 85),
            "kws": ("top-1", 90),
            "vww": ("top-1", 80),
            "sww": ("fps_fns", (8,8)),
        },
        "model_mapping": {
        },
        "required_tests": {
          "ad": ["accuracy", "performance"],
          "ic": ["accuracy", "performance"],
          "kws": ["accuracy", "performance"],
          "vww": ["accuracy", "performance"],
          "sww": ["energy"]
        },
        "optional_tests": {
          "ad": ["energy"],
          "ic": ["energy"],
          "kws": ["energy"],
          "vww": ["energy"],
          "sww": []
        },
        "required_files": ["log.txt", "results.json", "results.txt"]
    }
}
VALID_DIVISIONS = ["open", "closed"]
VALID_AVAILABILITIES = ["available", "preview", "rdi"]
EEMBC_REQUIRED_ACC_FILES = [
    "log.txt", "results.txt",
    "script.async",
]
MLC_REQUIRED_ACC_FILES = [
    "log.txt", "results.txt",
    "results.json",
]
ACC_FILE = "results.txt"
ACC_PATTERN = {
    "top-1":
        r".* Top[- ]1%?\s?[:=] ([\d\.]+).*", # match "Top-1: 91.1%" (old) or "Top 1% = 85.4" (new)
    "auc":
        r".* AUC\s?[:=] ([\d\.]+).*", # match "AUC: 0.93" (old) or  "AUC = 0.862" (new)
}
FILE_SIZE_LIMIT_MB = 500
MB_TO_BYTES = 1024*1024
EEMBC_REQUIRED_PERF_FILES = EEMBC_REQUIRED_ACC_FILES
EEMBC_REQUIRED_PERF_POWER_FILES = EEMBC_REQUIRED_ACC_FILES

MLC_REQUIRED_PERF_FILES = MLC_REQUIRED_ACC_FILES
MLC_REQUIRED_PERF_POWER_FILES = MLC_REQUIRED_ACC_FILES

OPTIONAL_PERF_FILES = [""]

def list_dir(*path):
  path = os.path.join(*path)
  return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def list_files(*path):
  path = os.path.join(*path)
  return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def list_empty_dirs_recursively(*path):
  path = os.path.join(*path)
  return [dirpath for dirpath, dirs, files in os.walk(path) if not dirs and not files]


def list_dirs_recursively(*path):
  path = os.path.join(*path)
  return [dirpath for dirpath, dirs, files in os.walk(path)]


def list_files_recursively(*path):
  path = os.path.join(*path)
  return [os.path.join(dirpath, file) for dirpath, dirs, files in os.walk(path) for file in files]


def split_path(m):
  return m.replace("\\", "/").split("/")

class Config():
  """Select config value by mlperf version and submission type."""

  def __init__(self,
               version,
               extra_model_benchmark_map,
               ignore_uncommited=False,
               more_power_check=False):
    self.base = MODEL_CONFIG.get(version)
    self.extra_model_benchmark_map = extra_model_benchmark_map
    self.version = version
    self.models = self.base["models"]
    #self.seeds = self.base["seeds"]
    #self.test05_seeds = self.base["test05_seeds"]
    self.accuracy_target = self.base["accuracy-target"]
    #self.performance_sample_count = self.base["performance-sample-count"]
    #self.latency_constraint = self.base.get("latency-constraint", {})
    #self.min_queries = self.base.get("min-queries", {})
    self.required = None
    self.optional = None
    self.ignore_uncommited = ignore_uncommited
    self.more_power_check = more_power_check

  def set_type(self, submission_type):
    if submission_type is None and self.version in ["v1.0", "v1.1", "v1.2", "v1.3"]:
      self.required = self.base["required-scenarios"]
      self.optional = self.base["optional-scenarios"]
    else:
      raise ValueError("invalid system type")

  def get_mlperf_model(self, model, extra_model_mapping = None):
    # preferred - user is already using the official name
    if model in self.models:
      return model

    # simple mapping, ie resnet50->resnet
    mlperf_model = self.base["model_mapping"].get(model)
    if mlperf_model:
      return mlperf_model

    # Custom mapping provided by the submitter
    if extra_model_mapping is not None:
      mlperf_model = extra_model_mapping.get(model)
      if mlperf_model:
        return mlperf_model

  def get_required(self, model):
    if self.version in ["v0.5"]:
      return set()
    model = self.get_mlperf_model(model)
    if model not in self.required:
      return None
    return set(self.required[model])

  def get_optional(self, model):
    if self.version in ["v0.5"]:
      return set(["SingleStream", "MultiStream", "Server", "Offline"])
    model = self.get_mlperf_model(model)
    if model not in self.optional:
      return set()
    return set(self.optional[model])

  def get_accuracy_target(self, model):
    if model not in self.accuracy_target:
      raise ValueError("model not known: " + model)
    return self.accuracy_target[model]

  def get_performance_sample_count(self, model):
    model = self.get_mlperf_model(model)
    if model not in self.performance_sample_count:
      raise ValueError("model not known: " + model)
    return self.performance_sample_count[model]

  def ignore_errors(self, line):
    for error in self.base["ignore_errors"]:
      if error in line:
        return True
    if self.ignore_uncommited and ("ERROR : Loadgen built with uncommitted "
                                   "changes!") in line:
      return True
    return False

  def get_min_query_count(self, model, scenario):
    model = self.get_mlperf_model(model)
    if model not in self.min_queries:
      raise ValueError("model not known: " + model)
    return self.min_queries[model].get(scenario)

  def has_new_logging_format(self):
    return self.version not in ["v0.5", "v0.7"]

  def uses_legacy_multistream(self):
    return self.version in ["v0.5", "v0.7", "v1.0", "v1.1", "v1.2"]

  def uses_early_stopping(self, scenario):
    return (self.version not in [
        "v0.5", "v0.7", "v1.0", "v1.1", "v1.2"
    ]) and (scenario in ["Server", "SingleStream", "MultiStream"])

  def has_query_count_in_log(self):
    return self.version not in ["v0.5", "v0.7", "v1.0", "v1.1", "v1.2"]

  def has_power_utc_timestamps(self):
    return self.version not in ["v0.5", "v0.7", "v1.0"]

def get_args():
  """Parse commandline."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True, help="submission directory")
  parser.add_argument(
      "--version",
      default="v1.3",
      choices=list(MODEL_CONFIG.keys()),
      help="mlperf version")
  parser.add_argument("--submitter", help="filter to submitter")
  parser.add_argument(
      "--csv", default="summary.csv", help="csv file with results")
  parser.add_argument(
      "--skip_compliance",
      action="store_true",
      help="Pass this cmdline option to skip checking compliance/ dir")
  parser.add_argument(
      "--extra-model-benchmark-map",
      help="File containing extra custom model mapping. It is assumed to be inside the folder open/<submitter>",
      default="model_mapping.json")
  parser.add_argument("--debug", action="store_true", help="extra debug output")
  parser.add_argument(
      "--submission-exceptions",
      action="store_true",
      help="ignore certain errors for submission")
  parser.add_argument(
      "--more-power-check",
      action="store_true",
      help="apply Power WG's check.py script on each power submission. Requires Python 3.7+"
  )
  args = parser.parse_args()
  return args


def check_results_dir(config,
                      filter_submitter,
                      skip_compliance,
                      csv,
                      debug=False):
  """
    Walk the results directory and do the checking.
    We are called with the cdw at the root of the submission directory.
    level1 division - closed|open|network
    level2 submitter - for example mlperf_org
    level3 - results, systems, measurements, code
    For results the structure from here is:
    results/$system_desc/$benchmark_model/$scenario/performance/run_n
    and
    results/$system_desc/$benchmark_model/$scenario/accuracy
    We first walk into results/$system_desc
        make sure there is a system_desc.json and its good
    Next we walk into the model
        make sure the model is good, make sure all required scenarios are there.
    Next we walk into each scenario
        check the performance directory
        check the accuracy directory
        if all was good, add the result to the results directory
        if there are errors write a None as result so we can report later what
        failed
    """
  head = [
      "Organization", "Availability", "Division", "BoardName", "SystemDesc",
      "Model", "MlperfModel",
      "Result", "ResultUnit", "Accuracy",
      "HasPower", "Power", "PowerUnit",
      "HostProcessorModelName", "HostProcessorFrequency", "HostProcessorMemory",
      "AcceleratorModelName", "AcceleratorFrequency", "AcceleratorMemory",
      "HardwareNotes",
      "InferenceFramework", "SoftwareLibraries", "SoftwareNotes",
  ]
  fmt = ",".join(["\"{}\""] * len(head)) + "\n"
  csv.write(",".join(head) + "\n")
  results = {}

  def log_result(submitter,
                 available,
                 division,
                 system_type,
                 system_desc,
                 model_name,
                 mlperf_model,
                 scenario_fixed,
                 r,
                 acc,
                 system_json,
                 name,
                 compliance,
                 errors,
                 config,
                 inferred=0,
                 power_metric=0):

    notes = system_json.get("hw_notes", "")
    if system_json.get("sw_notes"):
      notes = notes + ". " if notes else ""
      notes = notes + system_json.get("sw_notes")
    unit_dict = {
        "": "inf./sec.",
        "SingleStream": "Latency (ms)",
        "MultiStream": "Latency (ms)",
        "Offline": "Samples/s",
        "Server": "Queries/s",
    }
    power_unit_dict = {
        "": "uJ/inf.",
        "SingleStream": "millijoules",
        "MultiStream": "millijoules",
        "Offline": "Watts",
        "Server": "Watts",
    }
    unit = unit_dict[scenario_fixed]
    power_unit = power_unit_dict[scenario_fixed]

    csv.write(
        fmt.format(submitter,
                   available,
                   division,
                   system_json.get("Board Name"),
                   system_desc,
                   model_name,
                   mlperf_model,
                   r,
                   unit,
                   acc,
                   power_metric > 0,
                   power_metric,
                   power_unit,
                   system_json.get("Processor(s) Name"), # HostProcessorModelName
                   system_json.get("Processor(s) Frequencies"),
                   system_json.get("Processor memory type and capacity"),
                   system_json.get("Accelerator"),
                   system_json.get("Accelerator(s) Frequencies"),
                   system_json.get("Accelerator memory type and capacity"),
                   system_json.get("Hardware Notes"),
                   system_json.get("Inference Framework"),
                   system_json.get("Software Libraries"),
                   system_json.get("Software Notes"),
                ))

    # if power_metric > 0:
    #   csv.write(
    #       fmt.format(submitter, available, division, '\"' + system_type + '\"',
    #                  '\"' + system_name + '\"', system_desc, model_name,
    #                  mlperf_model, scenario_fixed, power_metric, acc,
    #                  system_json.get("number_of_nodes"),
    #                  '"' + system_json.get("host_processor_model_name") + '"',
    #                  system_json.get("host_processors_per_node"),
    #                  system_json.get("host_processor_core_count"),
    #                  '"' + system_json.get("accelerator_model_name") + '"',
    #                  '"' + str(system_json.get("accelerators_per_node")) + '"',
    #                  name.replace("\\", "/"),
    #                  '"' + system_json.get("framework", "") + '"',
    #                  '"' + system_json.get("operating_system", "") + '"',
    #                  '"' + notes + '"', compliance, errors, config.version,
    #                  inferred, power_metric > 0, power_unit))

  # we are at the top of the submission directory
  for division in list_dir("."):
    # we are looking at ./$division, ie ./closed
    if division not in VALID_DIVISIONS:
      if division not in [".git", ".github", "assets", ".vscode"]:
        log.error("invalid division in input dir %s", division)
      continue
    is_closed_or_network = division in ["closed", "network"]

    for submitter in list_dir(division):
      # we are looking at ./$division/$submitter, ie ./closed/mlperf_org
      if filter_submitter and submitter != filter_submitter:
        continue
      results_path = os.path.join(division, submitter, "results")
      if not os.path.exists(results_path):
        continue

      ## Apply folder checks
      dirs = list_dirs_recursively(division, submitter)
      files = list_files_recursively(division, submitter)

      # Check symbolic links
      symbolic_links = [f for f in files if os.path.islink(f)]
      if len(symbolic_links) > 0:
        log.error(
          "%s/%s contains symbolic links: %s",
          division,
          submitter,
          symbolic_links,
        )
        results[f"{division}/{submitter}"] = None
        continue

      # Check for files over 50 MB
      files_over_size_limit = [f for f in files if os.path.getsize(f) > FILE_SIZE_LIMIT_MB * MB_TO_BYTES]
      if len(files_over_size_limit) > 0:
        log.error(
          "%s/%s contains files with size greater than 50 MB: %s",
          division,
          submitter,
          files_over_size_limit,
        )
        results[f"{division}/{submitter}"] = None
        continue

      # Check files and folders with git unfriendly names
      dir_names = [(dir_, dir_.split("/")[-1]) for dir_ in dirs]
      file_names = [(file_, file_.split("/")[-1]) for file_ in files]
      git_error_names = [name[0] for name in dir_names if name[1].startswith(".")] + [
        name[0] for name in file_names if name[1].startswith(".")
      ]
      if len(git_error_names) > 0:
        log.warning(
          "%s/%s contains files with git unfriendly name: %s",
          division,
          submitter,
          git_error_names,
        )
        # results[f"{division}/{submitter}"] = None
        # continue

      # Check files and folders with spaces names
      space_error_names = [name[0] for name in dir_names if " " in name[1]] + [
        name[0] for name in file_names if " " in name[1]
      ]
      if len(space_error_names) > 0:
        log.warning(
          "%s/%s contains files with spaces in their names: %s",
          division,
          submitter,
          space_error_names,
        )
        # results[f"{division}/{submitter}"] = None
        # continue

      # Check for pycache folders
      pycache_dirs = [dir for dir in dirs if dir.endswith("__pycache__")]
      if len(pycache_dirs) > 0:
        log.error(
          "%s/%s has the following __pycache__ directories: %s",
          division,
          submitter,
          pycache_dirs,
        )
        results[f"{division}/{submitter}"] = None
        continue

      # Check for empty folders
      empty_dirs = list_empty_dirs_recursively(division, submitter)
      if len(empty_dirs) > 0:
        log.error(
          "%s/%s has the following empty directories: %s", 
          division,
          submitter, 
          empty_dirs
        )
        results[f"{division}/{submitter}"] = None
        continue

      # Check for extra model mapping
      extra_model_mapping = None
      if division == "open":
        model_mapping_path = f"{division}/{submitter}/{config.extra_model_benchmark_map}"
        if os.path.exists(model_mapping_path):
          with open(model_mapping_path) as fp:
            extra_model_mapping = json.load(fp)

      for system_desc in list_dir(results_path):
        # we are looking at ./$division/$submitter/results/$system_desc, ie ./closed/mlperf_org/results/t4-ort

        #
        # check if system_id is good.
        #
        system_id_json = os.path.join(division, submitter, "systems",
                                      system_desc + ".json")
        if not os.path.exists(system_id_json):
          log.error("no system_desc for %s/%s/%s", division, submitter,
                    system_desc)
          results[os.path.join(results_path, system_desc)] = None
          continue

        name = os.path.join(results_path, system_desc)
        with open(system_id_json) as f:
          system_json = json.load(f)
          available = system_json.get("Status").lower()
          if available not in VALID_AVAILABILITIES:
            log.error("%s has invalid status (%s)", system_id_json, available)
            results[name] = None
            continue
          system_type = system_json.get("system_type")
          config.set_type(system_type)

        #
        # Look at each model
        #
        for model_name in list_dir(results_path, system_desc):

          # we are looking at ./$division/$submitter/results/$system_desc/$model,
          #   ie ./closed/mlperf_org/results/t4-ort/bert
          name = os.path.join(results_path, system_desc, model_name)
          if os.path.exists(os.path.join(name, "EEMBC_RUNNER")):
            runner_type = "EEMBC_RUNNER"
            REQUIRED_ACC_FILES = EEMBC_REQUIRED_ACC_FILES
            REQUIRED_PERF_FILES = EEMBC_REQUIRED_PERF_FILES
            REQUIRED_PERF_POWER_FILES = EEMBC_REQUIRED_PERF_POWER_FILES
          else:
            runner_type = "MLC_RUNNER"
            REQUIRED_ACC_FILES = MLC_REQUIRED_ACC_FILES
            REQUIRED_PERF_FILES = MLC_REQUIRED_PERF_FILES
            REQUIRED_PERF_POWER_FILES = MLC_REQUIRED_PERF_POWER_FILES
          mlperf_model = config.get_mlperf_model(model_name, extra_model_mapping)

          if is_closed_or_network and mlperf_model not in config.models:
            # for closed/network divisions we want the model name to match.
            # for open division the model_name might be different than the task
            log.error("%s has an invalid model %s for closed/network division",
                      name, model_name)
            results[name] = None
            continue

          #
          # Look at each scenario
          #
          required_scenarios = config.get_required(mlperf_model)

          errors = 0
          # all_scenarios = set(
          #     # list(required_scenarios) +
          #     list(config.get_optional(mlperf_model)))
          for scenario in [""]:
            scenario_fixed = scenario
            # check accuracy
            accuracy_is_valid = False
            acc_path = os.path.join(name, "accuracy")

            if "accuracy" not in config.base["required_tests"][model_name]:
              pass # accuracy run is not required for this benchmark
            elif not os.path.exists(os.path.join(acc_path, ACC_FILE)):
              log.error(
                  "%s has no results.txt.", acc_path)
            else:
              diff = files_diff(list_files(acc_path), REQUIRED_ACC_FILES)
              if diff:
                log.error("%s has file list mismatch (%s)", acc_path, diff)
              accuracy_is_valid, acc = check_accuracy_dir(
                  config, mlperf_model, acc_path, debug or is_closed_or_network)
              if not accuracy_is_valid and not is_closed_or_network:
                if debug:
                  log.warning("%s, accuracy not valid but taken for open",
                              acc_path)
                accuracy_is_valid = True
              if not accuracy_is_valid:
                # a little below we'll not copy this into the results csv
                errors += 1
                log.error("%s, accuracy not valid", acc_path)

            inferred = 0
            # if scenario in ["Server"] and config.version in ["v0.5", "v0.7"]:
            #   n = ["run_1", "run_2", "run_3", "run_4", "run_5"]
            # else:
            #   n = ["run_1"]
            n = [""]

            # check if this submission has power logs
            power_path = os.path.join(name, "energy")
            has_power = os.path.exists(power_path)
            if has_power:
              log.info("Detected power logs for %s", name)

            for i in n:
              perf_path = os.path.join(name, "performance", i)
              has_performance = os.path.exists(perf_path)
              requires_performance = "performance" in config.base["required_tests"][model_name]
              allowed_tests = set.union(
                set(config.base["required_tests"][model_name]), 
                set(config.base["optional_tests"][model_name])
                )
              
              if has_power:
                required_perf_files = REQUIRED_PERF_FILES + REQUIRED_PERF_POWER_FILES
              else:
                required_perf_files = REQUIRED_PERF_FILES

              if requires_performance and not has_performance:
                log.error("%s is missing", perf_path)
                continue
              if has_performance and "performance" not in allowed_tests:
                log.warning("performance is not a valid test for %s.  Ignoring %s", model_name, perf_path)
              elif has_performance:
                diff = files_diff(list_files(perf_path), 
                                  required_perf_files, 
                                  OPTIONAL_PERF_FILES
                                  )
                if diff:
                  log.error("%s has file list mismatch (%s)", perf_path, diff)

                try:
                  is_valid, r, is_inferred = check_performance_dir(
                      config, mlperf_model, perf_path, scenario_fixed, division,
                      system_json)
                  if is_inferred:
                    inferred = 1
                    log.info("%s has inferred results, qps=%s", perf_path, r)
                except Exception as e:
                  log.error("%s caused exception in check_performance_dir: %s",
                            perf_path, str(e))
                  is_valid, r = False, None

              power_metric = 0
              if has_power:
                try:
                  ranging_path = os.path.join(name, "performance", "ranging")
                  power_is_valid, power_metric = check_power_dir(
                      power_path, scenario_fixed,
                      config)
                  if not power_is_valid:
                    is_valid = False
                    power_metric = 0
                except Exception as e:
                  log.error("%s caused exception in check_power_dir: %s",
                            perf_path, e)
                  is_valid, r, power_metric = False, None, 0

              if is_valid:
                results[
                    name] = r if r is None or power_metric == 0 else ("{:f} "
                                                                      "with "
                                                                      "power_metric"
                                                                      " = {:f}").format(
                        r, power_metric)
                # required_scenarios.discard(scenario_fixed)
              else:
                log.error("%s has issues", perf_path)
                errors += 1

            # check if compliance dir is good for CLOSED division
            compliance = 0 if is_closed_or_network else 1
            if is_closed_or_network and not skip_compliance:
              compliance_dir = os.path.join(division, submitter, "compliance",
                                            system_desc, model_name, scenario)
              if not os.path.exists(compliance_dir):
                log.error("no compliance dir for %s", name)
                results[name] = None
              else:
                if not check_compliance_dir(compliance_dir, mlperf_model,
                                            scenario_fixed, config, division,
                                            system_json):
                  log.error("compliance dir %s has issues", compliance_dir)
                  results[name] = None
                else:
                  compliance = 1

            if results.get(name):
              if accuracy_is_valid:
                log_result(
                    submitter,
                    available,
                    division,
                    system_type,
                    system_desc,
                    model_name,
                    mlperf_model,
                    scenario_fixed,
                    r,
                    acc,
                    system_json,
                    name,
                    compliance,
                    errors,
                    config,
                    inferred=inferred,
                    power_metric=power_metric)
              else:
                results[name] = None
                log.error("%s is OK but accuracy has issues", name)

          if required_scenarios:
            name = os.path.join(results_path, system_desc, model_name)
            if is_closed_or_network:
              results[name] = None
              log.error("%s does not have all required scenarios, missing %s",
                        name, required_scenarios)
            elif debug:
              log.warning("%s ignoring missing scenarios in open division (%s)",
                          name, required_scenarios)

  return results


def main():
  args = get_args()

  config = Config(
      args.version,
      args.extra_model_benchmark_map,
      ignore_uncommited=args.submission_exceptions,
      more_power_check=args.more_power_check)

  # compliance not yet supported
  args.skip_compliance = True

  with open(args.csv, "w") as csv:
    os.chdir(args.input)
    # check results directory
    results = check_results_dir(config, args.submitter, args.skip_compliance,
                                csv, args.debug)

  # log results
  log.info("---")
  with_results = 0
  for k, v in sorted(results.items()):
    if v:
      log.info("Results %s %s", k, v)
      with_results += 1
  log.info("---")
  for k, v in sorted(results.items()):
    if v is None:
      log.error("NoResults %s", k)

  # print summary
  log.info("---")
  log.info("Results=%d, NoResults=%d", with_results,
           len(results) - with_results)
  if len(results) != with_results:
    log.error("SUMMARY: submission has errors")
    return 1
  else:
    log.info("SUMMARY: submission looks OK")
    return 0

def check_accuracy_dir(config, model, path, verbose):
  is_valid = False
  acc = None
  hash_val = None
  acc_type, acc_target = config.get_accuracy_target(model)
  pattern = ACC_PATTERN[acc_type]
  with open(os.path.join(path, ACC_FILE), "r", encoding="utf-8") as f:
    for line in f:
      m = re.match(pattern, line)
      if m:
        acc = m.group(1)

  if acc and float(acc) >= acc_target:
    is_valid = True
  elif verbose:
    log.warning("%s accuracy not met: expected=%f, found=%s", path, acc_target,
                acc)

  # check if there are any errors in the detailed log
  # fname = os.path.join(path, "mlperf_log_detail.txt")
  # if not find_error_in_detail_log(config, fname):
  #   is_valid = False

  return is_valid, acc

def check_performance_dir(config, model, path, scenario_fixed, division,
                          system_json):
  is_valid = False
  res = None
  inferred = False

  fname = os.path.join(path, "results.txt")
  with open(fname, "r") as f:
    for line in f:
      m = re.match(r".* Median throughput is\s+([\d\.]+) inf\./sec\..*", line)
      if m:
        is_valid = True
        res = m.group(1)
  return is_valid, float(res), inferred

def check_power_dir(power_path, scenario_fixed,
                    config):

  is_valid = False
  res = None

  fname = os.path.join(power_path, "results.txt")
  with open(fname, "r") as f:
    for line in f:
      m = re.search(r"Median energy cost is ([\d\.]+) uJ/inf", line)
      if m:
        is_valid = True
        res = m.group(1)
  return is_valid, float(res)

def files_diff(list1, list2, optional=None):
  """returns a list of files that are missing or added."""
  if not optional:
    optional = []
  optional = optional + ["mlperf_log_trace.json", "results.json", ".gitkeep"]
  return set(list1).symmetric_difference(set(list2)) - set(optional)

if __name__ == "__main__":
  main_result = main()
  print(f"function main() returned {main_result}")
  # sys.exit(main_result)
