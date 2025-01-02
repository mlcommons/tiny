import csv
import os
from random import randrange


class DataSet:
  def __init__(self, dataset_path, truth_file, start_index=0):
    self._dataset_path = dataset_path
    self._truth_file = truth_file
    self._truth_data = None
    self._current_index = start_index

  def _read_truth_file(self):
    if not self._truth_data:
      with open(os.path.join(self._dataset_path, self._truth_file)) as file:
        reader = csv.DictReader(file, fieldnames=["file", "classes", "class", "bytes_to_send","stride"])
        self._truth_data = [f for f in reader]

  def get_file_by_index(self, index):
    self._read_truth_file()
    data = []
    # if an index is not specified, use the object's _current_index and increment it,
    # if one is specified, use that and don't increment _current_index
    if index is not None:
      inc_index = False
    else:
      index = self._current_index
      inc_index = True

    truth = None
    if index < len(self._truth_data):
      truth = self._truth_data[index]
      with open(os.path.join(self._dataset_path, truth.get("file")), "rb") as file:
        data = file.read()
    if inc_index:
      self._current_index += 1

    return truth, data