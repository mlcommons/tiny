import csv
import os
from random import randrange


class DataSet:
  def __init__(self, dataset_path, truth_file):
    self._dataset_path = dataset_path
    self._truth_file = truth_file
    self._truth_data = None

  def _read_truth_file(self):
    if not self._truth_data:
      with open(os.path.join(self._dataset_path, self._truth_file)) as file:
        reader = csv.DictReader(file, fieldnames=["file", "classes", "class"])
        self._truth_data = [f for f in reader]

  def get_file_by_index(self, index):
    self._read_truth_file()
    data = []
    index = index if index is not None else 0 # was randrange(len(self._truth_data))
    truth = None
    if index < len(self._truth_data):
      truth = self._truth_data[index]
      with open(os.path.join(self._dataset_path, truth.get("file")), "rb") as file:
        data = file.read()
    return truth, data
