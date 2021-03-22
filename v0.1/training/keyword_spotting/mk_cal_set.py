#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tensorflow import keras
from sklearn.metrics import roc_auc_score

import keras_model as models
import get_dataset as kws_data
import kws_util
import eval_functions_eembc as eembc_ev

num_classes = 12 # should probably draw this directly from the dataset.
# FLAGS = None

if __name__ == '__main__':
  fname = 'quant_cal_idxs.txt'
  num_cal_files = 50
  Flags, unparsed = kws_util.parse_command()

  print('We will download data to {:}'.format(Flags.data_dir))

  ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
  print("Done getting data")
  
  labels = np.array([])
  for _, batch_labels in ds_train:
    labels = np.hstack((labels, batch_labels))

  np.random.seed(31) # seed = 31 => each class has >= 3 samples
  idxs = np.random.choice(len(labels), size=num_cal_files, replace=False)
  print(f"Validation set has {len(labels)} samples.")
  print(f"Writing {num_cal_files} indices into validation set to file {fname} ...", end='')
  with open(fname, 'w') as fpo:
    for num in idxs:
      fpo.write(f"{num}\n")
  print(".. Done")
      
