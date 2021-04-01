#!/usr/bin/env python

import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score

import get_dataset as kws_data
import kws_util

num_classes = 12 # should probably draw this directly from the dataset.
# FLAGS = None

if __name__ == '__main__':
  fname = 'quant_cal_idxs.txt'
  num_files_per_label = 10

  Flags, unparsed = kws_util.parse_command()

  np.random.seed(2)
  tf.random.set_seed(2)

  print('We will download data to {:}'.format(Flags.data_dir))

  ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
  print("Done getting data")
  
  labels = np.array([])
  for _, batch_labels in ds_val:
    labels = np.hstack((labels, batch_labels))

  cal_idxs = np.array([], dtype=int)
  for l in np.unique(labels):
    all_label_idxs = np.nonzero(labels==l)[0] # nonzero => tuple of arrays; get the first/only one
    sel_label_idxs = np.random.choice(all_label_idxs, size=num_files_per_label, replace=False)
    cal_idxs = np.concatenate((cal_idxs, sel_label_idxs))

  print(f"Validation set has {len(labels)} samples.")
  print(f"Writing {len(cal_idxs)} indices into validation set to file {fname} ...", end='')
  with open(fname, 'w') as fpo:
    for num in cal_idxs:
      fpo.write(f"{num}\n")
  print(".. Done")
      
