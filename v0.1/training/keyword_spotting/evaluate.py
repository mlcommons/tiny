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
  Flags, unparsed = kws_util.parse_command()

  print('We will download data to {:}'.format(Flags.data_dir))

  ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
  print("Done getting data")

  if Flags.model_init_path is None:
    print("Starting with untrained model. Accuracy will be random-guess level at best.")
    model = models.get_model(args=Flags)
  else:
    print(f"Starting with pre-trained model from {Flags.model_init_path}")
    model = keras.models.load_model(Flags.model_init_path)

    model.summary()
  
  test_scores = model.evaluate(ds_test, return_dict=True)
  print("Test loss:", test_scores['loss'])
  print("Test accuracy:", test_scores['sparse_categorical_accuracy'])


  outputs = np.zeros((0,num_classes))
  labels = np.array([])
  for samples, batch_labels in ds_test:
    outputs = np.vstack((outputs, model.predict(samples)))
    labels = np.hstack((labels, batch_labels))

  predictions = np.argmax(outputs, axis=1)
  print("==== EEMBC calculate_accuracy Method ====")
  accuracy_eembc = eembc_ev.calculate_accuracy(outputs, labels)
  print(40*"=")

  print("==== SciKit-Learn AUC ====")
  auc_scikit = roc_auc_score(labels, outputs, multi_class='ovr')
  print(f"AUC (sklearn) = {auc_scikit}")
  print(40*"=")
  
  print("==== EEMBC calculate_auc ====")
  label_names = ["go", "left", "no", "off", "on", "right",
                 "stop", "up", "yes", "silence", "unknown"]

  auc_eembc = eembc_ev.calculate_auc(outputs, labels, label_names, Flags.model_architecture)
  print("---------------------")
