#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tensorflow import keras

import keras_model as models
import get_dataset as kws_data
import kws_util

num_classes = 12 # should probably draw this directly from the dataset.
# FLAGS = None

if __name__ == '__main__':
  Flags, unparsed = kws_util.parse_command()

  print('We will download data to {:}'.format(Flags.data_dir))
  print('We will train for {:} epochs'.format(Flags.epochs))

  ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
  print("Done getting data")

  # this is taken from the dataset web page.
  # there should be a better way than hard-coding this
  train_shuffle_buffer_size = 85511
  val_shuffle_buffer_size = 10102
  test_shuffle_buffer_size = 4890

  ds_train = ds_train.shuffle(train_shuffle_buffer_size)
  ds_val = ds_val.shuffle(val_shuffle_buffer_size)
  ds_test = ds_test.shuffle(test_shuffle_buffer_size)
  
  if Flags.model_init_path is None:
    print("Starting with untrained model")
    model = models.get_model(args=Flags)
  else:
    print(f"Starting with pre-trained model from {Flags.model_init_path}")
    model = keras.models.load_model(Flags.model_init_path)

  model.summary()
  
  callbacks = kws_util.get_callbacks(args=Flags)
  train_hist = model.fit(ds_train, validation_data=ds_val, epochs=Flags.epochs, callbacks=callbacks)
  kws_util.plot_training(Flags.plot_dir,train_hist)
  model.save(Flags.saved_model_path)
  
  if Flags.run_test_set:
    test_scores = model.evaluate(ds_test)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
