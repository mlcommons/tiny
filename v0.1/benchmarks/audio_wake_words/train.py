#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

import keras_model as models
import get_dataset as aww_data
import aww_util
from lr import get_callbacks
from plot import plot

num_classes = 12 # should probably draw this directly from the dataset.
# FLAGS = None

if __name__ == '__main__':
  Flags, unparsed = aww_util.parse_command()

  print('We will download data to {:}'.format(Flags.data_dir))
  print('We will train for {:} epochs'.format(Flags.epochs))

  ds_train, ds_test, ds_val = aww_data.get_training_data(Flags)
  print("Done getting data")
  model = models.get_model(args=Flags)
  model.summary()
  
  callbacks = get_callbacks(args=Flags)
  train_hist = model.fit(ds_train, validation_data=ds_val, epochs=Flags.epochs, callbacks=callbacks)
  plot(Flags.plot_dir,train_hist)
  model.save(Flags.saved_model_path)
  
  if Flags.run_test_set:
    test_scores = model.evaluate(ds_test)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
