#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

import models, aww_data

num_classes = 12 # should probably draw this directly from the dataset.
# FLAGS = None

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('HOME'), 'data'),
      help="""\
      Where to download the speech training data to. Or where it is already saved.
      """)
  parser.add_argument(
      '--preprocessed_data_dir',
      type=str,
      default=os.path.join(os.getenv('HOME'), 'data/speech_commands_preprocessed'),
      help="""\
      Where to store preprocessed speech data (spectrograms) or load it, if it exists
      with the same parameters as are used in the current run.
      """)
  parser.add_argument(
      '--save_preprocessed_data',
      type=bool,
      default=True,
      help="""\
      Where to download the speech training data to. Or where it is already saved.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=20.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--epochs',
      type=int,
      default=50,
      help='How many epochs to train',)
  parser.add_argument(
      '--num_train_samples',
      type=int,
      default=85511,
    help='How many samples from the training set to use',)
  parser.add_argument(
      '--num_val_samples',
      type=int,
      default=10102,
    help='How many samples from the validation set to use',)
  parser.add_argument(
      '--num_test_samples',
      type=int,
      default=4890,
    help='How many samples from the test set to use',)
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='ds_cnn',
      help='What model architecture to use')
  parser.add_argument(
      '--saved_model_path',
      type=str,
      default='pretrained_model',
      help='File name to load pretrained model')
  parser.add_argument(
      '--run_test_set',
      type=bool,
      default=True,
      help='Run model.eval() on test set if True')

  Flags, unparsed = parser.parse_known_args()

  print('We will download data to {:}'.format(Flags.data_dir))
  print('we will train for {:} epochs'.format(Flags.epochs))
  ds_train, ds_test, ds_val = aww_data.get_training_data(Flags)
  print("Done getting data")
  model = models.get_model(model_name=Flags.model_architecture)
  model.summary()
  

  train_hist = model.fit(ds_train, validation_data=ds_val, epochs=Flags.epochs)
  model.save(Flags.saved_model_path)
  
  if Flags.run_test_set:
    test_scores = model.evaluate(ds_test)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
