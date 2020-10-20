#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np


# I'd like to use this in ds_XXX.map(), but currently it doesn't work
def spec_feats(sample_dict): 
  """Runs TFL microfrontend and returns spectrogram"""
  audio = sample_dict['audio']
  label = sample_dict['label']
  paddings = [[0, 16000-tf.shape(audio)[0]]]
  audio = tf.pad(audio, paddings)
  audio16 = tf.cast(audio, 'int16')    
  spec = frontend_op.audio_microfrontend(audio16, sample_rate=16000, window_size=40, 
                                           window_step=20, num_channels=40)
  return spec, label

def convert_to_int16(sample_dict):
  audio = sample_dict['audio']
  label = sample_dict['label']
  audio16 = tf.cast(audio, 'int16')
  return audio16, label

def cast_and_pad(sample_dict):
  audio = sample_dict['audio']
  label = sample_dict['label']
  paddings = [[0, 16000-tf.shape(audio)[0]]]
  audio = tf.pad(audio, paddings)
  audio16 = tf.cast(audio, 'int16')
  return audio16, label

def wavds2specds(ds_wav):
  """ Convert a dataset of waveforms into a dataset of spectrograms
  """
  specgrams = []
  labels = []
  
  # cnt=0
  for cnt, (wav, label) in enumerate(ds_wav):    
    if wav.shape != (16000,) or label.shape != ():
        print(f"In Loop Shape is wrong at {cnt}: {wav.shape}, {label.shape}")
    cnt += 1
    spec = frontend_op.audio_microfrontend(wav, sample_rate=16000, window_size=40, window_step=20, num_channels=40)
    spec = tf.cast(spec, 'float32') / 1000.0
    specgrams.append(spec)
    # label = keras.utils.to_categorical(label, num_classes)
    labels.append(label)
    if (cnt % 250) == 0: 
      print(f"Converted {cnt} samples to spectrogram")


  ds_specs = tf.data.Dataset.from_tensor_slices((specgrams, labels))
  return ds_specs

def get_training_data(FLAGS):
  splits = ['train', 'test', 'validation']
  (ds_train, ds_test, ds_val), ds_info = tfds.load('speech_commands', split=splits, 
                                           data_dir=FLAGS.data_dir, with_info=True)
  ## Options: 
  #     split=None,
  #     data_dir=None,
  #     batch_size=1,
  #     download=True,
  #     as_supervised=False,
  #     with_info=False,
  #     builder_kwargs=None,
  #     download_and_prepare_kwargs=None,
  #     as_dataset_kwargs=None,


  ds_train = ds_train.map(cast_and_pad, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_test = ds_test.map(cast_and_pad, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_val = ds_val.map(cast_and_pad, num_parallel_calls=tf.data.experimental.AUTOTUNE)


  ds_train_specs = wavds2specds(ds_train)
  ds_train_specs = ds_train_specs.shuffle(ds_info.splits['train'].num_examples).batch(FLAGS.batch_size)
  # ds_train = ds_train.cache()
  # ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  ds_test_specs = wavds2specds(ds_test)
  ds_test_specs  = ds_test_specs.shuffle(ds_info.splits['test'].num_examples).batch(FLAGS.batch_size)  

  ds_val_specs = wavds2specds(ds_val)
  ds_val_specs   = ds_val_specs.shuffle(ds_info.splits['validation'].num_examples).batch(FLAGS.batch_size)

  print("created and batched dataset")
  
  return ds_train_specs, ds_test_specs, ds_val_specs
