#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import os, pickle


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

def wavds2specds(ds_wav, Flags):
  """ Convert a dataset of waveforms into a dataset of spectrograms
  """
  specgrams = []
  labels = []
  
  # cnt=0
  for cnt, (wav, label) in enumerate(ds_wav):    
    if wav.shape != (16000,) or label.shape != ():
        print(f"In Loop Shape is wrong at {cnt}: {wav.shape}, {label.shape}")
    cnt += 1
    spec = frontend_op.audio_microfrontend(wav, sample_rate=Flags.sample_rate,
                                           window_size=Flags.window_size_ms,
                                           window_step=Flags.window_stride_ms,
                                           num_channels=Flags.dct_coefficient_count)
    spec = tf.cast(spec, 'float32') / 1000.0
    specgrams.append(spec)
    # label = keras.utils.to_categorical(label, num_classes)
    labels.append(label)
    if (cnt % 250) == 0: 
      print(f"Converted {cnt} samples to spectrogram")
  print(f"Finished converting {cnt} samples to spectrogram.")
  ds_specs = tf.data.Dataset.from_tensor_slices((specgrams, labels))
  return ds_specs


def good_preproc_data_available(Flags):
  # Check whether preprocessed data is available in the Flags.preprocessed_data_dir with
  # the same preprocessing parameters (window stride, window size, feature count, sample rate)
  fname_params = os.path.join(Flags.preprocessed_data_dir, 'preproc_params.pkl')

  try:
    with open(fname_params, 'rb') as fpi:
      preproc_params = pickle.load(fpi)
    good_data = preproc_params['clip_duration_ms']       == Flags.clip_duration_ms and \
                (preproc_params['dct_coefficient_count']  == Flags.dct_coefficient_count) and \
                (preproc_params['sample_rate']            == Flags.sample_rate) and \
                (preproc_params['window_size_ms']         == Flags.window_size_ms) and \
                (preproc_params['window_stride_ms']       == Flags.window_stride_ms) and \
                (preproc_params['num_test_samples']       >= Flags.num_test_samples) and \
                (preproc_params['num_train_samples']      >= Flags.num_train_samples) and \
                (preproc_params['num_val_samples']        >= Flags.num_val_samples)
  except(FileNotFoundError, pickle.UnpicklingError, KeyError):
    good_data = False # the params file was not there, wasn't a pickle, or was missing a key
  return good_data

def get_training_data(Flags):

  spectrogram_length = int((Flags.clip_duration_ms- (Flags.window_size_ms - Flags.window_stride_ms)) /
                           Flags.window_stride_ms)
  
  ##  this is taken from the dataset web page.  there should be a better way than hard-coding this
  train_shuffle_buffer_size = 85511
  val_shuffle_buffer_size = 10102
  test_shuffle_buffer_size = 4890
  
  running_preprocessor=False
  if not good_preproc_data_available(Flags):
    running_preprocessor=True
    
    splits = ['train', 'test', 'validation']
    (ds_train, ds_test, ds_val), ds_info = tfds.load('speech_commands', split=splits, 
                                                data_dir=Flags.data_dir, with_info=True)

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
  

    ds_train = ds_train.shuffle(train_shuffle_buffer_size)
    ds_val = ds_val.shuffle(val_shuffle_buffer_size)
    ds_test = ds_test.shuffle(test_shuffle_buffer_size)

    if Flags.num_train_samples != -1:
      ds_train = ds_train.take(Flags.num_train_samples)
    if Flags.num_val_samples != -1:
      ds_val = ds_val.take(Flags.num_val_samples)
    if Flags.num_test_samples != -1:
      ds_test = ds_test.take(Flags.num_test_samples)
    
    ds_train = ds_train.map(cast_and_pad, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(cast_and_pad, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(cast_and_pad, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_train_specs = wavds2specds(ds_train, Flags)
    ds_test_specs = wavds2specds(ds_test, Flags)
    ds_val_specs = wavds2specds(ds_val, Flags)

    print("created and batched dataset")
  else: # there is good preprocessed data available
    print("Good preprocessed data available. We'll use that.")
    with open(os.path.join(Flags.preprocessed_data_dir, 'preproc_params.pkl'), 'rb') as fpi:
      preproc_params = pickle.load(fpi)

    ds_train_specs = tf.data.experimental.load(os.path.join(Flags.preprocessed_data_dir, 'train'),
                                               preproc_params['train_elem_spec'])
    ds_val_specs   = tf.data.experimental.load(os.path.join(Flags.preprocessed_data_dir, 'val'),
                                               preproc_params['val_elem_spec'])
    ds_test_specs  = tf.data.experimental.load(os.path.join(Flags.preprocessed_data_dir, 'test'),
                                               preproc_params['test_elem_spec'])
    if Flags.num_train_samples != -1:
      ds_train_specs = ds_train_specs.take(Flags.num_train_samples)
    if Flags.num_val_samples != -1:
      ds_val_specs = ds_val_specs.take(Flags.num_val_samples)
    if Flags.num_test_samples != -1:
      ds_test_specs = ds_test_specs.take(Flags.num_test_samples)

  if running_preprocessor and Flags.save_preprocessed_data:    
    print("Saving preprocessed data to {:}".format(Flags.preprocessed_data_dir))
    
    tf.data.experimental.save(ds_train_specs, os.path.join(Flags.preprocessed_data_dir, 'train'))
    tf.data.experimental.save(ds_val_specs, os.path.join(Flags.preprocessed_data_dir, 'val'))
    tf.data.experimental.save(ds_test_specs, os.path.join(Flags.preprocessed_data_dir, 'test'))

    preproc_params = {}
    preproc_params['train_elem_spec'] = ds_train_specs.element_spec
    preproc_params['val_elem_spec']   = ds_val_specs.element_spec
    preproc_params['test_elem_spec']  = ds_test_specs.element_spec    

    preproc_params['clip_duration_ms']       = Flags.clip_duration_ms          
    preproc_params['dct_coefficient_count']  = Flags.dct_coefficient_count     
    preproc_params['num_test_samples']       = Flags.num_test_samples          
    preproc_params['num_train_samples']      = Flags.num_train_samples         
    preproc_params['num_val_samples']        = Flags.num_val_samples           
    preproc_params['sample_rate']            = Flags.sample_rate               
    preproc_params['window_size_ms']         = Flags.window_size_ms            
    preproc_params['window_stride_ms']       = Flags.window_stride_ms         

    with open(os.path.join(Flags.preprocessed_data_dir, 'preproc_params.pkl'), 'wb') as fpo:
      pickle.dump(preproc_params, fpo)

  elif running_preprocessor:
    print("We have preprocessed the data, but we're not saving it")

  # Now that we've acquired the preprocessed data, either by processing or loading,
  ds_train_specs = ds_train_specs.shuffle(train_shuffle_buffer_size)
  ds_train_specs = ds_train_specs.batch(Flags.batch_size)
  # ds_train = ds_train.cache()
  # ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  ds_test_specs = ds_test_specs.shuffle(Flags.num_test_samples)
  ds_test_specs = ds_test_specs.batch(Flags.batch_size)  

  ds_val_specs = ds_val_specs.shuffle(Flags.num_val_samples)
  ds_val_specs = ds_val_specs.batch(Flags.batch_size)


  return ds_train_specs, ds_test_specs, ds_val_specs
