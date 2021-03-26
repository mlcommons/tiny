#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.platform import gfile

import matplotlib.pyplot as plt
import numpy as np
import os, pickle
import glob
import collections # needed for default dictionary, mapping label strings to numbers

import kws_util
import keras_model as models

label_ids = collections.defaultdict(lambda : 11, {"down":0, "go":1, "left":2, "no":3, "off":4, "on":5, 
                                               "right":6, "stop":7, "up":8, "yes":9, "silence":10, "unknown":11})

labels_str = list(label_ids.keys())

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

def convert_dataset(item):
    """Puts the dataset in the format Keras expects, (features, labels)."""
    audio = item['audio']
    label = item['label']
    return audio, label

def get_preprocess_audio_func(model_settings,is_training=False,background_data = []):
    def prepare_processing_graph(next_element):
        """Builds a TensorFlow graph to apply the input distortions.
        Creates a graph that loads a WAVE file, decodes it, scales the volume,
        shifts it in time, adds in background noise, calculates a spectrogram, and
        then builds an MFCC fingerprint from that.
        This must be called with an active TensorFlow session running, and it
        creates multiple placeholder inputs, and one output:
          - wav_filename_placeholder_: Filename of the WAV to load.
          - foreground_volume_placeholder_: How loud the main clip should be.
          - time_shift_padding_placeholder_: Where to pad the clip.
          - time_shift_offset_placeholder_: How much to move the clip in time.
          - background_data_placeholder_: PCM sample data for background noise.
          - background_volume_placeholder_: Loudness of mixed-in background.
          - mfcc_: Output 2D fingerprint of processed audio.
        Args:
          model_settings: Information about the current model being trained.
        """
        desired_samples = model_settings['desired_samples']
        background_frequency = model_settings['background_frequency']
        background_volume_range_= model_settings['background_volume_range_']

        wav_decoder = tf.cast(next_element['audio'], tf.float32)
        wav_decoder = wav_decoder/tf.reduce_max(wav_decoder)
        wav_decoder = tf.pad(wav_decoder,[[0,desired_samples-tf.shape(wav_decoder)[-1]]]) #Previously, decode_wav was used with desired_samples as the length of array. The default option of this function was to pad zeros if the desired samples are not found
        # Allow the audio sample's volume to be adjusted.
        foreground_volume_placeholder_ = tf.constant(1,dtype=tf.float32)
        
        scaled_foreground = tf.multiply(wav_decoder,
                                        foreground_volume_placeholder_)
        # Shift the sample's start position, and pad any gaps with zeros.
        time_shift_padding_placeholder_ = tf.constant([[2,2]], tf.int32)
        time_shift_offset_placeholder_ = tf.constant([2],tf.int32)
        scaled_foreground.shape
        padded_foreground = tf.pad(scaled_foreground, time_shift_padding_placeholder_, mode='CONSTANT')
        sliced_foreground = tf.slice(padded_foreground, time_shift_offset_placeholder_, [desired_samples])
        
    
        if is_training and background_data != []:
            background_volume_range = tf.constant(background_volume_range_,dtype=tf.float32)
            background_index = np.random.randint(len(background_data))
            background_samples = background_data[background_index]
            background_offset = np.random.randint(0, len(background_samples) - desired_samples)
            background_clipped = background_samples[background_offset:(background_offset + desired_samples)]
            background_clipped = tf.squeeze(background_clipped)
            background_reshaped = tf.pad(background_clipped,[[0,desired_samples-tf.shape(wav_decoder)[-1]]])
            background_reshaped = tf.cast(background_reshaped, tf.float32)
            if np.random.uniform(0, 1) < background_frequency:
                background_volume = np.random.uniform(0, background_volume_range_)
            else:
                background_volume = 0
            background_volume_placeholder_ = tf.constant(background_volume,dtype=tf.float32)
            background_data_placeholder_ = background_reshaped
            background_mul = tf.multiply(background_data_placeholder_,
                                 background_volume_placeholder_)
            background_add = tf.add(background_mul, sliced_foreground)
            sliced_foreground = tf.clip_by_value(background_add, -1.0, 1.0)
        
        stfts = tf.signal.stft(sliced_foreground, frame_length=model_settings['window_size_samples'], 
                               frame_step=model_settings['window_stride_samples'], fft_length=None,
                               window_fn=tf.signal.hann_window
                               )
        spectrograms = tf.abs(stfts)
        num_spectrogram_bins = stfts.shape[-1]
        #default values used by contrib_audio.mfcc as shown here https://kite.com/python/docs/tensorflow.contrib.slim.rev_block_lib.contrib_framework_ops.audio_ops.mfcc
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40 
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix( num_mel_bins, num_spectrogram_bins, model_settings['sample_rate'],
                                                                            lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        # Compute MFCCs from log_mel_spectrograms and take the first 13.
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :model_settings['dct_coefficient_count']]
        mfccs = tf.reshape(mfccs,[model_settings['spectrogram_length'], model_settings['dct_coefficient_count'], 1])
        next_element['audio'] = mfccs
        #next_element['label'] = tf.one_hot(next_element['label'],12)
        return next_element
    
    return prepare_processing_graph


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

def prepare_background_data(bg_path,BACKGROUND_NOISE_DIR_NAME):
    """Searches a folder for background noise audio, and loads it into memory.
    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
    the sample rate of the training data, but can be much longer in duration.
    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.
    Returns:
      List of raw PCM-encoded audio samples of background noise.
    Raises:
      Exception: If files aren't found in the folder.
    """
    background_data = []
    background_dir = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME)
    if not os.path.exists(background_dir):
      return background_data
    #with tf.Session(graph=tf.Graph()) as sess:
    #    wav_filename_placeholder = tf.placeholder(tf.string, [])
    #    wav_loader = io_ops.read_file(wav_filename_placeholder)
    #    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    search_path = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME,'*.wav')
    #for wav_path in gfile.Glob(search_path):
    #    wav_data = sess.run(wav_decoder, feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
    #    self.background_data.append(wav_data)
    for wav_path in gfile.Glob(search_path):
        #audio = tfio.audio.AudioIOTensor(wav_path)
        raw_audio = tf.io.read_file(wav_path)
        audio = tf.audio.decode_wav(raw_audio)
        background_data.append(audio[0])
    if not background_data:
        raise Exception('No background wav files were found in ' + search_path)
    return background_data

def maybe_download_data(Flags):
  pass # for now, manually download and extract data before running this
  # data_url = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
  # data_dir = os.path.join(Flags.data_dir, 'speech_commands_v0.02')
  # tf.keras.utils.get_file('speech_commands_v0.02.tar.gz', origin=data_url,
  #                         extract=True, cache_dir=data_dir)



def get_waveform_and_label(file_path):
  # modified from https://www.tensorflow.org/tutorials/audio/simple_audio
  ### JHH Right here is where you are working on.
  parts = tf.strings.split(file_path, os.path.sep)
  label_str = parts[-2]  # .numpy().decode('utf-8')
  unknown_bias = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0.1]) # so if nothing matches, select 11(unknown)
  label_id = tf.argmax(tf.cast(label_str == labels_str, tf.float32) + unknown_bias)
  audio_binary = tf.io.read_file(file_path)
  audio, _ = tf.audio.decode_wav(audio_binary)
  waveform = tf.squeeze(audio, axis=-1)
  
  return {'audio': waveform, 'label':label_id}


def files_to_datasets(Flags):
  data_path = os.path.join(Flags.data_dir, 'speech_commands_v0.02')
  fname_val_files = os.path.join(data_path, 'validation_list.txt')
  fname_test_files = os.path.join(data_path, 'testing_list.txt')

  with open(fname_val_files) as fpi_val:
    val_files = fpi_val.read().splitlines()
  # validation_list.txt only lists partial paths
  val_files = [os.path.join(data_path, fn) for fn in val_files]

  with open(fname_test_files) as fpi_tst:
    test_files = fpi_tst.read().splitlines()
  # testing_list.txt only lists partial paths
  test_files = [os.path.join(data_path, fn) for fn in test_files]
  
  all_files = glob.glob(data_path + '/*/*.wav')
  # exclude the _background_noise_ files
  train_files = [f for f in all_files if f.split('/')[-2][0] != '_']
  # validation and test files are listed explicitly in *_list.txt; train with everything else
  train_files = list(set(train_files) - set(test_files) - set(val_files))
  
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  ds_train_files = tf.data.Dataset.from_tensor_slices(train_files)
  ds_test_files  = tf.data.Dataset.from_tensor_slices(test_files)
  ds_val_files   = tf.data.Dataset.from_tensor_slices(val_files)

  ds_train = ds_train_files.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  ds_test  = ds_test_files.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  ds_val   = ds_val_files.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

  return (ds_train, ds_test, ds_val)

  
def get_training_data(Flags, get_waves=False):
  spectrogram_length = int((Flags.clip_duration_ms - Flags.window_size_ms +
                            Flags.window_stride_ms) / Flags.window_stride_ms)
  
  dct_coefficient_count=Flags.dct_coefficient_count 
  window_size_ms=Flags.window_size_ms 
  window_stride_ms=Flags.window_stride_ms
  clip_duration_ms=Flags.clip_duration_ms #expected duration in ms
  sample_rate=Flags.sample_rate
  label_count=12
  background_frequency = Flags.background_frequency
  background_volume_range_= Flags.background_volume
  model_settings = models.prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                               window_size_ms, window_stride_ms,
                               dct_coefficient_count,background_frequency)
  # this is taken from the dataset web page.
  # there should be a better way than hard-coding this
  train_shuffle_buffer_size = 85511
  val_shuffle_buffer_size = 10102
  test_shuffle_buffer_size = 4890
  bg_path=Flags.bg_path
  BACKGROUND_NOISE_DIR_NAME='_background_noise_' 
  background_data = prepare_background_data(bg_path,BACKGROUND_NOISE_DIR_NAME)
  splits = ['train', 'test', 'validation']
  # (ds_train, ds_test, ds_val), ds_info = tfds.load('speech_commands', split=splits, 
  #                                               data_dir=Flags.data_dir, with_info=True)
  (ds_train, ds_test, ds_val) = files_to_datasets(Flags)
  # ds_train = ds_train.shuffle(train_shuffle_buffer_size)
  # ds_val = ds_val.shuffle(val_shuffle_buffer_size)
  # ds_test = ds_test.shuffle(test_shuffle_buffer_size)

  if Flags.num_train_samples != -1:
    ds_train = ds_train.take(Flags.num_train_samples)
  if Flags.num_val_samples != -1:
    ds_val = ds_val.take(Flags.num_val_samples)
  if Flags.num_test_samples != -1:
    ds_test = ds_test.take(Flags.num_test_samples)
    
  if get_waves:
    ds_train = ds_train.map(cast_and_pad)
    ds_test  =  ds_test.map(cast_and_pad)
    ds_val   =   ds_val.map(cast_and_pad)
  else:
    ds_train = ds_train.map(get_preprocess_audio_func(model_settings,is_training=True,
                                                      background_data=background_data),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test  =  ds_test.map(get_preprocess_audio_func(model_settings,is_training=False,
                                                      background_data=background_data),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val   =   ds_val.map(get_preprocess_audio_func(model_settings,is_training=False,
                                                      background_data=background_data),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.map(convert_dataset)
    ds_test = ds_test.map(convert_dataset)
    ds_val = ds_val.map(convert_dataset)

  # Now that we've acquired the preprocessed data, either by processing or loading,
  ds_train = ds_train.batch(Flags.batch_size)
  ds_test = ds_test.batch(Flags.batch_size)  
  ds_val = ds_val.batch(Flags.batch_size)
  
  return ds_train, ds_test, ds_val


def create_c_files(dataset, root_filename="input_data", interpreter=None, elems_per_row=10):
  print("** WARNING **: This routine (create_c_files) is not completed")

  preamble = """
/* Copyright 2020 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/// \file
/// \brief Sample inputs for the audio wakewords model.
#include "kws/kws_input_data.h"
const int8_t g_kws_inputs[kNumKwsTestInputs][kKwsInputSize] = {
    {
  """

  dataset = dataset.unbatch().batch(1).as_numpy_iterator()

  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  input_scale, input_zero_point = input_details[0]["quantization"]

  input_type = np.int8
  
  dat, label = next(dataset)
  dat_q = np.array(dat/input_scale + input_zero_point, dtype=input_type)
  dat_q = dat_q.flatten()

  print("dat_q is type {:}".format(type(dat_q)))
  print("dat_q is shape {:}".format(dat_q.shape))
  print(f"Writing to {ofname}")
  num_elems = len(dat_q)

  with open(f"{root_filename}.cc", "w") as fpo:
    fpo.write(preamble)
    for cnt, val in enumerate(dat_q):
      if cnt < num_elems-1:
        fpo.write("{:d},".format(val))
      else:
        fpo.write("{:d}".format(val)) # last element => no comma
      if cnt+1 % elems_per_row == 0:
        fpo.write("\n")
    fpo.write("}};\n")

if __name__ == '__main__':
  Flags, unparsed = kws_util.parse_command()
  ds_train, ds_test, ds_val = get_training_data(Flags)

  if Flags.create_c_files:
    if Flags.target_set[0:3].lower() == 'val':
      target_data = ds_val
      print("Drawing from the  validation set")
    elif Flags.target_set[0:4].lower() == 'test':
      target_data = ds_test
      print("Drawing from the test set")
    elif Flags.target_set[0:5].lower() == 'train':
      target_data = ds_train    
      print("Drawing from  the training set")

    interpreter = tf.lite.Interpreter(model_path=Flags.tfl_file_name)

    create_c_files(dataset=target_data,
                   root_filename="kws_input_data",
                   interpreter=interpreter)
