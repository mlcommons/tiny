#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.platform import gfile

import functools

import matplotlib.pyplot as plt
import numpy as np
import os, pickle

import kws_util
import keras_model as models

word_labels = ["Down", "Go", "Left", "No", "Off", "On", "Right",
               "Stop", "Up", "Yes", "Silence", "Unknown"]

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
  """Puts the mnist dataset in the format Keras expects, (features, labels)."""
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
    if model_settings['feature_type'] != "td_samples":
      wav_decoder = wav_decoder/tf.reduce_max(wav_decoder)
    else:
      wav_decoder = wav_decoder/tf.constant(2**15,dtype=tf.float32)
    #Previously, decode_wav was used with desired_samples as the length of array. The
    # default option of this function was to pad zeros if the desired samples are not found
    wav_decoder = tf.pad(wav_decoder,[[0,desired_samples-tf.shape(wav_decoder)[-1]]]) 
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
    
    if model_settings['feature_type'] == 'mfcc':
      stfts = tf.signal.stft(sliced_foreground, frame_length=model_settings['window_size_samples'], 
                         frame_step=model_settings['window_stride_samples'], fft_length=None,
                         window_fn=tf.signal.hann_window
                         )
      spectrograms = tf.abs(stfts)
      num_spectrogram_bins = stfts.shape[-1]
      # default values used by contrib_audio.mfcc as shown here
      # https://kite.com/python/docs/tensorflow.contrib.slim.rev_block_lib.contrib_framework_ops.audio_ops.mfcc
      lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40 
      linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix( num_mel_bins, num_spectrogram_bins,
                                                                           model_settings['sample_rate'],
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

    elif model_settings['feature_type'] == 'lfbe':
      # apply preemphasis
      preemphasis_coef = 1 - 2 ** -5
      power_offset = 52
      num_mel_bins = model_settings['dct_coefficient_count']
      paddings = tf.constant([[0, 0], [1, 0]])
      # for some reason, tf.pad only works with the extra batch dimension, but then we remove it after pad
      sliced_foreground = tf.expand_dims(sliced_foreground, 0)
      sliced_foreground = tf.pad(tensor=sliced_foreground, paddings=paddings, mode='CONSTANT')
      sliced_foreground = sliced_foreground[:, 1:] - preemphasis_coef * sliced_foreground[:, :-1]
      sliced_foreground = tf.squeeze(sliced_foreground) 
      # compute fft
      stfts = tf.signal.stft(sliced_foreground,  frame_length=model_settings['window_size_samples'], 
                             frame_step=model_settings['window_stride_samples'], fft_length=None,
                             window_fn=functools.partial(
                               tf.signal.hamming_window, periodic=False),
                             pad_end=False,
                             name='STFT')
    
      # compute magnitude spectrum [batch_size, num_frames, NFFT]
      magspec = tf.abs(stfts)
      num_spectrogram_bins = magspec.shape[-1]
    
      # compute power spectrum [num_frames, NFFT]
      powspec = (1 / model_settings['window_size_samples']) * tf.square(magspec)
      powspec_max = tf.reduce_max(input_tensor=powspec)
      powspec = tf.clip_by_value(powspec, 1e-30, powspec_max) # prevent -infinity on log
    
      def log10(x):
        # Compute log base 10 on the tensorflow graph.
        # x is a tensor.  returns log10(x) as a tensor
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
    
      # Warp the linear-scale, magnitude spectrograms into the mel-scale.
      lower_edge_hertz, upper_edge_hertz = 0.0, model_settings['sample_rate'] / 2.0
      linear_to_mel_weight_matrix = (
          tf.signal.linear_to_mel_weight_matrix(
              num_mel_bins=num_mel_bins,
              num_spectrogram_bins=num_spectrogram_bins,
              sample_rate=model_settings['sample_rate'],
              lower_edge_hertz=lower_edge_hertz,
              upper_edge_hertz=upper_edge_hertz))

      mel_spectrograms = tf.tensordot(powspec, linear_to_mel_weight_matrix,1)
      mel_spectrograms.set_shape(magspec.shape[:-1].concatenate(
          linear_to_mel_weight_matrix.shape[-1:]))

      log_mel_spec = 10 * log10(mel_spectrograms)
      log_mel_spec = tf.expand_dims(log_mel_spec, -1, name="mel_spec")
    
      log_mel_spec = (log_mel_spec + power_offset - 32 + 32.0) / 64.0
      log_mel_spec = tf.clip_by_value(log_mel_spec, 0, 1)

      next_element['audio'] = log_mel_spec

    elif model_settings['feature_type'] == 'td_samples':
      ## sliced_foreground should have the right data.  Make sure it's the right format (int16)
      # and just return it.
      paddings = [[0, 16000-tf.shape(sliced_foreground)[0]]]
      wav_padded = tf.pad(sliced_foreground, paddings)
      wav_padded = tf.expand_dims(wav_padded, -1)
      wav_padded = tf.expand_dims(wav_padded, -1)
      next_element['audio'] = wav_padded
      
    return next_element
  
  return prepare_processing_graph


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


def get_training_data(Flags, get_waves=False, val_cal_subset=False):
  
  label_count=12
  background_frequency = Flags.background_frequency
  background_volume_range_= Flags.background_volume
  model_settings = models.prepare_model_settings(label_count, Flags)

  bg_path=Flags.bg_path
  BACKGROUND_NOISE_DIR_NAME='_background_noise_' 
  background_data = prepare_background_data(bg_path,BACKGROUND_NOISE_DIR_NAME)

  splits = ['train', 'test', 'validation']
  (ds_train, ds_test, ds_val), ds_info = tfds.load('speech_commands', split=splits, 
                                                data_dir=Flags.data_dir, with_info=True)

  if val_cal_subset:  # only return the subset of val set used for quantization calibration
    with open("quant_cal_idxs.txt") as fpi:
      cal_indices = [int(line) for line in fpi]
    cal_indices.sort()
    # cal_indices are the positions of specific inputs that are selected to calibrate the quantization
    count = 0  # count will be the index into the validation set.
    val_sub_audio = []
    val_sub_labels = []
    for d in ds_val:
      if count in cal_indices:          # this is one of the calibration inpus
        new_audio = d['audio'].numpy()  # so add it to a stack of tensors 
        if len(new_audio) < 16000:      # from_tensor_slices doesn't work for ragged tensors, so pad to 16k
          new_audio = np.pad(new_audio, (0, 16000-len(new_audio)), 'constant')
        val_sub_audio.append(new_audio)
        val_sub_labels.append(d['label'].numpy())
      count += 1
    # and create a new dataset for just the calibration inputs.
    ds_val = tf.data.Dataset.from_tensor_slices({"audio": val_sub_audio,
                                                 "label": val_sub_labels})

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
    # extract spectral features and add background noise
    ds_train = ds_train.map(get_preprocess_audio_func(model_settings,is_training=True,
                                                      background_data=background_data),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test  =  ds_test.map(get_preprocess_audio_func(model_settings,is_training=False,
                                                      background_data=background_data),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val   =   ds_val.map(get_preprocess_audio_func(model_settings,is_training=False,
                                                      background_data=background_data),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # change output from a dictionary to a feature,label tuple
    ds_train = ds_train.map(convert_dataset)
    ds_test = ds_test.map(convert_dataset)
    ds_val = ds_val.map(convert_dataset)

  # Now that we've acquired the preprocessed data, either by processing or loading,
  ds_train = ds_train.batch(Flags.batch_size)
  ds_test = ds_test.batch(Flags.batch_size)  
  ds_val = ds_val.batch(Flags.batch_size)
  
  return ds_train, ds_test, ds_val


if __name__ == '__main__':
  Flags, unparsed = kws_util.parse_command()
  ds_train, ds_test, ds_val = get_training_data(Flags)

  for dat in ds_train.take(1):
    print("One element from the training set has shape:")
    print(f"Input tensor shape: {dat[0].shape}")
    print(f"Label shape: {dat[1].shape}")
