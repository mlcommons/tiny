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
import aww_util

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
    image = item['audio']
    label = item['label']
    return image, label

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                               window_size_ms, window_stride_ms,
                               dct_coefficient_count,background_frequency):
      """Calculates common settings needed for all models.
      Args:
        label_count: How many classes are to be recognized.
        sample_rate: Number of audio samples per second.
        clip_duration_ms: Length of each audio clip to be analyzed.
        window_size_ms: Duration of frequency analysis window.
        window_stride_ms: How far to move in time between frequency windows.
        dct_coefficient_count: Number of frequency bins to use for analysis.
      Returns:
        Dictionary containing common settings.
      """
      desired_samples = int(sample_rate * clip_duration_ms / 1000)
      window_size_samples = int(sample_rate * window_size_ms / 1000)
      window_stride_samples = int(sample_rate * window_stride_ms / 1000)
      length_minus_window = (desired_samples - window_size_samples)
      if length_minus_window < 0:
        spectrogram_length = 0
      else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
      fingerprint_size = dct_coefficient_count * spectrogram_length
      return {
          'desired_samples': desired_samples,
          'window_size_samples': window_size_samples,
          'window_stride_samples': window_stride_samples,
          'spectrogram_length': spectrogram_length,
          'dct_coefficient_count': dct_coefficient_count,
          'fingerprint_size': fingerprint_size,
          'label_count': label_count,
          'sample_rate': sample_rate,
          'background_frequency': 0.8,
          'background_volume_range_': 0.1
      }

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


def get_training_data(Flags):

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
  model_settings = prepare_model_settings(label_count, sample_rate, clip_duration_ms,
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
  (ds_train, ds_test, ds_val), ds_info = tfds.load('speech_commands', split=splits, 
                                                data_dir=Flags.data_dir, with_info=True)

  ds_train = ds_train.shuffle(train_shuffle_buffer_size)
  ds_val = ds_val.shuffle(val_shuffle_buffer_size)
  ds_test = ds_test.shuffle(test_shuffle_buffer_size)

  if Flags.num_train_samples != -1:
     ds_train = ds_train.take(Flags.num_train_samples)
  if Flags.num_val_samples != -1:
     ds_val = ds_val.take(Flags.num_val_samples)
  if Flags.num_test_samples != -1:
     ds_test = ds_test.take(Flags.num_test_samples)
    
  ds_train_specs = ds_train.map(get_preprocess_audio_func(model_settings,is_training=True,background_data=background_data), num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_test_specs = ds_test.map(get_preprocess_audio_func(model_settings,is_training=False,background_data=background_data), num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_val_specs = ds_val.map(get_preprocess_audio_func(model_settings,is_training=False,background_data=background_data), num_parallel_calls=tf.data.experimental.AUTOTUNE)

  ds_train_specs = ds_train_specs.map(convert_dataset)
  ds_test_specs = ds_test_specs.map(convert_dataset)
  ds_val_specs = ds_val_specs.map(convert_dataset)
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

if __name__ == '__main__':
  Flags, unparsed = aww_util.parse_command()
  ds_train, ds_test, ds_val = get_training_data(Flags)
