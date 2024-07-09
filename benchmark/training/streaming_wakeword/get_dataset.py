#!/usr/bin/env python

import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.platform import gfile

import functools

import matplotlib.pyplot as plt
import numpy as np
import os, pickle, glob, time, copy

import str_ww_util as util

rng = np.random.default_rng(2024)
word_labels = ["Marvin", "Silence", "Unknown"]

def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)
  
def get_label(file_path):
  parts = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-2]
  
def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return {'audio':waveform, 'label':label}

def convert_labels_str2int(datum):
  """
  datum is {'audio':<audio>, 'label':<label_as_string>} 
  returns {'audio':<audio>, 'label':<label_as_int>}
  according to:
        keys=tf.constant(["marvin", "_silence", "_unknown"]),
        values=tf.constant([0, 1, 2]),
  """
  # build a lookup table
  label_map = tf.lookup.StaticHashTable(
      initializer=tf.lookup.KeyValueTensorInitializer(
          keys=tf.constant(["marvin", "_silence", "_unknown"]),
          values=tf.constant([0, 1, 2]),
      ),
      default_value=tf.constant(2), # map other labels to _unknown
      name="class_labels"
  )
  return {'audio': datum['audio'], 
          'label':label_map.lookup(datum['label'])
         }

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
  label = tf.one_hot(item['label'], depth=3, axis=-1, )
  return audio, label


def get_preprocess_audio_func(data_config, background_data = [], wave_frame_input=False):
  """
  data_config                      :  Data configuration
  background_data = [],            :  TF Dataset of background noise waveforms
  wave_frame_input (default False) :  If True, build a dedicated model to extract features.  This is mostly
                                      intended to be exported to a TFLite model that can run on the DUT
  """
  # building a TF graph here, but do not use tf.function, because the
  # if statements should be evaluated at graph construction time, not evaluation time
  def prepare_processing_graph(next_element):
    """Builds a TensorFlow graph to apply the input distortions.
    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, adds in background noise, calculates a spectrogram, and
    then builds an feature tensor from that.
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
      log_mel_spec: Information about the how the data should be processed
    """
    desired_samples = data_config['desired_samples']
    background_frequency = data_config['background_frequency']
    background_volume_range_= data_config['background_volume']
    foreground_volume_min_ = data_config['foreground_volume_min']
    foreground_volume_max_ = data_config['foreground_volume_max']

    if wave_frame_input:
      wav_decoder = tf.cast(next_element, tf.float32)
    else:
      wav_decoder = tf.cast(next_element['audio'], tf.float32)
    
    if not wave_frame_input: # don't rescale if we're only processing one frame of samples
      wav_decoder = wav_decoder - tf.reduce_mean(wav_decoder)
      wav_decoder = wav_decoder/tf.reduce_max(wav_decoder)

    if data_config['feature_type'] == "td_samples":
      wav_decoder = wav_decoder/tf.constant(2**15,dtype=tf.float32)

    if wave_frame_input:
      sliced_foreground = wav_decoder
    else:
      wav_decoder = tf.pad(wav_decoder,[[0,desired_samples-tf.shape(wav_decoder)[-1]]]) 
      
      # Allow the audio sample's volume to be adjusted.
      foreground_volume_placeholder_ = tf.random.uniform([1],minval=foreground_volume_min_,maxval=foreground_volume_max_)[0]
      scaled_foreground = tf.multiply(wav_decoder, foreground_volume_placeholder_)

      # Shift the sample's start position, and pad any gaps with zeros.
      time_shift_padding_placeholder_ = tf.constant([[2,2]], tf.int32)
      time_shift_offset_placeholder_ = tf.constant([2],tf.int32)
      
      padded_foreground = tf.pad(scaled_foreground, time_shift_padding_placeholder_, mode='CONSTANT')
      sliced_foreground = tf.slice(padded_foreground, time_shift_offset_placeholder_, [desired_samples])
    if data_config['background_frequency'] != 0 and background_data != []:
      background_volume_range = tf.constant(background_volume_range_,dtype=tf.float32)
      background_index = tf.random.uniform([1],minval=0,maxval=len(background_data), dtype=tf.int32)[0]
      background_samples = tf.gather(background_data, background_index) # graph-compatible equivalent to background_data[background_index]
      background_offset = tf.random.uniform([1],minval=0,maxval=len(background_samples) - desired_samples, dtype=tf.int32)[0]
      background_clipped = background_samples[background_offset:(background_offset + desired_samples)]
      background_clipped = tf.squeeze(background_clipped)
      background_reshaped = tf.pad(background_clipped,[[0,desired_samples-tf.shape(wav_decoder)[-1]]])
      background_reshaped = tf.cast(background_reshaped, tf.float32)
      
      bg_rand_draw = tf.random.uniform([1],minval=0.0,maxval=1.0)[0]
      background_volume_placeholder_ = tf.cond( # replace if block, so tf graph works
        bg_rand_draw < background_frequency, 
        lambda: tf.random.uniform([1],minval=0.5,maxval=background_volume_range_, dtype=tf.float32)[0],
        lambda: tf.constant(0.0, dtype=tf.float32)
      )

      background_data_placeholder_ = background_reshaped
      background_mul = tf.multiply(background_data_placeholder_,
                           background_volume_placeholder_)
      background_add = tf.add(background_mul, sliced_foreground)
      sliced_foreground = tf.clip_by_value(background_add, -1.0, 1.0)

    if data_config['feature_type'] == 'lfbe':
      # apply preemphasis
      preemphasis_coef = 1 - 2 ** -5
      power_offset = 52
      num_mel_bins = data_config['dct_coefficient_count']
      paddings = tf.constant([[0, 0], [1, 0]])
      # for some reason, tf.pad only works with the extra batch dimension, but then we remove it after pad
      if not wave_frame_input: # the feature extractor comes in already batched
        sliced_foreground = tf.expand_dims(sliced_foreground, 0)
      sliced_foreground = tf.pad(tensor=sliced_foreground, paddings=paddings, mode='CONSTANT')
      sliced_foreground = sliced_foreground[:, 1:] - preemphasis_coef * sliced_foreground[:, :-1]
      sliced_foreground = tf.squeeze(sliced_foreground)
      
      stfts = tf.signal.stft(sliced_foreground,  frame_length=data_config['window_size_samples'], 
                             frame_step=data_config['window_stride_samples'], fft_length=None,
                             window_fn=functools.partial(
                               tf.signal.hamming_window, periodic=False),
                             pad_end=False,
                             name='STFT')
    
      # compute magnitude spectrum [batch_size, num_frames, NFFT]
      magspec = tf.abs(stfts)
      num_spectrogram_bins = magspec.shape[-1]
    
      # compute power spectrum [num_frames, NFFT]
      powspec = (1 / data_config['window_size_samples']) * tf.square(magspec)
      powspec_max = tf.reduce_max(input_tensor=powspec)
      powspec = tf.clip_by_value(powspec, 1e-30, powspec_max) # prevent -infinity on log
    
      def log10(x):
        # Compute log base 10 on the tensorflow graph.
        # x is a tensor.  returns log10(x) as a tensor
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
    
      # Warp the linear-scale, magnitude spectrograms into the mel-scale.
      lower_edge_hertz, upper_edge_hertz = 0.0, data_config['sample_rate'] / 2.0
      linear_to_mel_weight_matrix = (
          tf.signal.linear_to_mel_weight_matrix(
              num_mel_bins=num_mel_bins,
              num_spectrogram_bins=num_spectrogram_bins,
              sample_rate=data_config['sample_rate'],
              lower_edge_hertz=lower_edge_hertz,
              upper_edge_hertz=upper_edge_hertz))

      mel_spectrograms = tf.tensordot(powspec, linear_to_mel_weight_matrix,1)
      mel_spectrograms.set_shape(magspec.shape[:-1].concatenate(
          linear_to_mel_weight_matrix.shape[-1:]))

      log_mel_spec = 10 * log10(mel_spectrograms)
      log_mel_spec = tf.expand_dims(log_mel_spec, -2, name="mel_spec")
    
      log_mel_spec = (log_mel_spec + power_offset - 32 + 32.0) / 64.0
      log_mel_spec = tf.clip_by_value(log_mel_spec, 0, 1)

      if wave_frame_input: 
        # for building a feature extractor model, we need to return a tensor
        next_element = log_mel_spec
      else:
        # for building a dataset, we're returning a dictionary
        next_element['audio'] = log_mel_spec

    elif model_settings['feature_type'] == 'td_samples':
      ## sliced_foreground should have the right data.  Make sure it's the right format (int16)
      # and just return it.
      paddings = [[0, 16000-tf.shape(sliced_foreground)[0]]]
      wav_padded = tf.pad(sliced_foreground, paddings)
      wav_padded = tf.expand_dims(wav_padded, -1)
      wav_padded = tf.expand_dims(wav_padded, -1)
      next_element['audio'] = wav_padded
    else:
      raise ValueError(f"Invalid value {data_config['feature_type']}.  Should be one of 'lfbe', 'td_samples'.")
      
    return next_element
  
  return prepare_processing_graph


def prepare_background_data(background_path, clip_len_samples):
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

  if background_path is None:
    return []
  elif isinstance(background_path, str):
    background_path = [background_path]
  elif not isinstance(background_path, list):
    raise ValueError(f"background_path should be either a string or list of strings")

  background_data = []
  # In order to work with tf.gather later, the background audio all need to be of the same length
  # skip any shorter than clip_len_samples.  For files over clip_len_samples, split into as 
  # many clips as possible of length clip_len_samples, discard any fractional bits
  # so e.g. w/ clip_len_samples=240,000 (15s at fs=16ksps), a 640000 sample file => 2 240-ksamp clips
  for dir in background_path:
    if not os.path.exists(dir):
      raise RuntimeError(f"Directory {dir} in background_path does not exist.")

    wav_list = gfile.Glob(os.path.join(dir, '*.wav'))
    if len(wav_list) == 0:
      raise RuntimeError(f"Directory {dir} in background_path contains no wav files.")

    for wav_path in wav_list: 
      raw_audio = tf.io.read_file(wav_path)
      audio, _ = tf.audio.decode_wav(raw_audio) # returns waveform, sample_rate
      if len(audio) < clip_len_samples:
        continue # file is too short, skip
      # split waveform into as many clips of length clip_len_samples as we can
      idx = 0
      while idx+clip_len_samples-1<len(audio):
        background_data.append(audio[idx:idx+clip_len_samples])
        idx += clip_len_samples
      if len(background_data) >= 50:
        break

  if len(background_data)==0:
    raise Exception('No background wav files were found in ' + background_path)
  return background_data

def prepare_background_data_ds(background_path, clip_len_samples):
  """ In-progress alternative version of this function that returns 
  the background data as a dataset.
  Searches a folder for background noise audio, and loads it into memory.
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

  if background_path is None:
    return []
  elif isinstance(background_path, str):
    background_path = [background_path]
  elif not isinstance(background_path, list):
    raise ValueError(f"background_path should be either a string or list of strings")

  background_data = []
  # In order to work with tf.gather later, the background audio all need to be of the same length
  # skip any shorter than clip_len_samples.  For files over clip_len_samples, split into as 
  # many clips as possible of length clip_len_samples, discard any fractional bits
  # so e.g. w/ clip_len_samples=240,000 (15s at fs=16ksps), a 640000 sample file => 2 240-ksamp clips


  wav_list = []
  for dir in background_path:
    if not os.path.exists(dir):
      raise RuntimeError(f"Directory {dir} in background_path does not exist.")
    wavs_this_dir = gfile.Glob(os.path.join(dir, '*.wav'))
    if len(wavs_this_dir) == 0:
      raise RuntimeError(f"Directory {dir} in background_path contains no wav files.")
    wav_list.append(wavs_this_dir)
    ds_bg_files = tf.data.Dataset.from_tensor_slices(wav_list)

  ## TODO: convert this to map operations
  ds_bg_files = ds_bg_files.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  
  ## get_waveform_and_label => return {'audio':waveform, 'label':label}
  # silent_wave_ds = silent_wave_ds.map(lambda dd: {
  #   'audio':tf.cast(dd['audio'], 'float32')  ,
  #   'label':tf.cast(dd['label'], 'int32')
  # })

  for wav_path in wav_list: 
    raw_audio = tf.io.read_file(wav_path)
    audio, _ = tf.audio.decode_wav(raw_audio) # returns waveform, sample_rate
    if len(audio) < clip_len_samples:
      continue # file is too short, skip
    # split waveform into as many clips of length clip_len_samples as we can
    idx = 0
    while idx+clip_len_samples-1<len(audio):
      background_data.append(audio[idx:idx+clip_len_samples])
      idx += clip_len_samples

  if len(background_data)==0:
    raise Exception('No background wav files were found in ' + background_path)
  # convert to tf dataset
  background_data = tf.data.Dataset.from_tensor_slices(background_data)

  return background_data

def add_empty_frames(ds, input_shape=None, num_silent=None, white_noise_scale=0.0, silent_label=1): 
  """
  ds: Dataset of dictionary elements with 'audio' and 'label'(int) keys.
  input_shape: shape of 'audio' elements
  num_silent: number of silenet elements to add 
  white_noise_scale: std dev of white gaussian noise added to each silent element.
  silent_label:  integer label paired with the silent elements
  
  Returns the original dataset concatenated with num_silent new elements.
  """
  rand_waves = rng.normal(loc=0.0, scale=white_noise_scale, size=(num_silent,)+tuple(input_shape))
  silent_labels = silent_label * np.ones(num_silent)
  silent_wave_ds = tf.data.Dataset.from_tensor_slices({'audio':rand_waves,
                                                       'label':silent_labels})
  silent_wave_ds = silent_wave_ds.map(lambda dd: {
    'audio':tf.cast(dd['audio'], 'float32')  ,
    'label':tf.cast(dd['label'], 'int32')
  })
  return ds.concatenate(silent_wave_ds)

def get_data_config(general_flags, split, cal_subset=False, wave_frame_input=False, **kwargs):
  """
  Builds a configuration (as a Namespace from argparse) containing all the flags for building
  a particular split (training, validation, test) of the dataset.
  general_flags            The Flags gathered from the command line with parse_command()
  split                    Generally either 'training', 'validation', or 'test'
  cal_subset=False         If True, return only the examples specified for calibration
  wave_frame_input=False   If true, return a configuration suitable for building a graph that
                           can extract features from a long waveform.
  **kwargs                 Any flags in kwargs are added to the config, overwriting any other value.
  """
  # collect the subset of flags necessary for building the datasets
  # does not include *_training, *_validation, or *_test flags
  data_keys =  [
    'data_dir', 'time_shift_ms', 
    'background_path', 'background_frequency',
    'sample_rate', 'clip_duration_ms',
    'window_size_ms', 'window_stride_ms',
    'feature_type', 'dct_coefficient_count',
    'batch_size',
    'num_bin_files', 'bin_file_path'
    ]
  # First populate the values that apply to all splits.  These can be overwritten
  # with either a split-specific flag (batch_size_validation) or with kwargs
  data_config = {k:general_flags.__dict__[k] for k in data_keys}
  data_config['desired_samples'] = int(general_flags.sample_rate * general_flags.clip_duration_ms / 1000)

  window_size_samples = int(general_flags.sample_rate * general_flags.window_size_ms / 1000)
  window_stride_samples = int(general_flags.sample_rate * general_flags.window_stride_ms / 1000)
  length_minus_window = (data_config['desired_samples'] - window_size_samples)
  if length_minus_window < 0:
    raise ValueError(
      f"Wave length {data_config['desired_samples']} is too short for window size {window_size_samples}"
      )
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  data_config["window_size_samples"] = window_size_samples
  data_config["window_stride_samples"] = window_stride_samples
  data_config["spectrogram_length"] = spectrogram_length
  if split == "test":
    # test split has no noise augmentation (validation split does).  This can
    # be overridden with an explicit background_frequency_test flag
    data_config['background_frequency'] = 0

  # shuffle training data, but not val/test.  Can be overridden w/ explicit shuffle_<split>
  data_config['shuffle'] = (split == "training")

  # any Flag ending in '_training' is written to the training config with the _training stripped
  # same for _validation, _test.  Could also be used for other splits if needed (e.g. _val2 should work)
  per_split_keys = [k for k in general_flags.__dict__.keys() if k.endswith(f"_{split}")]
  for k in per_split_keys:
    data_config[k.rsplit('_', 1)[0]] = general_flags.__dict__[k]

  # this builds a graph for extracting features from long recordings
  # Generally False except for the demo run on a long waveform
  data_config['wave_frame_input']=wave_frame_input
  data_config['cal_subset']=cal_subset

  # anything specified in kwargs overrides
  data_config.update(kwargs)

  # wrap to enable config.Flag style usage
  return util.DictWrapper(data_config)

def get_file_lists(data_dir):
  filenames = glob.glob(os.path.join(str(data_dir), '*', '*.wav'))
  # the full speech-commands set lists which files are to be used
  # as test and validation data; train with everything else
  
  fname_val_files = os.path.join(data_dir, 'validation_list.txt')    
  with open(fname_val_files) as fpi_val:
    val_files = fpi_val.read().splitlines()
  # validation_list.txt only lists partial paths, append to data_dir and sr
  val_files = [os.path.join(data_dir, fn).rstrip() for fn in val_files]
  
  # repeat for test files
  fname_test_files = os.path.join(data_dir, 'testing_list.txt')
  with open(fname_test_files) as fpi_tst:
    test_files = fpi_tst.read().splitlines()  
  test_files = [os.path.join(data_dir, fn).rstrip() for fn in test_files]    
    
  if os.sep != '/': 
    # the files validation_list.txt and testing_list.txt use '/' as path separator
    # if we're on a windows machine, replace the '/' with the correct separator
    val_files = [fn.replace('/', os.sep) for fn in val_files]
    test_files = [fn.replace('/', os.sep) for fn in test_files]
    
  # don't train with the _background_noise_ files; exclude when directory name starts with '_'
  train_files = [f for f in filenames if f.split(os.sep)[-2][0] != '_']
  # validation and test files are listed explicitly in *_list.txt; train with everything else
  train_files = list(set(train_files) - set(test_files) - set(val_files))

  return train_files, test_files, val_files

def get_all_datasets(Flags):

  flags_training = get_data_config(Flags, 'training')
  flags_validation = get_data_config(Flags, 'validation')
  flags_test = get_data_config(Flags, 'test')

  ## Build the data sets from files
  data_dir = Flags.data_dir
  train_files, test_files, val_files = get_file_lists(data_dir)

  ds_train = get_data(flags_training, train_files)
  ds_val = get_data(flags_validation, val_files)
  ds_test = get_data(flags_test, test_files)
  
  return ds_train, ds_test, ds_val

def get_data(Flags, file_list):
  
  label_count=3
  background_frequency = Flags.background_frequency
  background_volume_range_= Flags.background_volume
  AUTOTUNE = tf.data.AUTOTUNE

  background_data = prepare_background_data(
    Flags.background_path, 
    clip_len_samples=int(15.0*Flags.sample_rate)
    )

  dset = tf.data.Dataset.from_tensor_slices(file_list)
  
  dset = dset.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  dset = dset.map(convert_labels_str2int)

  if Flags.fraction_target > 0:
    # create copies of only the target words to balance the distribution.
    # noise will be added to them later
    dset_only_target = dset.filter(lambda dat: dat['label'] == 0)
    
    # filter() breaks cardinality(), so convert to an array and back again
    target_wavs = []
    target_labels = []
    for dat in dset_only_target:
      trg_wav = dat['audio']
      # also pad to desired length; a ragged array breaks this code
      trg_wav = np.pad(trg_wav, (0,Flags.desired_samples-len(trg_wav)))
      target_wavs.append(trg_wav)
      target_labels.append(dat['label'])

    dset_only_target = tf.data.Dataset.from_tensor_slices({'audio':target_wavs,
                                                      'label':target_labels})
    num_targets_orig = dset_only_target.cardinality()
    print(f"Only-target dataset has {num_targets_orig} elements.")

  if Flags.num_samples == -1:
    # how many non-target samples do we start with
    num_other = dset.cardinality() - dset_only_target.cardinality()
    # how many total samples will we have after adding silent, target up to the desired fractions
    num_samples = int(num_other.numpy()/ (1.0-Flags.fraction_silent-Flags.fraction_target))
  else:
    num_samples = Flags.num_samples
  num_silent = int(Flags.fraction_silent * num_samples)
  num_targets = int(Flags.fraction_target * num_samples)

  if Flags.cal_subset:  # only return the subset of val set used for quantization calibration
    with open("quant_cal_idxs.txt") as fpi:
      cal_indices = [int(line) for line in fpi]
    cal_indices.sort()
    # cal_indices are the positions of specific inputs that are selected to calibrate the quantization
    count = 0  # count will be the index into the validation set.
    cal_sub_audio = []
    cal_sub_labels = []
    for d in dset:
      if count in cal_indices:          # this is one of the calibration inpus
        new_audio = d['audio'].numpy()  # so add it to a stack of tensors 
        if len(new_audio) < 16000:      # from_tensor_slices doesn't work for ragged tensors, so pad to 16k
          new_audio = np.pad(new_audio, (0, 16000-len(new_audio)), 'constant')
        cal_sub_audio.append(new_audio)
        cal_sub_labels.append(d['label'].numpy())
      count += 1
    # and create a new dataset for just the calibration inputs.
    dset = tf.data.Dataset.from_tensor_slices({"audio": cal_sub_audio,
                                                 "label": cal_sub_labels})
    ## end of if Flags.cal_subset

  for dat in dset.take(1):
    wave_shape = dat['audio'].shape # we'll need this to build the silent dataset

  # create some silent samples, which noise will be added to as well
  if Flags.fraction_silent > 0:
    dset = add_empty_frames(dset, input_shape=wave_shape, num_silent=num_silent, 
                                white_noise_scale=0.1, silent_label=1)
  
  # add repetitions of the target to fill in the rest of num_samples
  while dset.cardinality()+num_targets_orig < num_samples:
    dset = dset.concatenate(dset_only_target)
  extra_targets_needed = dset.cardinality() - num_samples
  if extra_targets_needed > 0:
    dset = dset.concatenate(dset_only_target.take(extra_targets_needed))

  print(f"About to apply preprocessor with {dset.cardinality()} samples")
  # extract spectral features and add background noise
  audio_preprocessor = get_preprocess_audio_func(Flags,
                                            background_data=background_data)
  dset = dset.map(audio_preprocessor, num_parallel_calls=AUTOTUNE)

  # change output from a dictionary to a feature,label tuple
  dset = dset.map(convert_dataset)

  # The order of these next three steps is important: cache, then shuffle, then batch.
  # Cache at this point, so we don't have to repeat all the spectrogram calculations each epoch
  dset = dset.cache()

  if Flags.shuffle:
    # count the number of items in the training set.
    # this will take some time, but it reduces the time in the 1st epoch
    shuffle_buffer_size = dset.reduce(0, lambda x,_: x+1).numpy()  
    dset = dset.shuffle(shuffle_buffer_size)
    print(f"Shuffling with buffers size = {shuffle_buffer_size}")
  else:
    print(f"not shuffling")

  if Flags.num_samples != -1:
    dset = dset.take(Flags.num_samples)
  
  dset = dset.batch(Flags.batch_size)

  return dset

def is_batched(ds):
  ## This feels wrong/not robust, but I can't find a better way
  try:
    ds.unbatch()  # does not actually change ds. For that we would ds=ds.unbatch()
  except:
    return False # we'll assume that the error on unbatching is because the ds is not batched.
  else:
    return True  # if we were able to unbatch it then it must have been batched (??)

def count_labels(ds, label_index=1):
  """
  returns a dictionary with each found unique label as key and
  the number of samples with that label as value.
  label_index: key to index the label from each item in the dataset.
  if each item is a tuple/list, label_index should be an integer
  if each item is a dict, label_index should be the key.
  """
  if is_batched(ds):
    ds = ds.unbatch()
  
  label_counts = {}
  for dat in ds:
    new_label = dat[label_index].numpy()
    if new_label in label_counts:
      label_counts[new_label] += 1
    else:
      label_counts[new_label] = 1
  return label_counts

if __name__ == '__main__':
  Flags = util.parse_command()
  ds_train, ds_test, ds_val = get_all_datasets(Flags)

  for dat in ds_train.take(1):
    print("One element from the training set has shape:")
    print(f"Input tensor shape: {dat[0].shape}")
    print(f"Label shape: {dat[1].shape}")
  print(f"Number of each class in training set:\n {count_labels(ds_train, label_index=1)}")
