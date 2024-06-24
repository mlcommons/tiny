#!/usr/bin/env python

import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.python.platform import gfile

import tensorflow_model_optimization as tfmot

from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_quantize_configs import (
    NoOpQuantizeConfig,
)
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_quantize_registry import (
    DefaultNBitConvQuantizeConfig,
    DefaultNBitQuantizeConfig,
)
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_quantize_scheme import (
    DefaultNBitQuantizeScheme,
)


import matplotlib.pyplot as plt
import numpy as np
import platform
import json

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2

def prepare_model_settings(label_count, args):
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
  desired_samples = int(args.sample_rate * args.clip_duration_ms / 1000)
  if args.feature_type == 'td_samples':
    window_size_samples = 1
    spectrogram_length = desired_samples
    dct_coefficient_count = 1
    window_stride_samples = 1
    fingerprint_size = desired_samples
  else:
    dct_coefficient_count = args.dct_coefficient_count
    window_size_samples = int(args.sample_rate * args.window_size_ms / 1000)
    window_stride_samples = int(args.sample_rate * args.window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
      spectrogram_length = 0
    else:
      spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
      fingerprint_size = args.dct_coefficient_count * spectrogram_length
  
  return {
    'desired_samples': desired_samples,
    'window_size_samples': window_size_samples,
    'window_stride_samples': window_stride_samples,
    'feature_type': args.feature_type, 
    'spectrogram_length': spectrogram_length,
    'dct_coefficient_count': dct_coefficient_count,
    'fingerprint_size': fingerprint_size,
    'label_count': label_count,
    'sample_rate': args.sample_rate,
    'background_frequency': 0.8, # args.background_frequency
    'background_volume_range_': args.background_volume,
    'foreground_volume_min': args.foreground_volume_min,
    'foreground_volume_max': args.foreground_volume_max,
  }

def conv_block(inputs,
                 repeat,
                 kernel_size,
                 filters,
                 dilation,
                 stride,
                 filter_separable,
                 residual=False,
                 padding='same',
                 dropout=0.0,
                 activation='relu',
                 scale=True):
  """Convolutional block.  Can be optionally residual

  It is based on paper
  Jasper: An End-to-End Convolutional Neural Acoustic Model
  https://arxiv.org/pdf/1904.03288.pdf

  Args:
    inputs: input tensor
    repeat: number of repeating DepthwiseConv1D and Conv1D block
    kernel_size: kernel size of DepthwiseConv1D in time dim
    filters: number of filters in DepthwiseConv1D and Conv1D
    dilation: dilation in time dim for DepthwiseConv1D
    stride: stride in time dim for DepthwiseConv1D
    filter_separable: use separable conv or standard conv
    residual: if True residual connection is added
    padding: can be 'same' or 'causal'
    dropout: dropout value
    activation: type of activation function (string)
    scale: apply scaling in batchnormalization layer

  Returns:
    output tensor

  Raises:
    ValueError: if padding has invalid value
  """
  if residual and (padding not in ('same', 'causal')):
    raise ValueError('padding should be same or causal for residual blocks')

  net = inputs
  
  if kernel_size > 0:
    # DepthwiseConv1D
    if padding=='causal':
      print("Adding padding before DWConv2D")
      net = tf.pad(net, [[0, 0], [kernel_size-1, 0], [0, 0], [0, 0]], 'constant')
      dw_pad = 'valid'
    elif padding == 'valid':
      dw_pad = 'valid'
    elif padding == 'same':
      dw_pad = 'same'
        
    net = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(kernel_size, 1),
        strides=(stride, stride),
        padding=dw_pad,
        dilation_rate=(dilation, 1),
        use_bias=False)(
      net)

  # Conv2D 1x1 - streamable by default
  net = keras.layers.Conv2D(
      filters=filters, kernel_size=1, use_bias=False, padding='valid')(
          net)
  
  net = keras.layers.BatchNormalization(scale=scale)(net)

  if residual:
    ## have not tested this lately
    # Conv1D 1x1 - streamable by default
    net_res = keras.layers.Conv2D(
        filters=filters, kernel_size=1, use_bias=False, padding='valid')(
            inputs)
    net_res = keras.layers.BatchNormalization(scale=scale)(net_res)
    print(f"net.shape = {net.shape}, net_res.shape = {net_res.shape}")
    net = keras.layers.Add()([net, net_res])

  net = keras.layers.Activation(activation)(net)
  net = keras.layers.Dropout(rate=dropout)(net)
  return net

def select_optimizer(Flags, learning_rate):
  """
  Flags = Parsed command line arguments.
  Chooses an optimizer, (potentially) using the command line flags and platform architecture.
  Currently just returns Adam, but uses the legacy Adam on Apple Silicon, because standard Keras adam 
      is reported to run slow on Apple processors.
  """
  if platform.processor() == 'arm':
    print(f"Apple Silicon platform detected. Using legacy adam as standard Keras Adam is slow on this processor.")
    optimizer = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
  else:
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
  return optimizer
  
def get_model(args, use_qat=False):
  model_name = args.model_architecture

  label_count=3
  model_settings = prepare_model_settings(label_count, args)

  # For testing on long waveforms (measuring false alarms, hit rate), we 
  # can make the model accept a variable length input.  
  # If flags does not have the option, default to False
  variable_length = ('variable_length' in args and args.variable_length)

  if args.model_config_path:
    with open(args.model_config_path, 'r') as fpi:
        model_config = json.load(fpi)
    print(f"Read model config from {args.model_config_path}")  
  else:
    model_config = None

  
  if model_name=="ds_tcn":
    print("DS TCN model for streaming invoked")
    if model_config is not None:
      print(f"Using model config from {args.model_config_path}")
      ds_filters          = model_config['ds_filters']
      ds_repeat           = model_config['ds_repeat']
      ds_residual         = model_config['ds_residual']
      ds_kernel_size      = model_config['ds_kernel_size']
      ds_stride           = model_config['ds_stride']
      ds_dilation         = model_config['ds_dilation']
      ds_padding          = model_config['ds_padding']
      ds_filter_separable = model_config['ds_filter_separable']
      ds_scale            = model_config['ds_scale']
      ds_max_pool         = model_config['ds_max_pool']
      dropout             = model_config['dropout']
      activation          = model_config['activation']
    else:
      print(f"Using default model config (hard-coded in keras_model.py)")
      ds_filters          = [128, 128, 128, 32]
      ds_repeat           = [1, 1, 1, 1]
      ds_residual         = [0, 0, 0, 0]
      ds_kernel_size      = [3, 5, 10, 15]
      ds_stride           = [1, 1, 1, 1]
      ds_dilation         = [1, 1, 1, 1]
      ds_padding          = ['valid', 'valid', 'valid', 'valid']
      ds_filter_separable = [1, 1, 1, 1]
      ds_scale            = 1
      ds_max_pool         = 0
      dropout = 0.2
      activation = "relu"
    # check that all the lists are the same length
    num_blocks = len(ds_filters)
    for param_list, param_name in [(ds_filters         , "ds_filters"),
                                   (ds_repeat          , "ds_repeat"),       
                                   (ds_residual        , "ds_residual"),
                                   (ds_kernel_size     , "ds_kernel_size"),
                                   (ds_stride          , "ds_stride"),
                                   (ds_dilation        , "ds_dilation"),
                                   (ds_padding         , "ds_padding"),
                                   (ds_filter_separable, "ds_filter_separable")
                                   ]:
      if len(param_list) != num_blocks:
        print(f"{param_name} = {param_list}")
        raise ValueError(f"All config lists must be the same length ({num_blocks}).  {param_name} has length {len(param_list)}")

    num_frames_training = model_settings['spectrogram_length'] # length in time
    if variable_length:
      input_len = None # let length in time be variable
    else:
      input_len = num_frames_training
      
    input_shape = [input_len, 
                   1,
                   model_settings['dct_coefficient_count'] # number of features
                   ]

    weight_decay = 1e-4
    regularizer = l2(weight_decay)

    # Model layers
    print(f"Input shape = {input_shape}")
    input_spec = Input(shape=input_shape)
    
    net = input_spec
    
    # make it [batch, time, 1, feature]
    # net = tf.keras.backend.expand_dims(net, axis=2)
    
    for count in range(len(ds_stride)):  
      net = conv_block(net, repeat=ds_repeat[count], 
                         kernel_size=ds_kernel_size[count],
                         filters=ds_filters[count],
                         dilation=ds_dilation[count],
                         stride=ds_stride[count],
                         filter_separable=ds_filter_separable[count], 
                         residual=ds_residual[count], 
                         padding=ds_padding[count], 
                         scale=ds_scale,
                         dropout=dropout,
                         activation=activation)
      
    # net = tf.keras.layers.GlobalAveragePooling2D()(net)
    # if input shape is variable, we have to fix the pool size, so we can't use Global Pooling, 
    # but this has to change if preprocessing changes

    ## added length to DSC filters so there is nothing left to average in the time dimension
    # pool_len_time = 5 # is there a good way to infer this from the shape when input length is variable (None)
    # net = tf.keras.layers.AveragePooling2D((pool_len_time,1), strides=(1,1))(net)

    # time axis should be reduced to 1 after avg pool, so we don't
    # want to flatten across time when we have variable length, since that 
    # is modeling multiple inferences in one run.
    if variable_length:
      # net = tf.squeeze(net, axis=2) # this is what we want, but squeeze does not work with QAT
      # keep the (unknown:-1) time duration (shape[0]) and channels (shape[-1]).  Remove the singleton feature dimension
      net = tf.keras.layers.Reshape((-1, net.shape[-1]))(net) 
    else:
      net = tf.keras.layers.Flatten()(net)
    # if len(net.shape) > 2 and net.shape[1] is not None: # more than (batch, units)
    #   net = tf.keras.layers.Flatten()(net)
    
    net = tf.keras.layers.Dense(label_count, activation="softmax")(net) 
    model =  tf.keras.Model(input_spec, net)
    
    ########################################
  
  elif model_name=="fc4":
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(model_settings['spectrogram_length'],
                                             model_settings['dct_coefficient_count'])),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(model_settings['label_count'], activation="softmax")
    ])

  else:
    raise ValueError("Model name {:} not supported".format(model_name))

  optimizer = select_optimizer(args, args.learning_rate)

  if use_qat:
    annotated_model = tfmot.quantization.keras.quantize_annotate_model(model)
    with tfmot.quantization.keras.quantize_scope():
        model = tfmot.quantization.keras.quantize_apply(
            annotated_model,
            scheme=DefaultNBitQuantizeScheme(
                disable_per_axis=False,
                num_bits_weight=8,
                num_bits_activation=8,
            ),
        )  
        
  model.compile(
    optimizer=optimizer,  
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.CategoricalAccuracy(),
            keras.metrics.Precision(class_id=0, name='precision'), # prec = true_pos / (true_pos + false_pos)
            keras.metrics.Recall(class_id=0, name='recall'),    # recall = true_pos / (true_pos + false_neg)
            ],
  )

  return model

def apply_qat(float_model, Flags, init_lr=None):
  annotated_model = tfmot.quantization.keras.quantize_annotate_model(float_model)
  with tfmot.quantization.keras.quantize_scope():
    qat_model = tfmot.quantization.keras.quantize_apply(
      annotated_model,
      scheme=DefaultNBitQuantizeScheme(
          disable_per_axis=False,
          num_bits_weight=8,
          num_bits_activation=8,
      ),
    )  
  if init_lr is None:
    init_lr = Flags.learning_rate
    
  optimizer = select_optimizer(Flags, init_lr)
    
  qat_model.compile(
    optimizer=optimizer,  # Optimizer
    # Loss function to minimize
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    # List of metrics to monitor
    metrics=[keras.metrics.CategoricalAccuracy(),
             keras.metrics.Precision(class_id=0, name='precision'), # prec = true_pos / (true_pos + false_pos)
             keras.metrics.Recall(class_id=0, name='recall'),    # recall = true_pos / (true_pos + false_neg)
            ],
  )

  return qat_model    
