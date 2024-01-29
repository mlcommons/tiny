#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import platform

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
    'background_volume_range_': 0.1
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
                 scale=True,
                 use_one_step=True,
                 build_streaming=True):
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
    use_one_step: this parameter will be used for streaming only

  Returns:
    output tensor

  Raises:
    ValueError: if padding has invalid value
  """
  if residual and (padding not in ('same', 'causal')):
    raise ValueError('padding should be same or causal for residual blocks')

  net = inputs
  
  if kernel_size > 0:
    print("under kernel_size > 0")
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




def get_model(args):
  model_name = args.model_architecture

  label_count=3
  model_settings = prepare_model_settings(label_count, args)


  if model_name=="ds_tcn":
    print("DS CNN model invoked")

    ds_filters          = [128, 64, 64, 64, 128]
    ds_repeat           = [1, 1, 1, 1, 1]
    ds_residual         = [0, 0, 0, 0, 0]
    ds_kernel_size      = [5, 5, 11, 13, 15]
    ds_stride           = [1, 1, 1, 1, 1]
    ds_dilation         = [1, 1, 1, 1, 1]

    ds_filter_separable = [1, 1, 1, 1, 1]
    ds_scale            = 1
    ds_max_pool         = 0

    # check that all the lists are the same length
    num_blocks = len(ds_filters)
    for param_list, param_name in [(ds_filters         , "ds_filters"),
                                   (ds_repeat          , "ds_repeat"),       
                                   (ds_residual        , "ds_residual"),
                                   (ds_kernel_size     , "ds_kernel_size"),
                                   (ds_stride          , "ds_stride"),
                                   (ds_dilation        , "ds_dilation"),
                                   (ds_filter_separable, "ds_filter_separable")
                                   ]:
      if len(param_list) != num_blocks:
        raise ValueError(f"All config lists must be the same length ({num_blocks}).  param_name has length {len(param_list)}")
        

    input_shape = [model_settings['spectrogram_length'], # length in time
                   model_settings['dct_coefficient_count'] # number of features
                   ]
    filters = 64
    weight_decay = 1e-4
    regularizer = l2(weight_decay)

    
    # Model layers
    # Input pure conv2d
    inputs = Input(shape=input_shape)

    net = inputs
    
    # make it [batch, time, 1, feature]
    net = tf.keras.backend.expand_dims(net, axis=2)
    
    
    for count in range(len(ds_stride)):  
      print(f"count={count}, before resnet block, net shape = {net.shape}")
      net = conv_block(net, repeat=ds_repeat[count], 
                         kernel_size=ds_kernel_size[count],
                         filters=ds_filters[count],
                         dilation=ds_dilation[count],
                         stride=ds_stride[count],
                         filter_separable=ds_filter_separable[count], 
                         residual=ds_residual[count], 
                         padding=ds_padding[count], 
                         scale=flags.ds_scale,
                         dropout=flags.dropout,
                         activation=flags.activation, 
                         use_one_step=(flags.data_stride<=1), 
                         build_streaming=build_streaming)
      print(f"after resnet block, net shape = {net.shape}")
    net = tf.keras.layers.GlobalAveragePooling2D()(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(flags.label_count)(net) # target + _silence, _noise
    return tf.keras.Model(input_audio, net)
    
    ########################################

    
    x = Flatten()(x)
    outputs = Dense(model_settings['label_count'], activation='softmax')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)


    

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

  elif model_name == 'ds_cnn':
    print("DS CNN model invoked")
    input_shape = [model_settings['spectrogram_length'], model_settings['dct_coefficient_count'],1]
    filters = 64
    weight_decay = 1e-4
    regularizer = l2(weight_decay)
    final_pool_size = (int(input_shape[0]/2), int(input_shape[1]/2))
    
    # Model layers
    # Input pure conv2d
    inputs = Input(shape=input_shape)
    x = Conv2D(filters, (10,4), strides=(2,2), padding='same', kernel_regularizer=regularizer)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2)(x)

    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Third layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Fourth layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Reduce size and apply final softmax
    x = Dropout(rate=0.4)(x)

    x = AveragePooling2D(pool_size=final_pool_size)(x)
    x = Flatten()(x)
    outputs = Dense(model_settings['label_count'], activation='softmax')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)

  elif model_name == 'td_cnn':
    print("TD CNN model invoked")
    input_shape = [model_settings['spectrogram_length'], model_settings['dct_coefficient_count'],1]
    print(f"Input shape = {input_shape}")
    filters = 64
    weight_decay = 1e-4
    regularizer = l2(weight_decay)

    # Model layers
    # Input time-domain conv
    inputs = Input(shape=input_shape)
    x = Conv2D(filters, (512,1), strides=(384, 1),
               padding='valid', kernel_regularizer=regularizer)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Reshape((41,64,1))(x)
    
    # True conv 
    x = Conv2D(filters, (10,4), strides=(2,2), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2)(x)

    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Third layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Fourth layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Reduce size and apply final softmax
    x = Dropout(rate=0.4)(x)

    # x = AveragePooling2D(pool_size=(25,5))(x)
    x = GlobalAveragePooling2D()(x)

    x = Flatten()(x)
    outputs = Dense(model_settings['label_count'], activation='softmax')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)

  else:
    raise ValueError("Model name {:} not supported".format(model_name))

  if platform.processor() == 'arm':
    print(f"Apple Silicon platform detected. Using legacy adam as standard Keras Adam is slow on this processor.")
    optimizer = keras.optimizers.legacy.Adam(learning_rate=args.learning_rate)
  else:
    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
  
  model.compile(
    #optimizer=keras.optimizers.RMSprop(learning_rate=args.learning_rate),  # Optimizer
    optimizer=optimizer,  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )

  return model
