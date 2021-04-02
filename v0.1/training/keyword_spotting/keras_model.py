#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, AveragePooling2D
from tensorflow.keras.regularizers import l2

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



def get_model(args):
  model_name = args.model_architecture

  label_count=12
  model_settings = prepare_model_settings(label_count, args.sample_rate, args.clip_duration_ms,
                               args.window_size_ms, args.window_stride_ms,
                               args.dct_coefficient_count,args.background_frequency)

  if model_name=="fc4":
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(model_settings['spectrogram_length'], model_settings['dct_coefficient_count'])),
        tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(model_settings['label_count'], activation="softmax")
    ])

  elif model_name == 'ds_cnn':
    print("DS CNN model invoked")
    input_shape = [model_settings['spectrogram_length'], model_settings['dct_coefficient_count'],1]
    filters = 64
    weight_decay = 1e-4
    regularizer = l2(weight_decay)

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

    x = AveragePooling2D(pool_size=(25,5))(x)

    x = Flatten()(x)
    outputs = Dense(model_settings['label_count'], activation='softmax')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)




  else:
    raise ValueError("Model name {:} not supported".format(model_name))

  model.compile(
    #optimizer=keras.optimizers.RMSprop(learning_rate=args.learning_rate),  # Optimizer
    optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )

  return model
