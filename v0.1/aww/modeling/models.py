#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np


def get_model(model_name="fc4", num_classes=12):
  if model_name=="fc4":
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(49, 40)),
        tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),  
        tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
  else:
    raise ValueError("Model name {:} not supported".format(model_name))

  model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )

  return model
