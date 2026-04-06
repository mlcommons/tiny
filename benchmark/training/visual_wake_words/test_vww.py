# Lint as: python3
"""Training for the visual wakewords person detection model.

The visual wakewords person detection model is a core model for the TinyMLPerf
benchmark suite. This script provides source for how the reference model was
created and trained, and can be used as a starting point for open submissions
using re-training.
"""

import os
import datetime

from absl import app
# from train_vww_multiple import WarmUpCosine

import tensorflow as tf
assert tf.__version__.startswith('2')

import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 96
BATCH_SIZE = 32

BASE_DIR = os.path.join(os.getcwd(), 'vw_coco2014_96')

TRAIN_RUNS = 2

def main(argv):
    print(f"KERAS MODEL TEST")
    if len(argv) >= 2:
       model_file = argv[-1]
    else:
       print('test_vww: no model specified, using default model')
       model_file = 'trained_models/vww_96.h5'

    print(f"Loading model file {model_file}...")
    model = tf.keras.models.load_model(model_file, compile=False)
    model.compile(
       optimizer="adam",
       loss="categorical_crossentropy",
       metrics=["accuracy"]
       )


    print("Model loaded, generating test data...")
    batch_size = 50
    validation_split = 0.1
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=.1,
        horizontal_flip=True,
        validation_split=validation_split,
        rescale=1. / 255)
    test_generator = datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation',
        color_mode='rgb')
    model.evaluate(test_generator)
    
    return 0

if __name__ == '__main__':
  app.run(main)
