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

import pandas as pd
import os


def construct_path(row):
    # Remove '.bin' and get the raw ID
    image_id = row['id'].replace('.bin', '')
    
    # Construct the base filename
    # We use 'val' here as per your example, but you can adjust logic if train is needed
    filename = f"COCO_val2014_{image_id}.jpg"
    
    # Determine subdirectory and class string based on the label (third field)
    if int(row['label']) == 1:
        return os.path.join('person', filename), 'person'
    else:
        return os.path.join('non_person', filename), 'non_person'



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
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_generator = datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation',
        color_mode='rgb')

####

    df_labels = pd.read_csv('y_labels.csv', header=None, names=['id', 'count', 'label'])

    # build the jpb filename from the binfile filename from y_labels.
    df_labels[['rel_path', 'class_name']] = df_labels.apply(
        lambda row: pd.Series(construct_path(row)), axis=1
    )

    # Filter the dataframe to only include files that actually exist to avoid Keras errors
    root_dir = 'vw_coco2014_96'
    df_labels['exists'] = df_labels['rel_path'].apply(
        lambda x: os.path.exists(os.path.join(root_dir, x))
    )
    df_final = df_labels[df_labels['exists'] == True].copy()

    print(f"Loaded {len(df_final)} valid image paths.")

    test_generator = datagen.flow_from_dataframe(
        dataframe=df_final,
        directory=root_dir,      
        x_col="rel_path",        
        y_col="class_name",      
        target_size=(96, 96),
        batch_size=32,
        class_mode="categorical", 
        shuffle=False             
    )
    
    model.evaluate(test_generator)
    
    return 0

if __name__ == '__main__':
  app.run(main)
