# Lint as: python3
"""Training for the visual wakewords person detection model.

The visual wakewords person detection model is a core model for the TinyMLPerf
benchmark suite. This script provides source for how the reference model was
created and trained, and can be used as a starting point for open submissions
using re-training.
"""

import os

from absl import app

import tensorflow as tf
assert tf.__version__.startswith('2')

IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20

BASE_DIR = os.path.join(os.getcwd(), "dataset")

def main(argv):
    if len(argv) >= 2:
        model = tf.keras.models.load_model(argv[1])
    else:
        core = tf.keras.applications.MobileNet(
            input_shape=(96, 96, 1),
            alpha=0.25,
            include_top=True,
            weights=None,
            classes=2,
            classifier_activation='softmax',
            dropout=0.5)
        core.trainable = True

        model = tf.keras.Sequential([
            core
        ])

        # Add weight regularization to fight overfitting.
        regularizer=tf.keras.regularizers.l1_l2(0.01)
        for layer in model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        # When we change the layers attributes, the change only happens in the
        # model config file
        model_json = model.to_json()

        # load the model from the config
        model = tf.keras.models.model_from_json(model_json)

    model.summary()

    # Extract labeled data from dataset directory.
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='training',
        color_mode='grayscale')
    val_generator = datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation',
        color_mode='grayscale')

    print(train_generator.class_indices)

    # Train and save model.
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history_fine = model.fit(train_generator,
                             steps_per_epoch=len(train_generator),
                             epochs=EPOCHS,
                             validation_data=val_generator,
                             validation_steps=len(val_generator),
                             batch_size=BATCH_SIZE)

    # Save model HDF5
    if len(argv) >= 3:
        model.save(argv[2])
    else:
        model.save('vww_96.h5')


if __name__ == '__main__':
    app.run(main)
