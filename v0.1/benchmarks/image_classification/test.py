import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import keras
from sklearn.metrics import roc_auc_score

import train

if __name__ == "__main__":
    """show it works"""

    cifar_10_dir = 'cifar-10-batches-py'

    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        train.load_cifar_10_data(cifar_10_dir)

    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)

    model = tf.keras.models.load_model('trained_models/trainedResnet_20201214_1120.h5')

    test_metrics = model.evaluate(x=test_data, y=test_labels, batch_size=32, verbose=1, return_dict=True)

    print("Trained model evaluate on test data with keras evaluate method")
    print("Test loss: ", test_metrics['loss'])
    print("Test acc: ", test_metrics['accuracy'])

    predictions = []
    for test_sample in test_data:
        prediction = model.predict(np.expand_dims(test_sample, axis=0))
        predictions.append(prediction.reshape(10,))
    predictions = np.array(predictions)

    print(predictions.shape)

    auc = roc_auc_score(test_labels, predictions)
    print("Area under curve: ", auc)
