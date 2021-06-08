'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

test.py: performances on cifar10 test set
target performances: https://github.com/SiliconLabs/platform_ml_models/tree/master/eembc/CIFAR10_ResNetv1
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import keras
from sklearn.metrics import roc_auc_score

import train
import eval_functions_eembc
import keras_model

model_name = keras_model.get_model_name()

if __name__ == "__main__":

    cifar_10_dir = 'cifar-10-batches-py'

    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        train.load_cifar_10_data(cifar_10_dir)

    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)
    label_classes = np.argmax(test_labels,axis=1)
    print("Label classes: ", label_classes.shape)

    model = tf.keras.models.load_model('trained_models/' + model_name + '.h5')

    test_metrics = model.evaluate(x=test_data, y=test_labels, batch_size=32, verbose=1, return_dict=True)

    print("Performances on cifar10 test set")
    print("Keras evaluate method")
    print("Accuracy keras: ", test_metrics['accuracy'])
    print("---------------------")

    predictions = []
    for test_sample in test_data:
        prediction = model.predict(np.expand_dims(test_sample, axis=0))
        predictions.append(prediction.reshape(10,))
    predictions = np.array(predictions)

    print("EEMBC calculate_accuracy method")
    accuracy_eembc = eval_functions_eembc.calculate_accuracy(predictions, label_classes)
    print("---------------------")

    auc_scikit = roc_auc_score(test_labels, predictions)
    print("sklearn.metrics.roc_auc_score method")
    print("AUC sklearn: ", auc_scikit)
    print("---------------------")

    print("EEMBC calculate_auc method")
    auc_eembc = eval_functions_eembc.calculate_auc(predictions, label_classes, label_names, model_name)
    print("---------------------")
