'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

tflite_test.py: converted models performances on cifar10 test set
'''

import tensorflow as tf
import numpy as np
import h5py
import os
import sys
import train
import eval_functions_eembc
from sklearn.metrics import roc_auc_score
import keras_model

np.set_printoptions(threshold=sys.maxsize)

# if True uses the official MLPerf Tiny subset of CIFAR10 for validation
# if False uses the full CIFAR10 validation set
PERF_SAMPLE = True
# if True uses quantized model
QUANT_MODEL = True

if QUANT_MODEL:
    _name = keras_model.get_quant_model_name()
    model_path = 'trained_models/' + _name + '_quant.tflite'
else:
    _name = keras_model.get_quant_model_name()
    model_path = 'trained_models/' + _name + '.tflite'

if __name__ == '__main__':
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cifar_10_dir = 'cifar-10-batches-py'

    train_data, train_filenames, train_labels, test_imgs, test_filenames, test_labels, label_names = \
        train.load_cifar_10_data(cifar_10_dir)

    if PERF_SAMPLE:
        _idxs = np.load('perf_samples_idxs.npy')
        test_imgs = test_imgs[_idxs]
        test_labels = test_labels[_idxs]
        test_filenames = test_filenames[_idxs]

    label_classes = np.argmax(test_labels, axis=1)
    print("Label classes: ", label_classes.shape)

    if QUANT_MODEL:
        test_imgs = test_imgs.astype(np.int64) - 128
        test_imgs = test_imgs.astype(np.int8)
    else:
        test_imgs = test_imgs.astype(np.float32)

    predictions = []
    for img in test_imgs:
        input_data = img.reshape(1, 32, 32, 3)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data.reshape(10,))
    predictions = np.array(predictions)

    print("EEMBC calculate_accuracy method")
    accuracy_eembc = eval_functions_eembc.calculate_accuracy(predictions, label_classes)
    print("---------------------")

    auc_scikit = roc_auc_score(test_labels, predictions)
    print("sklearn.metrics.roc_auc_score method")
    print("AUC sklearn: ", auc_scikit)
    print("---------------------")

    print("EEMBC calculate_auc method")
    auc_eembc = eval_functions_eembc.calculate_auc(predictions, label_classes, label_names, model_path)
    print("---------------------")
