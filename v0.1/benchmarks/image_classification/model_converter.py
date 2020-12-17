'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

model_converter.py desc: converts floating point model to fully int8
'''

import tensorflow as tf
import keras
import numpy as np
import train

tfmodel_path = 'trained_models/trainedResnet.h5'
tfmodel = keras.models.load_model(tfmodel_path)
cifar_10_dir = 'cifar-10-batches-py'

def representative_dataset_generator():
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        train.load_cifar_10_data(cifar_10_dir)
    _idx = np.array(range(10000))
    np.random.shuffle(_idx)
    for i in _idx[:500]:
        print(i)
        sample_img = np.expand_dims(np.array(test_data[i], dtype=np.float32), axis=0)
        yield [sample_img]

if __name__ == '__main__':
    converter = tf.lite.TFLiteConverter.from_keras_model(tfmodel)
    tflite_model = converter.convert()
    open('trained_models/gest_rec_model.tflite', 'wb').write(tflite_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset_generator
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    open('trained_models/gest_rec_quantized_model.tflite', 'wb').write(tflite_quant_model)
