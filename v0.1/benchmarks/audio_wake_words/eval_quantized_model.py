import tensorflow as tf
import os
import numpy as np
import argparse

import get_dataset as aww_data
import aww_util


def predict(interpreter, data):

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  
  # Test the model on input data.
  input_shape = input_details[0]['shape']

  input_data = np.array(data, dtype=np.int8)
  output_data = np.empty_like(data)
  
  interpreter.set_tensor(input_details[0]['index'], input_data[i:i+1, :])
  interpreter.invoke()
  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  output_data[i:i+1, :] = interpreter.get_tensor(output_details[0]['index'])
  return output_data
  

if __name__ == '__main__':
  Flags, unparsed = aww_util.parse_command()

  num_samps = 2000
  scale = 0.66

  ds_train, ds_test, ds_val = aww_data.get_training_data(Flags)
  
  interpreter = tf.lite.Interpreter(model_path=Flags.tfl_file_name)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  input_shape = input_details[0]['shape']

  output_data = []
  labels = []
  
  test_data = ds_test.unbatch().batch(1).take(num_samps).as_numpy_iterator()
  for dat, label in test_data:
    dat_q = np.array((dat/scale)*256-128, dtype=np.int8)
    # dat_q = np.expand_dims(dat_q, 3)
    
    interpreter.set_tensor(input_details[0]['index'], dat_q)
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))
    labels.append(label[0])

  num_correct = np.sum(np.array(labels) == output_data)
  acc = num_correct / num_samps
  print(f"Accuracy = {acc:5.3f} ({num_correct}/{num_samps})")
