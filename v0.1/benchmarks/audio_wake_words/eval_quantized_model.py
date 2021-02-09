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

  ds_train, ds_test, ds_val = aww_data.get_training_data(Flags)
  
  interpreter = tf.lite.Interpreter(model_path=Flags.tfl_file_name)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  input_shape = input_details[0]['shape']

  output_data = []
  labels = []
  
  if Flags.target_set[0:3].lower() == 'val':
    eval_data = ds_val
    print("Evaluating on the validation set")
  elif Flags.target_set[0:4].lower() == 'test':
    eval_data = ds_test
    print("Evaluating on the test set")
  elif Flags.target_set[0:5].lower() == 'train':
    eval_data = ds_train    
    print("Evaluating on the training set")
    
  eval_data = eval_data.unbatch().batch(1).as_numpy_iterator()
  input_scale, input_zero_point = input_details[0]["quantization"]
  
  for dat, label in eval_data:
    dat_q = np.array(dat/input_scale + input_zero_point, dtype=np.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], dat_q)
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))
    labels.append(label[0])

  num_correct = np.sum(np.array(labels) == output_data)
  acc = num_correct / len(labels)
  print(f"Accuracy = {acc:5.3f} ({num_correct}/{len(labels)})")
