import tensorflow as tf
import os
import numpy as np
import argparse

import get_dataset as kws_data
import kws_util

if __name__ == '__main__':
  Flags, unparsed = kws_util.parse_command()

  num_test_files = 50
  test_file_path = os.path.join(os.getenv('HOME'), 'kws_test_files')
  word_labels = ["Down", "Go", "Left", "No", "Off", "On", "Right",
                 "Stop", "Up", "Yes", "Silence", "Unknown"]

  num_labels = len(word_labels)
  ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
  
  interpreter = tf.lite.Interpreter(model_path=Flags.tfl_file_name)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  input_shape = input_details[0]['shape']

  if Flags.target_set[0:3].lower() == 'val':
    eval_data = ds_val
    print("Evaluating on the validation set")
  elif Flags.target_set[0:4].lower() == 'test':
    eval_data = ds_test
    print("Evaluating on the test set")
  elif Flags.target_set[0:5].lower() == 'train':
    eval_data = ds_train    
    print("Evaluating on the training set")
    
  eval_data = eval_data.unbatch().batch(1).take(num_test_files).as_numpy_iterator()
  input_scale, input_zero_point = input_details[0]["quantization"]

  output_data = []
  labels = []  
  file_names = []
  count = 0

  for dat, label in eval_data:
    dat_q = np.array(dat/input_scale + input_zero_point, dtype=np.int8) # should match input type in quantize.py
    label_str = word_labels[label[0]]
    fname = f"tst_{count:06d}_{label_str}_{label[0]}.bin"
    with open(os.path.join(test_file_path, fname), "wb") as fpo:
      fpo.write(dat_q.flatten())
    interpreter.set_tensor(input_details[0]['index'], dat_q)
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))
    labels.append(label[0])
    file_names.append(fname)
    count += 1
        
  num_correct = np.sum(np.array(labels) == output_data)
  acc = num_correct / len(labels)
  print(f"Accuracy = {acc:5.3f} ({num_correct}/{len(labels)})")

  with open(os.path.join(test_file_path, "y_labels.csv"), "w") as fpo_true_labels:
    for (fname, lbl) in zip(file_names, labels):
      fpo_true_labels.write(f"{fname}, {num_labels}, {lbl}\n")
  with open(os.path.join(test_file_path, "tflm_labels.csv"), "w") as fpo_tflm_labels:
    for (fname, out) in zip(file_names, output_data):
      fpo_tflm_labels.write(f"{fname}, {num_labels}, {out}\n")


  


