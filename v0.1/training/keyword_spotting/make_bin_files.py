import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

import get_dataset as kws_data
import kws_util
import keras_model as models

if __name__ == '__main__':
  Flags, unparsed = kws_util.parse_command()

  num_test_files = Flags.num_bin_files
  test_file_path = Flags.bin_file_path
  
  print(f"Extracting {num_test_files} to {test_file_path}")
  word_labels = ["Down", "Go", "Left", "No", "Off", "On", "Right",
                 "Stop", "Up", "Yes", "Silence", "Unknown"]

  num_labels = len(word_labels)
  ds_train, ds_test, ds_val = kws_data.get_training_data(Flags, val_cal_subset=True)
  
  if Flags.target_set[0:3].lower() == 'val':
    eval_data = ds_val
    print("Evaluating on the validation set")
  elif Flags.target_set[0:4].lower() == 'test':
    eval_data = ds_test
    print("Evaluating on the test set")
  elif Flags.target_set[0:5].lower() == 'train':
    eval_data = ds_train    
    print("Evaluating on the training set")

  model_settings = models.prepare_model_settings(num_labels, Flags)

  if Flags.feature_type == "mfcc":
    output_type = np.int8
    quant_min, quant_max = -128, 127
    # we should really do both of these in the way that the LFBE is doing
    # since the MFCC style depends on a specific TFL model, but since
    # now (4/24/21) bin files for mfcc features are already published,
    # we'll wait until v0.2 to unify the bin file quantization calibration
    interpreter = tf.lite.Interpreter(model_path=Flags.tfl_file_name)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_shape[0] = 0
    input_scale, input_zero_point = input_details[0]["quantization"]

  elif Flags.feature_type == "lfbe":
    output_type = np.uint8
    quant_min, quant_max = 0, 255
    # iterate over all the input tensors in the quant calibration subset of ds_val
    # find the range and mid-point, and calculate scale and zero_point
    input_shape = (0, model_settings['spectrogram_length'],
                   model_settings['dct_coefficient_count'], 1)
    all_dat   = np.zeros(input_shape, dtype='float32')

    # collect all the calibration data
    for dat, label in ds_val.as_numpy_iterator(): 
      all_dat = np.concatenate((all_dat, dat))

    input_scale = 1.0*(np.max(all_dat) - np.min(all_dat)) / 255.0
    input_zero_point = (np.max(all_dat)+np.min(all_dat))/(2*input_scale)

    orig_min = np.min(all_dat)
    orig_max = np.max(all_dat)    
    input_scale = 1.0 / 255.0  # LFBEs are clipped to [0.0,1.0] in feature extraction
    input_zero_point = 0.0
    # quantizing as:  quantized_value = int(original_value/input_scale + input_zero_point)
    print(f"Calibration data ranged from {orig_min} to {orig_max}.")

  elif Flags.feature_type == "td_samples":
    input_shape = (0, 16000, 1, 1)
    output_type = np.int16
    quant_min, quant_max = -2.0**15, 2.0**15-1
    input_scale = 1.0/2**15
    input_zero_point = 0.0
    
  print(f"Scale factor = {input_scale}, zero point = {input_zero_point}")
  output_data = []
  labels = []  
  file_names = []
  count = 0

  # set true to run the TFL model on each input before writing it to files.
  # This will also generate a file tflm_labels.csv (similar to y_labels.csv)
  # recording what the model predicted for each input
  test_tfl_on_bin_files = False

  all_dat   = np.zeros(input_shape, dtype='float32')
  all_dat_q   = np.zeros(input_shape, dtype=output_type)
  
  # make the target directory and all directories above it if it doesn't exist
  os.makedirs(test_file_path, exist_ok = True) 

  eval_data = eval_data.unbatch().batch(1).take(num_test_files).as_numpy_iterator()
  for dat, label in eval_data:
    dat_q = np.array(dat/input_scale + input_zero_point, dtype=output_type) # should match input type in quantize.py

    label_str = word_labels[label[0]]
    fname = f"tst_{count:06d}_{label_str}_{label[0]}.bin"
    with open(os.path.join(test_file_path, fname), "wb") as fpo:
      fpo.write(dat_q.flatten())

    if test_tfl_on_bin_files:
      interpreter.set_tensor(input_details[0]['index'], dat_q)
      interpreter.invoke()
      #  X.get_tensor() returns a copy of the tensor data; X.tensor() => pointer to the tensor
      output_data.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))

    labels.append(label[0])
    file_names.append(fname)
    all_dat   = np.concatenate((all_dat,   dat  ))
    all_dat_q = np.concatenate((all_dat_q, dat_q))
    count += 1

  print(f"FP32      feature data ranges from {np.min(all_dat)  } to {np.max(all_dat)  } with mean = {np.mean(all_dat)}")
  print(f"Quantized feature data ranges from {np.min(all_dat_q)} to {np.max(all_dat_q)} with mean = {np.mean(all_dat_q)}")
    
  with open(os.path.join(test_file_path, "y_labels.csv"), "w") as fpo_true_labels:
    for (fname, lbl) in zip(file_names, labels):
      fpo_true_labels.write(f"{fname}, {num_labels}, {lbl}\n")

  if test_tfl_on_bin_files:
    num_correct = np.sum(np.array(labels) == output_data)
    acc = num_correct / len(labels)
    print(f"Accuracy = {acc:5.3f} ({num_correct}/{len(labels)})")
    with open(os.path.join(test_file_path, "tflm_labels.csv"), "w") as fpo_tflm_labels:
      for (fname, out) in zip(file_names, output_data):
        fpo_tflm_labels.write(f"{fname}, {num_labels}, {out}\n")

  fig, axes = plt.subplots(3,1, figsize=(6, 8))
  axes[0].plot(all_dat.flatten(), all_dat_q.flatten(), 'r.')
  axes[1].hist(all_dat.flatten(),bins=20);
  axes[2].hist(all_dat_q.flatten(),bins=20);
  plt.savefig(f"test_data_quantization_{Flags.feature_type}.png")

