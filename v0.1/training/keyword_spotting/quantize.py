import tensorflow as tf
import os
import numpy as np
import argparse

import get_dataset as kws_data
import kws_util

if __name__ == '__main__':
  Flags, unparsed = kws_util.parse_command()

  num_calibration_steps = 100
  converter = tf.lite.TFLiteConverter.from_saved_model(Flags.saved_model_path)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  
  _, _, ds_val = kws_data.get_training_data(Flags)
  ds_val = ds_val.unbatch().batch(1) 
  print("finished loading and unbatching/rebatching dataset")

  with open("quant_cal_idxs.txt") as fpi:
    cal_indices = [int(line) for line in fpi]
  cal_indices.sort()

  
  def representative_dataset_gen():
    current_index = 0
    for idx in cal_indices:
      # we want to element idx out of ds_val.  is there a cleaner way? JH
      print(f"About to find element {idx}")
      while current_index <= idx:
        next_input = next(ds_val.as_numpy_iterator())[0]
        current_index += 1
      print(f"About to return element {idx}")
      yield [next_input]
    
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset_gen
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8  # or tf.uint8; should match dat_q in eval_quantized_model.py
  converter.inference_output_type = tf.int8  # or tf.uint8

  print("Setup complete.  About to convert model")
  tflite_quant_model = converter.convert()
  print("Just converted model")
  with open(Flags.tfl_file_name, "wb") as fpo:
    fpo.write(tflite_quant_model)

