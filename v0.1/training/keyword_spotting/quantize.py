import tensorflow as tf
import os
import numpy as np
import argparse

import get_dataset as aww_data
import aww_util

if __name__ == '__main__':
  Flags, unparsed = aww_util.parse_command()

  num_calibration_steps = 100
  converter = tf.lite.TFLiteConverter.from_saved_model(Flags.saved_model_path)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  
  _, _, ds_val = aww_data.get_training_data(Flags)
  ds_val = ds_val.unbatch().batch(1) 

  def representative_dataset_gen():
    for _ in range(num_calibration_steps):
      next_input = next(ds_val.as_numpy_iterator())[0]
      yield [next_input]
    
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset_gen
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8  # or tf.uint8; should match dat_q in eval_quantized_model.py
  converter.inference_output_type = tf.int8  # or tf.uint8
  tflite_quant_model = converter.convert()

  with open(Flags.tfl_file_name, "wb") as fpo:
    fpo.write(tflite_quant_model)

