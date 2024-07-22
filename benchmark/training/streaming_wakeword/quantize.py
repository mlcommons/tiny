import tensorflow as tf

from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.platform import gfile

import tensorflow_model_optimization as tfmot

from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_quantize_configs import (
    NoOpQuantizeConfig,
)
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_quantize_registry import (
    DefaultNBitConvQuantizeConfig,
    DefaultNBitQuantizeConfig,
)
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_quantize_scheme import (
    DefaultNBitQuantizeScheme,
)

import numpy as np

import sys, os, pickle
import shutil
import json
from scipy.io import  wavfile
from IPython import display

import str_ww_util as util
import get_dataset
import keras_model as models

Flags = util.parse_command()
pretrained_model_path = Flags.saved_model_path
tfl_file_name = Flags.tfl_file_name

cal_set = np.load("calibration_samples.npz")
cal_specs = cal_set["specs"]
cal_labels = cal_set["labels"]

with tfmot.quantization.keras.quantize_scope(): # needed for the QAT wrappers
  model = keras.models.load_model(pretrained_model_path)
    
converter = tf.lite.TFLiteConverter.from_keras_model(model)

if True: 
  # If we omit this block, we'll get a floating-point TFLite model,
  # with this block, the weights and activations should be quantized to 8b integers, 
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  def representative_dataset_gen():
    for next_spec, label in zip(cal_specs, cal_labels):
      print(f"label = {label} ")
      next_spec = np.expand_dims(next_spec, 0).astype(np.float32)
      yield [next_spec]
    
  converter.representative_dataset = representative_dataset_gen
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # use this one
  # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

  converter.inference_input_type = tf.int8  # or tf.uint8; should match dat_q in eval_quantized_model.py
  converter.inference_output_type = tf.int8  # or tf.uint8

tflite_quant_model = converter.convert()

with open(tfl_file_name, "wb") as fpo:
  fpo.write(tflite_quant_model)
print(f"Wrote to {tfl_file_name}")
