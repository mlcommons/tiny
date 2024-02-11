#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os, argparse

from tensorflow import keras
import tensorflow

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

import keras_model as models
import get_dataset as str_ww_data
import str_ww_util as util


num_classes = 3 # should probably draw this directly from the dataset.

Flags, unparsed = util.parse_command()

print('We will download data to {:}'.format(Flags.data_dir))
print('We will train for {:} epochs'.format(Flags.epochs))

ds_train, ds_test, ds_val = str_ww_data.get_training_data(Flags)
print("Done getting data")

# this is taken from the dataset web page.
# there should be a better way than hard-coding this
train_shuffle_buffer_size = 85511
val_shuffle_buffer_size = 10102
test_shuffle_buffer_size = 4890

ds_train = ds_train.shuffle(train_shuffle_buffer_size)
ds_val = ds_val.shuffle(val_shuffle_buffer_size)
ds_test = ds_test.shuffle(test_shuffle_buffer_size)

if Flags.model_init_path is None:
  print("Starting with untrained model")
  model = models.get_model(args=Flags, use_qat=False)
else:
  print(f"Starting with pre-trained model from {Flags.model_init_path}")
  model = keras.models.load_model(Flags.model_init_path)

model.summary()

callbacks = util.get_callbacks(args=Flags)   

if Flags.use_qat:
  float_epochs = Flags.pretrain_epochs
  qat_epochs = Flags.epochs - Flags.pretrain_epochs
else:
  float_epochs = Flags.epochs
  qat_epochs = 0
    
if float_epochs > 0:
  train_hist = model.fit(ds_train, validation_data=ds_val, epochs=float_epochs, callbacks=callbacks)
  util.plot_training(Flags.plot_dir,train_hist)
  model.save(Flags.saved_model_path.split('.')[0] + '_float.h5')


# get the final learning rate after fine tuning so we can start back at the same LR
post_train_lr = model.optimizer._decayed_lr(np.float32)

if qat_epochs > 0:
  model_qat = models.apply_qat(model, Flags, init_lr=post_train_lr)
  train_hist_qat = model_qat.fit(ds_train, validation_data=ds_val, 
                                 epochs=qat_epochs, callbacks=callbacks)
  util.plot_training(Flags.plot_dir,train_hist_qat, suffix='_qat')
  model.save(Flags.saved_model_path)



if Flags.run_test_set:
  test_scores = model.evaluate(ds_test)
  print("Test loss:", test_scores[0])
  print("Test accuracy:", test_scores[1])
