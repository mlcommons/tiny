import tensorflow as tf
import argparse
import aww_data
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    type=str,
    default=os.path.join(os.getenv('HOME'), 'data'),
    help="""\
      Where to download the speech training data to. Or where it is already saved.
    """)
parser.add_argument(
    '--preprocessed_data_dir',
    type=str,
    default=os.path.join(os.getenv('HOME'), 'data/speech_commands_preprocessed'),
    help="""\
    Where to store preprocessed speech data (spectrograms) or load it, if it exists
    with the same parameters as are used in the current run.
    """)
parser.add_argument(
    '--save_preprocessed_data',
    type=bool,
    default=True,
    help="""\
    Where to download the speech training data to. Or where it is already saved.
    """)
parser.add_argument(
    '--silence_percentage',
    type=float,
    default=10.0,
    help="""\
    How much of the training data should be silence.
    """)
parser.add_argument(
    '--unknown_percentage',
    type=float,
    default=10.0,
    help="""\
    How much of the training data should be unknown words.
    """)
parser.add_argument(
    '--time_shift_ms',
    type=float,
    default=100.0,
    help="""\
    Range to randomly shift the training audio by in time.
    """)
parser.add_argument(
    '--sample_rate',
    type=int,
    default=16000,
    help='Expected sample rate of the wavs',)
parser.add_argument(
    '--clip_duration_ms',
    type=int,
    default=1000,
    help='Expected duration in milliseconds of the wavs',)
parser.add_argument(
    '--window_size_ms',
    type=float,
    default=30.0,
    help='How long each spectrogram timeslice is',)
parser.add_argument(
    '--window_stride_ms',
    type=float,
    default=20.0,
    help='How long each spectrogram timeslice is',)
parser.add_argument(
    '--dct_coefficient_count',
    type=int,
    default=40,
    help='How many bins to use for the MFCC fingerprint',)
parser.add_argument(
    '--num_train_samples',
    type=int,
    default=85511,
    help='How many samples from the training set to use',)
parser.add_argument(
    '--num_val_samples',
    type=int,
    default=10102,
    help='How many samples from the validation set to use',)
parser.add_argument(
    '--num_test_samples',
    type=int,
    default=4890,
    help='How many samples from the test set to use',)
parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
parser.add_argument(
    '--model_architecture',
    type=str,
    default='ds_cnn',
    help='What model architecture to use')
parser.add_argument(
    '--saved_model_path',
    type=str,
    default='pretrained_model',
    help='File name to load pretrained model')
parser.add_argument(
    '--tfl_file_name',
    default='aww_model.tflite',
    help='File name to which the TF Lite model will be saved')

num_calibration_steps = 10
saved_model_dir = './pretrained_model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

Flags, unparsed = parser.parse_known_args()
_, _, ds_val = aww_data.get_training_data(Flags)
# ds_val = ds_val.batch(1) # can we use a larger batch?

def representative_dataset_gen():
  for _ in range(num_calibration_steps):
      next_input = np.expand_dims(next(ds_val.as_numpy_iterator())[0], 3)
      yield [next_input]
    

# tst_val = next(representative_dataset_gen)
# print('rep data shape = {:}'.format(tst_val.shape))

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()
with open(Flags.tfl_file_name, "wb") as fpo:
    fpo.write(tflite_quant_model)

