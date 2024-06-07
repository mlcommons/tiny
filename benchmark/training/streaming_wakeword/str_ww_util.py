import os
import argparse
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow import keras

def parse_command():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('HOME'), 'data', 'speech_commands_v0.02'),
      help="""\
      Where to download the speech training data to. Or where it is already saved.
      """)
  parser.add_argument(
      '--bg_path',
      type=str,
      default=os.path.join(os.getenv('HOME'), 'data', 'speech_commands_v0.02'),
      help="""\
      Where to find background noise folder.
      """)
  parser.add_argument(
      '--use_qat',
      dest='use_qat',
      action='store_true',
      help="""\
      Enable quantization-aware training
      """)
  parser.add_argument(
      '--no_use_qat',
      dest='use_qat',
      action='store_false',
      help="""\
      no_use_qat will disable quantization-aware training
      """)
  parser.set_defaults(use_qat=True)
  parser.add_argument(
      '--reps_of_target_training',
      type=int,
      default=12,
      help="""\
      Number of repetitions of the target wakeword added to the training set (before noise is added).
      """)
  parser.add_argument(
      '--reps_of_target_validation',
      type=int,
      default=2,
      help="""\
      Number of repetitions of the target wakeword added to the validation set (before noise is added).
      """)    
  parser.add_argument(
      '--num_silent_training',
      type=int,
      default=10000,
      help="""\
      Number of silent frames added to the training set (before noise is added).
      """)
  parser.add_argument(
      '--num_silent_validation',
      type=int,
      default=1200,
      help="""\
      Number of silent frames added to the validation set (before noise is added).
      """)
  parser.add_argument(
      '--foreground_volume_min',
      type=float,
      default=0.05,
      help="""\
      Minimum level for how loud the foreground words should be, between 0 and 1. Word volume will vary 
      randomly, uniformly  between foreground_volume_min and foreground_volume_max.
      """)
  parser.add_argument(
      '--foreground_volume_max',
      type=float,
      default=1.0,
      help="""\
      Maximum level for how loud the foreground words should be, between 0 and 1. Word volume will vary 
      randomly, uniformly  between foreground_volume_min and foreground_volume_max.
      """)  
  parser.add_argument(
      '--background_volume',
      type=float,
      default=2.0,
      help="""\
      How loud the background noise should be, between 0 and 1.  Noise volume will vary 
      randomly between zero and background_volume.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
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
      default=64.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=32.0,
      help='How much time between each spectrogram timeslice.',)
  parser.add_argument(
      '--feature_type',
      type=str,
      default="lfbe",
      choices=["mfcc", "lfbe", "td_samples"],
      help='Type of input features. Valid values: "mfcc" (default), "lfbe", "td_samples"',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many MFCC or log filterbank energy features')
  parser.add_argument(
      '--epochs',
      type=int,
      default=36,
      help="""\
      How many (total) epochs to train. If use_qat is enabled, and pretrain_epochs>0
      then the model will pretrain (without QAT) for pretrain_epochs, then train 
      with QAT for epochs-pretrain_epochs.
      """)
  parser.add_argument(
      '--pretrain_epochs',
      type=int,
      default=20,
      help="""\
      How many (total) epochs to train. If use_qat is enabled, and pretrain_epochs>0
      then the model will pretrain (without QAT) for pretrain_epochs, then fine-tune 
      with QAT for epochs-pretrain_epochs.  If pretrain_epochs > epochs, then there
      will be no QAT fine-tuning.
      """)
  parser.add_argument(
      '--num_samples_training',
      type=int,
      default=-1, # 85511,
    help='How many samples from the training set to use',)
  parser.add_argument(
      '--num_samples_validation',
      type=int,
      default=-1, # 10102,
    help='How many samples from the validation set to use',)
  parser.add_argument(
      '--num_samples_test',
      type=int,
      default=-1, # 4890,
    help='How many samples from the test set to use',)
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--num_bin_files',
      type=int,
      default=1000,
      help='How many binary test files for benchmark runner to create',)
  parser.add_argument(
      '--bin_file_path',
      type=str,
      default=os.path.join(os.getenv('HOME'), 'kws_test_files'),
      help="""\
      Directory where plots of binary test files for benchmark runner are written.
      """)
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='ds_tcn',
      help='What model architecture to use')
  parser.add_argument(
      '--run_test_set',
      dest='run_test_set',
      action="store_true",
      help='In train.py, run model.eval() on test set if True')
  parser.add_argument(
      '--no_run_test_set',
      dest='run_test_set',
      action="store_false",
      help='In train.py, do not run model.eval() on test set')
  parser.set_defaults(run_test_set=True)
  parser.add_argument(
      '--saved_model_path',
      type=str,
      default='trained_models/str_ww_model.h5',
      help='In quantize.py, path to load pretrained model from; in train.py, destination for trained model')
  parser.add_argument(
      '--model_init_path',
      type=str,
      default=None,
      help='Path to load pretrained model for evaluation or starting point for training')
  parser.add_argument(
      '--model_config_path',
      type=str,
      default=None,
      help='Path to json file defining a model dictionary.  If None, standard config is used.')  
  parser.add_argument(
      '--tfl_file_name',
      default='trained_models/kws_model.tflite',
      help='File name to which the TF Lite model will be saved (quantize.py) or loaded (eval_quantized_model)')
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
      help='Initial LR',)
  parser.add_argument(
      '--lr_sched_name',
      type=str,
      default='reduce_on_plateau',
      help="""\
      lr schedule scheme name to be picked from lr.py.  Currently support either 
      "reduce_on_plateau" or "step_function"
      """
      ) 
  parser.add_argument(
      '--plot_dir',
      type=str,
      default='./plots',
      help="""\
      Directory where plots of accuracy vs Epochs are stored
      """)
  parser.add_argument(
      '--target_set',
      type=str,
      default='test',
      help="""\
      For eval_quantized_model, which set to measure.
      """)

  Flags = parser.parse_args()

  if Flags.foreground_volume_min > Flags.foreground_volume_max:
    raise ValueError(f"foreground_volume_min ({Flags.foreground_volume_min}) must be no",
                     f"larger than foreground_volume_max ({Flags.foreground_volume_max})")

  return Flags


def plot_training(plot_dir,history, suffix=''):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Val Accuracy')
    if 'precision' in history.history:
      plt.plot(history.history['precision'], label='Training Precision')
    if 'recall' in history.history:
      plt.plot(history.history['recall'], label='Training Recall')
    if 'precision' in history.history:
      plt.plot(history.history['val_precision'], label='Val Precision')
    if 'recall' in history.history:
      plt.plot(history.history['val_recall'], label='Val Recall')
    plt.title('Metrics vs Epoch')
    plt.xlabel('Epoch')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.savefig(f"{plot_dir}/acc{suffix}.png")

def step_function_wrapper(batch_size):
    def step_function(epoch, lr):
        if (epoch < 12):
            return 0.0005
        elif (epoch < 24):
            return 0.0001
        elif (epoch < 36):
            return 0.00002
        else:
            return 0.00001
    return step_function

def get_callbacks(args):
    lr_sched_name = args.lr_sched_name
    batch_size = args.batch_size
    initial_lr = args.learning_rate
    callbacks = None
    if(lr_sched_name == "step_function"):
        callbacks = [keras.callbacks.LearningRateScheduler(step_function_wrapper(batch_size),verbose=1)]
    elif lr_sched_name == "reduce_on_plateau":
        callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                       patience=3, min_lr=1e-6)]
    return callbacks


    
