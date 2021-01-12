import os
import argparse

def parse_command():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('HOME'), 'data'),
      help="""\
      Where to download the speech training data to. Or where it is already saved.
      """)
  parser.add_argument(
      '--bg_path',
      type=str,
      default=os.path.join(os.getenv('PWD')),
      help="""\
      Where to find background noise folder.
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
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
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
      default=10,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--epochs',
      type=int,
      default=36,
      help='How many epochs to train',)
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
      '--run_test_set',
      type=bool,
      default=True,
      help='Run model.eval() on test set if True')
  parser.add_argument(
      '--saved_model_path',
      type=str,
      default='trained_models/scratch',
      help='File name to load pretrained model')
  parser.add_argument(
      '--tfl_file_name',
      default='aww_model.tflite',
      help='File name to which the TF Lite model will be saved')
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.00001,
      help='Initial LR',)
  parser.add_argument(
      '--lr_sched_name',
      type=str,
      default='step_function',
      help='lr schedule scheme name to be picked from lr.py')  
  parser.add_argument(
      '--plot_dir',
      type=str,
      default=os.path.join(os.getenv('HOME'), 'plot_dir'),
      help="""\
      Directory where plots of accuracy vs Epochs are stored
      """)
  parser.add_argument(
      '--preproc_dtype',
      type=str,
      default='uint8',
      help='Data type of the preprocessed (filterbanked) audio')
  parser.add_argument(
      '--force_preproc',
      type=bool,
      default=False,
      help='Re-run the preprocessing even if it appears that good processed data are available')

  Flags, unparsed = parser.parse_known_args()
  return Flags, unparsed
