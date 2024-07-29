import argparse, os, math, re
import matplotlib.pyplot as plt
import numpy as np

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
      '--test_wav_path',
      type=str,
      default="long_wav.wav",
      help="""\
      Wav file to run the model on for the long-wav test.
      """)
  # the long test waveform (final test) draws from speech/librivox, music/hd-classical under MUSAN
  default_bg_path_train = os.path.join(os.getenv('HOME'), 'data', 'speech_commands_v0.02', '_background_noise_')
  default_bg_path_train += "," + os.path.join(os.getenv('HOME'), 'data', "musan", "noise", "free-sound")
  default_bg_path_train += "," + os.path.join(os.getenv('HOME'), 'data', "musan", "speech", "us-gov")

  default_bg_path_val = os.path.join(os.getenv('HOME'), 'data', "musan", "noise", "sound-bible")
  default_bg_path_val += "," + os.path.join(os.getenv('HOME'), 'data', "musan", "speech", "librivox")

  parser.add_argument(
      '--background_path_training',
      type=str,
      default=default_bg_path_train,
      help="""\
      Where to find background noise wav files for training. Directories separated by comma (',').
      """)
  parser.add_argument(
      '--background_path_validation',
      type=str,
      default=default_bg_path_val,
      help="""\
      Where to find background noise wav files for training. Directories separated by comma (',').
      """)
  parser.add_argument(
      '--background_path_test',
      type=str,
      default='',
      help="""\
      Where to find background noise wav files for test. Directories separated by comma (',').
      """)        
  parser.add_argument(
      '--num_background_clips',
      type=int,
      default=100,
      help="""\
      Number of (15-sec) background clips to assemble.  Used to augment samples with background noise.  Too high will use excessive memory.
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
      '--fraction_target_training',
      type=float,
      default=0.2,
      help="""\
      Fraction (0.0-1.0) of the training set that should be the target wakeword.
      Target words will be duplicated to reach this fraction, but no words will be discarded,
      even if that results in the fraction of target words being higher than requested.
      """)
  parser.add_argument(
      '--fraction_target_validation',
      type=float,
      default=0.2,
      help="""\
      Fraction (0.0-1.0) of the validation set that should be the target wakeword
      Target words will be duplicated to reach this fraction, but no words will be discarded,
      even if that results in the fraction of target words being higher than requested.
      """)
  parser.add_argument(
      '--fraction_target_test',
      type=float,
      default=0.2,
      help="""\
      Fraction (0.0-1.0) of the test set that should be the target wakeword.
      Target words will be duplicated to reach this fraction, but no words will be discarded,
      even if that results in the fraction of target words being higher than requested.
      """)      
  parser.add_argument(
      '--fraction_silent_training',
      type=float,
      default=0.1,
      help="""\
      Fraction (0.0-1.0) of training set consisting of silent frames (before noise is added).
      """)
  parser.add_argument(
      '--fraction_silent_validation',
      type=float,
      default=0.1,
      help="""\
      Fraction (0.0-1.0) of validation set consisting of silent frames (before noise is added).
      """)
  parser.add_argument(
      '--fraction_silent_test',
      type=float,
      default=0.1,
      help="""\
      Fraction (0.0-1.0) of test set consisting of silent frames (before noise is added).
      """)      
  parser.add_argument(
      '--foreground_volume_min_training',
      type=float,
      default=0.25,
      help="""\
      For training set, minimum level for how loud the foreground words should be, between 0 and 1. Word volume will vary 
      randomly, uniformly  between foreground_volume_min and foreground_volume_max.
      """)
  parser.add_argument(
      '--foreground_volume_min_validation',
      type=float,
      default=0.25,
      help="""\
      (Validation set) Minimum level for how loud the foreground words should be, between 0 and 1. Word volume will vary 
      randomly, uniformly  between foreground_volume_min and foreground_volume_max.
      """)
  parser.add_argument(
      '--foreground_volume_min_test',
      type=float,
      default=0.25,
      help="""\
      (Test set) Minimum level for how loud the foreground words should be, between 0 and 1. Word volume will vary 
      randomly, uniformly  between foreground_volume_min and foreground_volume_max.
      """)
  parser.add_argument(
      '--foreground_volume_max_training',
      type=float,
      default=1.5,
      help="""\
      Maximum level for how loud the foreground words should be in the training set, between 0 and 1. Word
      volume will vary randomly, uniformly  between foreground_volume_min and foreground_volume_max. 
      """)
  parser.add_argument(
      '--foreground_volume_max_validation',
      type=float,
      default=1.5,
      help="""\
      Maximum level for how loud the foreground words should be in the val set, between 0 and 1. Word volume 
      will vary randomly, uniformly  between foreground_volume_min and foreground_volume_max.
      """)  
  parser.add_argument(
      '--foreground_volume_max_test',
      type=float,
      default=1.5,
      help="""\
      Maximum level for how loud the foreground words should be in the test set, between 0 and 1. Word volume
      will vary randomly, uniformly  between foreground_volume_min and foreground_volume_max.
      """)        
  parser.add_argument(
      '--background_volume_training',
      type=float,
      default=1.5,
      help="""\
      How loud the background noise should be, between 0 and 1.  Noise volume will vary 
      randomly between zero and background_volume.
      """)
  parser.add_argument(
      '--background_volume_validation',
      type=float,
      default=1.5,
      help="""\
      How loud the background noise should be, between 0 and 1.  Noise volume will vary 
      randomly between zero and background_volume.
      """)
  parser.add_argument(
      '--background_volume_test',
      type=float,
      default=0.0,
      help="""\
      How loud the background noise should be, between 0 and 1.  Noise volume will vary 
      randomly between zero and background_volume.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      What fraction of the samples have background noise mixed in.
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
    help='Total samples in the training set. Set to -1 to use all the data',)
  parser.add_argument(
      '--num_samples_validation',
      type=int,
      default=-1, # 10102,
    help='Total samples in the  validation set. Set to -1 to use all the data',)
  parser.add_argument(
      '--num_samples_test',
      type=int,
      default=-1, # 4890,
    help='How many samples from the test set to use. Set to -1 to use all the data',)
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size for training',)
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
      help='In quantize.py or evaluate.py, path to load pretrained model from; in train.py, destination for trained model')
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
      default='trained_models/strm_ww_int8.tflite',
      help='File name to which the TF Lite model will be saved (quantize.py) or loaded (eval_quantized_model)')
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
      help='Initial LR',)
  parser.add_argument(
      '--l2_reg',
      type=float,
      default=0.001,
      help='L2 regularization coefficient for conv layers',)      
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
  parser.add_argument(
      '--use_tflite_model',
      action="store_true",
      help="""\
        In eval_long_wav.py, run the TFLite model. Otherwise, run the standard keras model
        """)

  Flags = parser.parse_args()

  # the 2nd half of these assignments is so that an empty original string yields an empty list instead of ['']
  Flags.background_path_training   = Flags.background_path_training.split(',') if len(Flags.background_path_training)>0 else []
  Flags.background_path_validation = Flags.background_path_validation.split(',') if len(Flags.background_path_validation)>0 else []
  Flags.background_path_test       = Flags.background_path_test.split(',') if len(Flags.background_path_test)>0 else []
  
  if Flags.foreground_volume_min_training > Flags.foreground_volume_max_training:
    raise ValueError(f"foreground_volume_min_training ({Flags.foreground_volume_min_training}) must be no",
                     f"larger than foreground_volume_max_training ({Flags.foreground_volume_max_training})")

  if Flags.foreground_volume_min_validation > Flags.foreground_volume_max_validation:
    raise ValueError(f"foreground_volume_min_validation ({Flags.foreground_volume_min_validation}) must be no",
                     f"larger than foreground_volume_max_validation ({Flags.foreground_volume_max_validation})")
                     
  if Flags.foreground_volume_min_test > Flags.foreground_volume_max_test:
    raise ValueError(f"foreground_volume_min_test ({Flags.foreground_volume_min_test}) must be no",
                     f"larger than foreground_volume_max_test ({Flags.foreground_volume_max_test})")

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


    
class DictWrapper(dict):
    """
    Allow access to dictionary through d.attr as well as d['attr']
    Taken from https://stackoverflow.com/questions/50841165/how-to-correctly-wrap-a-dict-in-python-3
    """
    def __getattr__(self, item):
        return super().__getitem__(item)

    def __setattr__(self, item, value):
        return super().__setitem__(item, value)

def zero2nan(x):
    y = x.copy()
    y[y==0] = np.nan
    return y

def debounce_detections(detection_signal, sample_rate=16000, debounce_time=1.0):
    # repeated short positives near the same time should only count as one detection,
    # but an excessively long detection (e.g. 5 sec of continuous positive) should 
    # count as multiple detections.  So, the algorithm is:
    # * we walk through the detection signal (with correct detections already masked out).  
    # * When we encounter a detection, count it and zero out the next fp_debounce_samples (~1s *16ks/s)
    # * continue to the next detection
    # this may be slow

    # In ww_false_detected, positives in the original model output are '1'
    # During the debouncing process, we'll zero out all of the original detections
    # and use '-1' to keep track of the beginning of each separate detection
    # Then we'll convert -1 back to +1 so each +1 indicates the beginning of
    # one distinct detection.

    debounce_samples = int(debounce_time*sample_rate)
    idx = 0
    detection_signal = np.copy(detection_signal)
    while idx < len(detection_signal):
        if not np.any(detection_signal[idx:]==1):
            break
        idx = np.min(np.nonzero(detection_signal==1))

        detection_signal[idx:idx+debounce_samples] = 0
        detection_signal[idx] = -1 # so that it's not caught by the nonzero condition above.
        idx = idx+debounce_samples
      
    detection_signal[detection_signal==-1] = 1
    return detection_signal

def get_true_and_false_detections(detection_signal, ww_present, Flags,
                                  true_pos_max_delay_sec=0.5, false_pos_suppresion_delay_sec=1.0
                                 ):
    """
    detection_signal:       binary output of the detector (0=absent, 1=present), on the time-scale of spectrograms,
                            i.e. at the same effective sampling rate as the model output (not the acoustic sample rate)
    ww_present:             ground truth for when the wakeword is present, at the acoustic sample rate
    
    true_pos_max_delay_sec: After the end of a wakeword instance, detection_signal must be high within 
                            <true_pos_max_delay_sec> to register a true positive
    false_pos_suppresion_delay_sec: After the end of a wakeword instance, positives within 
                                    <false_pos_suppresion_delay_sec> will *not* cause a false positive.

    returns:                (true_detects, false_detects, false_rejects)
                            debounced true and false detections and false_rejects
                            All are at the same sample rate and length as ww_present (plottable directly 
                            with the original waveform).
                            Both are 1 at the start of a detection (or missed wakeword), 0 elsewhere.
                            
                                            
    """
  
    # count true detections for det_delay_tol samples after ww is present
    det_delay_tol = int(true_pos_max_delay_sec*Flags.sample_rate)
    # detections within det_mask_delay samples after the ww end are ignored during false positive counting
    det_mask_delay = int(false_pos_suppresion_delay_sec*Flags.sample_rate)

    # match sample rate and length to ww_present
    ww_detected = np.repeat(detection_signal, Flags.window_stride_ms*Flags.sample_rate/1000)
    extra_zeros = np.zeros(len(ww_present)-len(ww_detected))
    ww_detected = np.concatenate((extra_zeros, ww_detected), axis=0)
    
    ww_false_detected = np.copy(ww_detected)
    ww_false_rejected = np.zeros(ww_detected.shape)
    ww_true_detected  = np.zeros(ww_detected.shape)

    ww_starts = np.nonzero(np.diff(ww_present)>0.5)[0]
    ww_stops = np.nonzero(np.diff(ww_present)<-0.5)[0]
    if len(ww_starts) != len(ww_stops) or ww_present[0] > 0 or ww_present[-1] > 0:
        raise ValueError("ww_present should have the same number of rising and falling",
                         "edges and should start and end with '0's, i.e. no wakeword should",
                         "be in progress at the beginning or end of the waveform.")
    ww_windows = [(n_start, n_stop) for (n_start, n_stop) in zip(ww_starts, ww_stops)]
  
    true_positives = 0
    false_negatives = 0
    for win in ww_windows:
        if np.any(ww_detected[win[0]:win[1]+det_delay_tol]==1): # is detection signal positive within the window?
            true_positives += 1
            ww_true_detected[win[0]] = 1 # just mark the beginning of the actual wakeword 
            # print(f"Counted a true positive for wakeword at {win[0]}:{win[1]}")
        else:
            false_negatives += 1
            ww_false_rejected[win[0]] = 1
            # print(f"Counted a false negative for wakeword at {win[0]}:{win[1]}")
        # mask out any detections in the window or within det_mask_delay samples after it
        ww_false_detected[win[0]:win[1]+det_mask_delay]=0

    ww_false_detected = debounce_detections(ww_false_detected, Flags.sample_rate)
    return ww_true_detected,  ww_false_detected, ww_false_rejected  

def replace_env_vars(str_in, env_dict=None):
    """
    Replaces any sub-strings enclosed by curly braces in str_in with the value 
    of the corresponding variable from env_dict.
    env_dict: a dict (or dict-like) of the form {'VAR_NAME':'value'}
    """
    if env_dict is None:
        env_dict = os.environ

    matches = re.findall(r"{(\w+)}", str_in)
    new_str = str_in
    for env_var_name in matches:
        if env_var_name in env_dict:
            env_var_value = env_dict[env_var_name]
            # Replace the enclosed string and curly braces with the value of the environment variable
            new_str = new_str.replace("{" + env_var_name + "}", env_var_value)
        else:
            raise ValueError(f"Environment variable {env_var_name} not found")
            
    return new_str

