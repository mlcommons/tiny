import json, os, pprint
import numpy as np
from scipy.io import  wavfile

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' 

import tensorflow_model_optimization as tfmot
import tensorflow as tf
import keras

import str_ww_util as util
import keras_model as models
import get_dataset

Flags = util.parse_command()

det_thresh = 0.95
samp_freq = Flags.sample_rate
# For wav file 'long_wav.wav', the wakeword windows should be in 'long_wav_ww_windows.json'
ww_windows_file = Flags.test_wav_path.split('.')[0] + '_ww_windows.json'

if Flags.use_tflite_model:
    interpreter = tf.lite.Interpreter(model_path=Flags.tfl_file_name)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    output_data = []
    labels = []
    input_scale, input_zero_point = input_details[0]["quantization"]
    output_scale, output_zero_point = output_details[0]["quantization"]
else:
    with tfmot.quantization.keras.quantize_scope(): # needed for the QAT wrappers
        model_std = keras.models.load_model(Flags.model_init_path) # normal model for fixed-length inputs

    ## Build a model that can accept variable-length inputs to process the long waveform/spectrogram
    Flags.variable_length=True
    model_varlen = models.get_model(args=Flags, use_qat=False) # model with variable-length input
    Flags.variable_length=False
    # transfer weights from trained model into variable-length model
    model_varlen.set_weights(model_std.get_weights())

wav_sampling_freq, long_wav = wavfile.read(Flags.test_wav_path)
assert wav_sampling_freq == samp_freq

data_config = get_dataset.get_data_config(Flags, 'validation')

long_wav = long_wav / np.max(np.abs(long_wav)) # scale into [-1.0, +1.0] range
t = np.arange(len(long_wav))/samp_freq

feature_extractor = get_dataset.get_lfbe_func(data_config)
long_spec = feature_extractor(long_wav).numpy()
print(f"Long waveform shape = {long_wav.shape}, spectrogram shape = {long_spec.shape}")

if Flags.use_tflite_model:
    yy_q = np.nan*np.zeros((long_spec.shape[0]-input_shape[1]+1,3))

    for idx in range(long_spec.shape[0]-input_shape[1]+1):
        spec = long_spec[idx:idx+input_shape[1],:,:]
        spec = np.expand_dims(spec, 0) # add batch dimension  
        spec_q = np.array(spec/input_scale + input_zero_point, dtype=np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], spec_q)
        interpreter.invoke()
        # get_tensor() returns a copy of the tensor data.
        # Use `tensor()` to get a pointer to it
        yy_q[idx,:] = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize so that softmax output is in range [0,1]
    yy = (yy_q.astype(np.float32) - output_zero_point)*output_scale
else:
    yy = model_varlen(np.expand_dims(long_spec, 0))[0].numpy()

## shows detection when ww activation > thresh
with open(ww_windows_file, 'r') as fpi:
  ww_windows = json.load(fpi)
ww_present = np.zeros(len(long_wav))
for t_start, t_stop in ww_windows:
  idx_start = int(t_start*samp_freq)
  idx_stop  = int(t_stop*samp_freq)
  ww_present[idx_start:idx_stop] = 1

ww_detected_spec_scale = (yy[:,0]>det_thresh).astype(int)
ww_true_detects, ww_false_detects, ww_false_rejects = util.get_true_and_false_detections(ww_detected_spec_scale, ww_present, Flags)

flags_validation = get_dataset.get_data_config(Flags, 'validation')
flags_validation.batch_size = 50

## Build the data sets from files
data_dir = Flags.speech_commands_path
_, _, val_files = get_dataset.get_file_lists(data_dir)
ds_val = get_dataset.get_data(flags_validation, val_files)

if not Flags.use_tflite_model:
    val_loss, val_acc, val_prec, val_recl = model_std.evaluate(ds_val)

print(f"Results: false_detections={np.sum(ww_false_detects!=0)},",
      f"true_detections={np.sum(ww_true_detects!=0)},",
      f"false_rejections={np.sum(ww_false_rejects!=0)},", end=""
      )

if not Flags.use_tflite_model:
    print(f"val_loss={val_loss:5.4f}, val_acc={val_acc:5.4f}, val_precision={val_prec:5.4f}, val_recall={val_recl:5.4f}")

