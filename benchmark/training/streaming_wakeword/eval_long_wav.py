import json, os, pprint
import numpy as np
from scipy.io import  wavfile

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
        model_std = keras.models.load_model(Flags.saved_model_path) # normal model for fixed-length inputs

    ## Build a model that can accept variable-length inputs to process the long waveform/spectrogram
    Flags.variable_length=True
    model_varlen = models.get_model(args=Flags, use_qat=False) # model with variable-length input
    Flags.variable_length=False
    # transfer weights from trained model into variable-length model
    model_varlen.set_weights(model_std.get_weights())


wav_sampling_freq, long_wav = wavfile.read(Flags.test_wav_path)
assert wav_sampling_freq == samp_freq

data_config_long = get_dataset.get_data_config(Flags, 'training')
data_config_long['foreground_volume_max'] = data_config_long['foreground_volume_min'] = 1.0 # scale to [-1.0,1.0]
data_config_long['background_frequency'] = 0.0 # do not add background noise or time-shift the input
data_config_long['time_shift_ms'] = 0.0
data_config_long['desired_samples']= len(long_wav)

long_wav = long_wav / np.max(np.abs(long_wav)) # scale into [-1.0, +1.0] range
t = np.arange(len(long_wav))/samp_freq

feature_extractor_long = get_dataset.get_preprocess_audio_func(data_config_long)
# the feature extractor needs a label (in 1-hot format), but it doesn't matter what it is
long_spec = feature_extractor_long({'audio':long_wav, 'label':[0.0, 0.0, 0.0]})['audio'].numpy()
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

print(f"False detections: {np.sum(ww_false_detects!=0)},",
      f"True detections: {np.sum(ww_true_detects!=0)},",
      f"False rejections: {np.sum(ww_false_rejects!=0)}")
