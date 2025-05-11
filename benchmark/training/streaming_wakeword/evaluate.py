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

Flags = util.parse_command("evaluate")

if Flags.saved_model_path is None:
    err_str = "Model to be evaluated must be specified with --saved_model_path"
    raise RuntimeError(err_str)

det_thresh = 0.95
samp_freq = Flags.sample_rate


if Flags.stream_config is None:
   test_streaming = False
else:
    test_streaming = True
    with open(Flags.stream_config, 'r') as fpi:
        stream_test_config = json.load(fpi)

    wav_file = stream_test_config[0]['wav_file']
    wav_file = os.path.join(Flags.wav_dir, wav_file)
    ww_windows = stream_test_config[0]['detection_windows']

if Flags.specgram:
    specgram = np.load(Flags.specgram)['specgram']
    # accept anything that squeezes to Nx<num_features>,
    # set shape to (1,time_steps,1,features)
    specgram = np.expand_dims(np.squeeze(specgram), [0,2]) 
    long_spec = None
    long_spec_q = None
    if specgram.dtype == np.int8:
        long_spec_q = specgram
        num_frames = long_spec_q.shape[1]
    elif specgram.dtype == np.int16:
        long_spec_q = specgram.astype(np.int8)
        num_frames = long_spec_q.shape[1]
    elif specgram.dtype in [np.dtype('float32'), np.dtype('float64')]:
        long_spec = specgram
        num_frames = long_spec.shape[1]
else:
    long_spec = None
    long_spec_q = None

use_tflite = Flags.saved_model_path.rsplit('.')[-1] in ["tflite", "tfl"]

if use_tflite:
    interpreter = tf.lite.Interpreter(model_path=Flags.saved_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    output_data = []
    labels = []
    input_scale, input_zero_point = input_details[0]["quantization"]
    output_scale, output_zero_point = output_details[0]["quantization"]
elif Flags.saved_model_path.rsplit('.')[-1] in ["h5", "keras"]:
    with tfmot.quantization.keras.quantize_scope(): # needed for the QAT wrappers
        model_std = keras.models.load_model(Flags.saved_model_path) # normal model for fixed-length inputs
    ## Build a model that can accept variable-length inputs to process the long waveform/spectrogram
    Flags.variable_length=True
    model_varlen = models.get_model(args=Flags, use_qat=False) # model with variable-length input
    Flags.variable_length=False
    # transfer weights from trained model into variable-length model
    model_varlen.set_weights(model_std.get_weights())
else:
    raise RuntimeError("Model file ({Flags.saved_model_path}) must end in '.tflite', '.h5', or '.keras'.")

if test_streaming:
    long_wav = None
    if long_spec is None and long_spec_q is None:
        wav_sampling_freq, long_wav = wavfile.read(wav_file)
        assert wav_sampling_freq == samp_freq

        if len(long_wav.shape) > 1: # stereo (or higher multi-channel) wav
            long_wav = long_wav[:,0] # just take 1st channel

        long_wav = long_wav / 2**15 # scale into [-1.0, +1.0] range
        t = np.arange(len(long_wav))/samp_freq

        data_config = get_dataset.get_data_config(Flags, 'validation')
        feature_extractor = get_dataset.get_lfbe_func(data_config)
        long_spec = feature_extractor(long_wav).numpy()
        long_spec = np.expand_dims(long_spec, 0) # add batch dimension  
        num_frames = long_spec.shape[1]
    if long_wav is not None:
        print(f"Long waveform shape = {long_wav.shape}. ")
    if long_spec is not None:
        print(f"Long spectrogram (float) shape = {long_spec.shape}. ")
    if long_spec_q is not None:
        print(f"Long spectrogram (quantized) shape = {long_spec_q.shape}. ")



    if use_tflite:
        yy_q = np.nan*np.zeros((num_frames-input_shape[1]+1,3))
        if long_spec_q is None:
            long_spec_q = np.array(long_spec/input_scale + input_zero_point, dtype=np.int8)
        
        for idx in range(num_frames-input_shape[1]+1):
            spec_q = long_spec_q[0:1, idx:idx+input_shape[1],:,:]

            interpreter.set_tensor(input_details[0]['index'], spec_q)
            interpreter.invoke()
            # get_tensor() returns a copy of the tensor data.
            # Use `tensor()` to get a pointer to it
            yy_q[idx,:] = interpreter.get_tensor(output_details[0]['index'])

        # Dequantize so that softmax output is in range [0,1]
        yy = (yy_q.astype(np.float32) - output_zero_point)*output_scale
    else:
        yy = model_varlen(long_spec)[0].numpy()

    ww_present = np.zeros(int(stream_test_config[0]['length_sec']*stream_test_config[0]['sample_rate']))
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

if not use_tflite:
    val_loss, val_acc, val_prec, val_recl = model_std.evaluate(ds_val)

print(f"Results: false_detections={np.sum(ww_false_detects!=0)},",
      f"true_detections={np.sum(ww_true_detects!=0)},",
      f"false_rejections={np.sum(ww_false_rejects!=0)},", end=""
      )

if not use_tflite:
    print(f"val_loss={val_loss:5.4f}, val_acc={val_acc:5.4f}, val_precision={val_prec:5.4f}, val_recall={val_recl:5.4f}")
else:
    print("") # We need a newline
