import json, os, pprint
import numpy as np
from scipy.io import  wavfile

import str_ww_util as util
import get_dataset

Flags = util.parse_command("build_long_wav")

# trim out leading/trailing space with less than rel_thresh*max(waveform)
rel_thresh = Flags.rel_thresh
samp_freq = Flags.sample_rate
wav_spec_file = Flags.wav_spec

wav_filename = Flags.long_wav_name
specgram_filename = Flags.long_wav_name.rsplit('.',1)[0]  # all but extension
specgram_filename += "_specgram.npz"
window_filename = Flags.long_wav_name.rsplit('.',1)[0]  # all but extension
window_filename += "_ww_windows.json"

try:
    with open('streaming_config.json', 'r') as fpi:
        streaming_config = json.load(fpi)
    musan_path = streaming_config['musan_path']
except:
    raise RuntimeError("""
        In this directory, copy streaming_config_template.json to streaming_config.json
        and edit it to point to the directories where you have the speech commands dataset
        and the MUSAN noise data set.
        """)

with open(wav_spec_file, 'r') as fpi:
    wav_spec = json.load(fpi)
long_wav_len_sec = wav_spec['length_sec']


def trim_and_normalize(wav_in, rel_thresh):
  """
  Trims leading and trailing 'quiet' segments, where quiet is defined as 
  less than rel_thresh*max(wav_in).
  Then scales such that RMS of trimmed wav = 1.0
  """
  idx_start = np.min(np.nonzero(ww_wav > np.max(ww_wav)*rel_thresh))
  idx_stop  = np.max(np.nonzero(ww_wav > np.max(ww_wav)*rel_thresh))
  
  wav_out = wav_in[idx_start:idx_stop]
  wav_out = wav_out / np.std(wav_out) 
  return wav_out


long_wav = np.zeros(int(long_wav_len_sec*samp_freq), dtype=np.float32)
for bg_path, t_start, t_stop, rms_level in wav_spec['configs_background']: 
    bg_path = util.replace_env_vars(bg_path, env_dict=streaming_config)
    bg_sampling_freq, tmp_wav = wavfile.read(bg_path)
    assert bg_sampling_freq == samp_freq
    tmp_wav = tmp_wav[:int((t_stop-t_start)*samp_freq)]
    tmp_wav = rms_level*(tmp_wav / np.std(tmp_wav)) # normalize to RMS=1.0 then scale
    long_wav[int(t_start*samp_freq):int(t_stop*samp_freq)] += tmp_wav

ww_present = np.zeros(long_wav.shape)
ww_windows = [] # populate this with start,stop tuples

for fname, insertion_secs, ampl in wav_spec['configs_wakeword']:
    fname = util.replace_env_vars(fname, env_dict=streaming_config)
    ww_sampling_freq, ww_wav = wavfile.read(fname)
    assert int(ww_sampling_freq) == samp_freq
    index = int(insertion_secs * samp_freq)
    
    ww_wav = trim_and_normalize(ww_wav, rel_thresh)
    assert index+len(ww_wav) < len(long_wav)
    
    long_wav[index:index+len(ww_wav)] += ampl*ww_wav
    ww_present[index:index+len(ww_wav)] = 1
    # each ww_window is the start,stop time in seconds when the wakeword is present
    ww_windows.append((insertion_secs,insertion_secs+len(ww_wav)/samp_freq))


data_config_long = get_dataset.get_data_config(Flags, 'training')
# data_config_long['foreground_volume_max'] = data_config_long['foreground_volume_min'] = 1.0 # scale to [-1.0,1.0]
# data_config_long['background_frequency'] = 0.0 # do not add background noise or time-shift the input
# data_config_long['time_shift_ms'] = 0.0
# data_config_long['desired_samples']= len(long_wav)

import json
with open("data_config_script.json", "w") as fpo:
    json.dump(data_config_long, fpo, indent=4)

long_wav = long_wav / np.max(np.abs(long_wav)) # scale into [-1.0, +1.0] range
# emulate INT16 quantization, dequant, so this spectrogram matches 
# one from reading the wav file.
long_wav_int16 = (long_wav*(2**15)).astype(np.int16)

feature_extractor_long = get_dataset.get_lfbe_func(data_config_long)

wavfile.write(wav_filename, 16000, long_wav_int16)

# the feature extractor needs a label (in 1-hot format), but it doesn't matter what it is   
# long_spec = feature_extractor_long({'audio':long_wav_int16/2**15, 'label':[0.0, 0.0, 0.0]})['audio'].numpy()
long_spec = feature_extractor_long(long_wav_int16/2**15).numpy()
print(f"Long waveform shape = {long_wav.shape}, spectrogram shape = {long_spec.shape}")

np.savez_compressed(specgram_filename, specgram=long_spec)

pretty_json_str = pprint.pformat(ww_windows, compact=True).replace("(","[").replace(")","]")
with open(window_filename, 'w') as fpo:
    fpo.write(pretty_json_str)
    fpo.write("\n")