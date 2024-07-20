import json, os, pprint
import numpy as np
from scipy.io import  wavfile

import str_ww_util as util
import get_dataset

Flags = util.parse_command()

rel_thresh = 0.05 # trim out leading/trailing space with less than rel_thresh*max(waveform)
samp_freq = Flags.sample_rate
wav_spec_file = "long_wav_spec.json"

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
    bg_sampling_freq, tmp_wav = wavfile.read(bg_path)
    assert bg_sampling_freq == samp_freq
    tmp_wav = tmp_wav[:int((t_stop-t_start)*samp_freq)]
    tmp_wav = rms_level*(tmp_wav / np.std(tmp_wav)) # normalize to RMS=1.0 then scale
    long_wav[int(t_start*samp_freq):int(t_stop*samp_freq)] += tmp_wav

ww_present = np.zeros(long_wav.shape)
ww_windows = [] # populate this with start,stop tuples

for fname, insertion_secs, ampl in wav_spec['configs_wakeword']:
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
data_config_long['foreground_volume_max'] = data_config_long['foreground_volume_min'] = 1.0 # scale to [-1.0,1.0]
data_config_long['background_frequency'] = 0.0 # do not add background noise or time-shift the input
data_config_long['time_shift_ms'] = 0.0
data_config_long['desired_samples']= len(long_wav)

long_wav = long_wav / np.max(np.abs(long_wav)) # scale into [-1.0, +1.0] range

feature_extractor_long = get_dataset.get_preprocess_audio_func(data_config_long)
# the feature extractor needs a label (in 1-hot format), but it doesn't matter what it is   
long_spec = feature_extractor_long({'audio':long_wav, 'label':[0.0, 0.0, 0.0]})['audio'].numpy()

print(f"Long waveform shape = {long_wav.shape}, spectrogram shape = {long_spec.shape}")
wavfile.write('long_wav.wav', 16000, (long_wav*(2**15)).astype(np.int16))

pretty_json_str = pprint.pformat(ww_windows, compact=True).replace("(","[").replace(")","]")
with open('long_wav_ww_windows.json', 'w') as fpo:
    fpo.write(pretty_json_str)