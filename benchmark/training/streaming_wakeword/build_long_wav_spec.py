import json, os, pprint
import numpy as np
import str_ww_util as util

Flags = util.parse_command()

long_wav_len_sec = 20*60.0 # 20 minutes
num_targets = 100
min_ww_ampl = 0.25
max_ww_ampl = 0.75

rng = np.random.default_rng()

bg_file_configs = [
    ('{musan_path}'+'/speech/librivox/speech-librivox-0149.wav', 100.0, 300.0, 0.1),
    ('{musan_path}'+'/speech/librivox/speech-librivox-0150.wav', 100.0, 300.0, 0.1),
    ('{musan_path}'+'/speech/librivox/speech-librivox-0152.wav', 100.0, 300.0, 0.1),
    ('{musan_path}'+'/speech/librivox/speech-librivox-0163.wav', 100.0, 300.0, 0.1),
    ('{musan_path}'+'/speech/librivox/speech-librivox-0163.wav', 100.0, 300.0, 0.1),
    ('{musan_path}'+'/speech/librivox/speech-librivox-0164.wav', 100.0, 300.0, 0.1),
    ('{musan_path}'+'/speech/librivox/speech-librivox-0165.wav', 100.0, 300.0, 0.1),
    ('{musan_path}'+'/speech/librivox/speech-librivox-0166.wav', 100.0, 300.0, 0.1),
    ('{musan_path}'+'/music/hd-classical/music-hd-0002.wav', 300.0, 450.0, 0.5),
    ('{musan_path}'+'/music/jamendo/music-jamendo-0000.wav', 450.0, 600.0, 0.5),
    ('{musan_path}'+'/speech/librivox/speech-librivox-0052.wav', 600.0, 1200.0, 0.5),
]

intervals = rng.exponential(scale=7.0, size=(num_targets))
# intervals < 3 sec may mess up the counting
intervals = intervals[intervals>3.0]
insertion_secs = np.cumsum(intervals[:50])
# This should spread the words out in time, but if the parameters
# are changed, then double check this.
# stop the wake words 95% into the 1st half.
insertion_secs *= (.475*long_wav_len_sec)/insertion_secs[-1] 
ww_amplitudes = rng.uniform(low=min_ww_ampl, high=min_ww_ampl, size=(len(insertion_secs)))

ww_files = []
for line in open(os.path.join(Flags.data_dir, 'testing_list.txt')):
    if line.split('/')[0] == 'marvin':
        ww_path = "{speech_commands_path}" + line.strip()
        ww_files.append(ww_path)

ww_files = ww_files[:len(insertion_secs)]
# fg_configs = [(f,t,a) for (f,t,a) in zip(ww_files, 
fg_configs = [(f,t,a) for (f,t,a) in zip(ww_files, insertion_secs, ww_amplitudes)]

long_wav_config = {
    "configs_background":bg_file_configs, 
    "configs_wakeword":fg_configs,
    "length_sec":long_wav_len_sec,
    "sample_rate":Flags.sample_rate
    }

# nicely format, but use json-compatible "double quotes"
pretty_json_str = pprint.pformat(long_wav_config, compact=True).replace("'",'"')
pretty_json_str = pretty_json_str.replace("(","[").replace(")","]")
with open('long_wav_spec.json', 'w') as fpo:
    fpo.write(pretty_json_str)
    # json.dump(long_wav_config, fpo, indent=4)


