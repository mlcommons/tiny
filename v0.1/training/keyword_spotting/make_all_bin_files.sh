
echo "\n*** Generating Log Filterbank Energy binary files (lfbe) ***"
python make_bin_files.py --bin_file_path=${HOME}/kws_bin_files/lfbe --feature_type=lfbe

echo "\n*** Generating Mel-frequency Cepstral Coefficient files (mfcc) ***"
python make_bin_files.py --bin_file_path=${HOME}/kws_bin_files/mfcc --feature_type=mfcc

echo "\n*** Generating Raw Time-domain Waveform binary files (td_sample) ***"
python make_bin_files.py --bin_file_path=${HOME}/kws_bin_files/td_samples --feature_type=td_samples
