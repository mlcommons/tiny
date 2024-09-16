# build the calibration subset of the validation split, using Flags.cal_subset=True
# write the spectrograms and labels to an npz file

import numpy as np
import str_ww_util as util
from get_dataset import get_data, get_file_lists, get_data_config

Flags = util.parse_command("extract_cal_subset")
rng = np.random.default_rng()

# these flags need to align with the ones in create_cal_idx_list.py
flags_calibration = get_data_config(Flags, 'validation')
flags_calibration.cal_subset = True
flags_calibration.batch_size = 1
flags_calibration.fraction_target = -1.0 # neg => just use original target samples

data_dir = Flags.data_dir
_, _, val_files = get_file_lists(data_dir)

ds_cal = get_data(flags_calibration, val_files)

for x,y in ds_cal.take(1):
  input_shape = x.shape

num_cal_samples = ds_cal.cardinality()
print(f"Number of cal samples = {num_cal_samples}. Input shape = {input_shape}")

specs = np.zeros((num_cal_samples,) + input_shape[1:])
labels = np.zeros(num_cal_samples)

for i,(spec,lbl) in enumerate(ds_cal):
    labels[i] = np.argmax(lbl)
    specs[i,...] = spec[0,...]

np.savez_compressed("calibration_samples.npz", specs=specs, labels=labels)
