import numpy as np
import str_ww_util as util
from get_dataset import get_data, get_file_lists, get_data_config

Flags = util.parse_command()

rng = np.random.default_rng()

flags_validation = get_data_config(Flags, 'validation')

## Build the data sets from files
data_dir = Flags.data_dir
_, _, val_files = get_file_lists(data_dir)

ds_val = get_data(flags_validation, val_files)

label_mat = np.zeros((ds_val.cardinality()*ds_val._batch_size.numpy(), 3))
with open('val_labels_2.txt', 'w') as fpo:
  for i,(spec,lbl) in enumerate(ds_val.unbatch()):
    fpo.write(f"{i:4} {lbl}\n")
    label_mat[i,:] = lbl
label_mat = label_mat[np.sum(label_mat, axis=1)!=0] # discard the unpopulated remainder of the terminal partial batch

idx_class0 = np.nonzero(label_mat[:,0]==1)[0]
idx_class1 = np.nonzero(label_mat[:,1]==1)[0]
idx_class2 = np.nonzero(label_mat[:,2]==1)[0]

cal_samples_per_class = 15
idx_class0_cal = rng.choice(idx_class0, size=cal_samples_per_class, replace=False)
idx_class1_cal = rng.choice(idx_class1, size=cal_samples_per_class, replace=False)
idx_class2_cal = rng.choice(idx_class2, size=cal_samples_per_class, replace=False)

cal_idxs = np.concatenate((idx_class0_cal, idx_class1_cal, idx_class2_cal))
with open("quant_cal_idxs.txt", "w") as fpo:
  for i in cal_idxs:
    fpo.write(f"{i}\n")