import numpy as np
import os
import train
from PIL import Image

perf_samples_dir = 'perf_samples'
cifar_10_dir = 'cifar-10-batches-py'

if __name__ == '__main__':
    if not os.path.exists(perf_samples_dir):
        os.makedirs(perf_samples_dir)

    _idxs = np.load('perf_samples_idxs.npy')
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        train.load_cifar_10_data(cifar_10_dir)

    for i in _idxs:
        sample_img = np.array(test_data[i])
        im = Image.fromarray(sample_img)
        im.save(perf_samples_dir + '/' + test_filenames[i].decode('UTF-8'))