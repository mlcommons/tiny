import numpy as np
import os
import train
from PIL import Image

perf_samples_dir = 'perf_samples'
cifar_10_dir = 'cifar-10-batches-py'

if __name__ == '__main__':
    if not os.path.exists(perf_samples_dir):
        os.makedirs(perf_samples_dir)

    label_output_file = open('y_labels.txt', 'w')

    _idxs = np.load('perf_samples_idxs.npy')
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        train.load_cifar_10_data(cifar_10_dir)

    for i in _idxs:
        _output_str = '{name},{classes},{label}\n'.format(name=test_filenames[i].decode('UTF-8'), classes=10, label=np.argmax(test_labels[i]))
        label_output_file.write(_output_str)
        sample_img = np.array(test_data[i])
        im = Image.fromarray(sample_img, mode='RGB')
        im.save(perf_samples_dir + '/' + test_filenames[i].decode('UTF-8')[:-3] + 'bmp', quality='keep')
    label_output_file.close()
