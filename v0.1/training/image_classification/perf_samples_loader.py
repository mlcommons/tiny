import numpy as np
import os
import train
import struct

perf_samples_dir = 'perf_samples'
cifar_10_dir = 'cifar-10-batches-py'

if __name__ == '__main__':
    if not os.path.exists(perf_samples_dir):
        os.makedirs(perf_samples_dir)

    label_output_file = open('y_labels.csv', 'w')

    _idxs = np.load('perf_samples_idxs.npy')
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        train.load_cifar_10_data(cifar_10_dir)

    for i in _idxs:
        _output_str = '{name},{classes},{label}\n'.format(name=test_filenames[i].decode('UTF-8')[:-3] + 'bin', classes=10, label=np.argmax(test_labels[i]))
        label_output_file.write(_output_str)
        sample_img = np.array(test_data[i]).flatten()

        f = open(perf_samples_dir + '/' + test_filenames[i].decode('UTF-8')[:-3] + 'bin', "wb")
        mydata = sample_img
        myfmt = 'B' * len(mydata)
        bin = struct.pack(myfmt, *mydata)
        f.write(bin)
        f.close()

    label_output_file.close()
