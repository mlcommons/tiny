import numpy as np
import os
import train

RANDOM_SEED = 8108
perf_samples_dir = 'perf_samples'
cifar_10_dir = 'cifar-10-batches-py'

if __name__ == '__main__':
    if not os.path.exists(perf_samples_dir):
        os.makedirs(perf_samples_dir)

    label_output_file = open('y_labels.csv', 'w')

    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        train.load_cifar_10_data(cifar_10_dir)

    my_labels = test_labels.argmax(axis=-1)
    all_idxs = np.arange(len(test_labels))
    balanced_idxs = []
    rng = np.random.default_rng(RANDOM_SEED)
    for label in range(10):
        mask = my_labels == label
        class_idxs = rng.choice(all_idxs[mask], size=20, replace=False)
        balanced_idxs.append(class_idxs)

    _idxs = np.concatenate(balanced_idxs)
    rng.shuffle(_idxs)    
    print("Perf IDs: ", _idxs)
    print("Perf labels: ", my_labels[_idxs])
    np.save('perf_samples_idxs.npy', _idxs)
