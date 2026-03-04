'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

train.py desc: loads data, trains and saves model, plots training metrics
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Lambda, Resizing, RandomFlip, RandomRotation, RandomTranslation
import eval_functions_eembc
from sklearn.metrics import roc_auc_score
import keras_model
import gc

import datetime

EPOCHS = 2
BS = 32

# get date and time to save model
dt = datetime.datetime.today()
year = dt.year
month = dt.month
day = dt.day
hour = dt.hour
minute = dt.minute

"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 
training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains 
exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random 
order, but some training batches may contain more images from one class than another. Between them, the training 
batches contain exactly 5000 images from each class.
"""

#learning rate schedule
def lr_schedule(epoch):
    initial_learning_rate = 3.6950e-5
    decay_per_epoch = 0.99
    lrate = initial_learning_rate * (decay_per_epoch ** epoch)
    print('Learning rate = %f'%lrate)
    return lrate

#lr_scheduler = LearningRateScheduler(lr_schedule)

#optimizer
#optimizer = tf.keras.optimizers.Adam()

#define data generator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    #brightness_range=(0.9, 1.2),
    #contrast_range=(0.9, 1.2),
    validation_split=0.2
)

def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, to_categorical(cifar_train_labels), \
        cifar_test_data, cifar_test_filenames, to_categorical(cifar_test_labels), cifar_label_names

def cifar10_process(train_data, test_data, batch_size=BS, size=(224, 224), enrich=True, verbose=False):
    x, y = size
    train_data_scaled = np.zeros((len(train_data), x, y, 3), dtype=np.uint8)
    test_data_scaled = np.zeros((len(test_data), x, y, 3), dtype=np.uint8)
    for i in range(len(train_data)):
        if i % batch_size == 0:
            end = i + batch_size
            train_data_scaled[i:end] = Resizing(x, y)(train_data[i:end])
            if verbose and i % 512 == 0:
                print(f"Rescaled training images {i}-{end-1}")
            if enrich:
                train_data_scaled[i:end] = RandomFlip('horizontal')(train_data_scaled[i:end])
                train_data_scaled[i:end] = RandomRotation(0.1)(train_data_scaled[i:end])
                train_data_scaled[i:end] = RandomTranslation(0.1, 0.1)(train_data_scaled[i:end])
                if verbose and i % 512 == 0:
                    print(f"Enriched training images {i}-{end-1}")
    for i in range(len(test_data)):
        if i % batch_size == 0:
            end = i + batch_size
            test_data_scaled[i:end] = Resizing(x, y)(test_data[i:end])
            if verbose and i % 512 == 0:
                print(f"Rescaled test images {i}-{end-1}")
            if enrich:
                test_data_scaled[i:end] = RandomFlip('horizontal')(test_data_scaled[i:end])
                test_data_scaled[i:end] = RandomRotation(0.1)(test_data_scaled[i:end])
                test_data_scaled[i:end] = RandomTranslation(0.1, 0.1)(test_data_scaled[i:end])
                if verbose and i % 512 == 0:
                    print(f"Enriched test images {i}-{end-1}")
    return train_data_scaled, test_data_scaled

if __name__ == "__main__":
    """load cifar10 data and trains model"""
    print("Loading and scaling CIFAR10 training and test data to 224x224...", end=" ")
    cifar_10_dir = 'cifar-10-batches-py'

    train_data_raw, train_filenames, train_labels, test_data_raw, test_filenames, test_labels, label_names = \
        load_cifar_10_data(cifar_10_dir)
    
    # rescale stuff, then clear the input vectors
    train_data, test_data = cifar10_process(train_data_raw, test_data_raw, enrich=False)
    print("done\n")
    train_data_raw = None
    test_data_raw = None
    gc.collect()

    print("Train data: ", train_data.shape)
    print("Train filenames: ", train_filenames.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)

    # validation code merged into script
    # includes sentinel for official TinyMLPerf validation sample vs. full CIFAR10 val
    PERF_SAMPLE = True
    if PERF_SAMPLE:
        _idxs = np.load('perf_samples_idxs.npy')
        test_data = test_data[_idxs]
        test_labels = test_labels[_idxs]
        test_filenames = test_filenames[_idxs]

    print()
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)
    label_classes = np.argmax(test_labels,axis=1)
    print("Label classes: ", label_classes.shape)

    # Don't forget that the label_names and filesnames are in binary and need conversion if used.

    # display some random training images in a 25x25 grid
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, train_data.shape[0])
            ax[m, n].imshow(train_data[idx])
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.savefig(f"train_data.png") # replacement for the figure thing because
                                   # this script runs in a non-GUI Docker container

    new_model = keras_model.effnet_v2s(transfer=True)
    new_model.load_weights('trained_models/trainedEffNet.h5')
    # enable training on everything that's not BatchNorm
    for i in new_model.layers:
        if i.name == 'efficientnetv2-s':
            for j in i.layers:
                if not isinstance(j, tf.keras.layers.BatchNormalization):
                    j.trainable = True
            break
    new_model.summary()

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(train_data)

    lr_scheduler = LearningRateScheduler(lr_schedule)
    optimizer = tf.keras.optimizers.Adam()
    new_model.compile(
        optimizer=optimizer, loss='categorical_crossentropy', metrics='accuracy', loss_weights=None,
        weighted_metrics=None, run_eagerly=None)
    test_metrics = new_model.evaluate(x=test_data, y=test_labels, batch_size=32, verbose=1, return_dict=True)

    # fits the model on batches with real-time data augmentation:
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"trained_models/trainedEffNet_checkpoint.h5",
        monitor='accuracy'
    )
    History = new_model.fit(datagen.flow(train_data, train_labels, batch_size=BS),
            steps_per_epoch=len(train_data) / BS, epochs=EPOCHS, callbacks=[lr_scheduler, checkpoint], validation_data=(test_data, test_labels))
    
    dt = datetime.datetime.today()
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    timestamp = f"{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}"
    model_name = f"trainedEffNet_tuned_{timestamp}.h5"
    plt.clf() # clear figure in case the previous figure was not plotted
    plt.plot(np.array(range(EPOCHS)), History.history['loss'])
    plt.plot(np.array(range(EPOCHS)), History.history['val_loss'])
    plt.legend(labels=['Training', 'Validation'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f"trained_models/{model_name}_train_val_loss.png")
    plt.clf()
    plt.plot(np.array(range(EPOCHS)), History.history['accuracy'])
    plt.plot(np.array(range(EPOCHS)), History.history['val_accuracy'])
    plt.legend(labels=['Training', 'Validation'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(f"trained_models/{model_name}_train_val_acc.png")
    new_model.save("trained_models/" + model_name)

    # verify readback
    model = tf.keras.models.load_model('trained_models/' + model_name)

    test_metrics = model.evaluate(x=test_data, y=test_labels, batch_size=32, verbose=1, return_dict=True)

    print("Performances on cifar10 test set")
    print("Keras evaluate method")
    print("Accuracy keras: ", test_metrics['accuracy'])
    print("---------------------")

    predictions = model.predict(test_data)

    print("EEMBC calculate_accuracy method")
    accuracy_eembc = eval_functions_eembc.calculate_accuracy(predictions, label_classes)
    print("---------------------")

    auc_scikit = roc_auc_score(test_labels, predictions)
    print("sklearn.metrics.roc_auc_score method")
    print("AUC sklearn: ", auc_scikit)
    print("---------------------")

    print("EEMBC calculate_auc method")
    auc_eembc = eval_functions_eembc.calculate_auc(predictions, label_classes, label_names, model_name)
    print("---------------------")
    print()
