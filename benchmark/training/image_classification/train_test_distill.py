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
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Resizing, RandomFlip, RandomRotation, RandomTranslation
import eval_functions_eembc
from sklearn.metrics import roc_auc_score
import keras_model
import gc

import datetime
import random

EPOCHS = 1000
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
# def lr_schedule(epoch):
#     initial_learning_rate = 0.001
#     decay_per_epoch = 0.99
#     lrate = initial_learning_rate * (decay_per_epoch ** epoch)
#     print('Learning rate = %f'%lrate)
#     return lrate

# Reference:
# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )

#lr_scheduler = LearningRateScheduler(lr_schedule)

#optimizer
#optimizer = tf.keras.optimizers.Adam()

# Reference:
# https://www.tutorialexample.com/implement-kl-divergence-loss-in-tensorflow-tensorflow-tutorial/
def kl_divergence(true_p, q):
    true_prob = tf.nn.softmax(true_p, axis = 1)
    loss_1 = -tf.nn.softmax_cross_entropy_with_logits(logits=true_p, labels=true_prob)
    loss_2 = tf.nn.softmax_cross_entropy_with_logits(logits=q, labels=true_prob)   
    loss = loss_1 + loss_2
    return loss

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

def cifar10_batch(train_data, test_data, batch_size=BS, verbose=False):
    num_train_steps = int(len(train_data) / batch_size) + 1
    num_test_steps = int(len(test_data) / batch_size) + 1
    (_, x, y, z) = train_data.shape
    train_data_batched = []
    test_data_batched = []
    j = 0
    for i in range(len(train_data)):
        if i % batch_size == 0:
            if i + batch_size >= len(train_data):
                end = len(train_data)
            else:
                end = i + batch_size
            train_data_batched.append(train_data[i:end])
            j += 1
    j = 0
    for i in range(len(test_data)):
        if i % batch_size == 0:
            if i + batch_size >= len(test_data):
                end = len(test_data)
            else:
                end = i + batch_size
            test_data_batched.append(test_data[i:end])
            j += 1
    
    val_samples = int(len(train_data) * 0.2)
    random.shuffle(train_data)
    val_data_batched = train_data_batched[:val_samples]
    train_data_batched = train_data_batched[val_samples:]
    
    return train_data_batched, val_data_batched, test_data_batched

if __name__ == "__main__":
    """load cifar10 data and trains model"""
    print("Loading and scaling CIFAR10 training and test data to 224x224...", end=" ")
    cifar_10_dir = 'cifar-10-batches-py'

    train_data_raw, train_filenames, train_labels, test_data_raw, test_filenames, test_labels, label_names = \
        load_cifar_10_data(cifar_10_dir)
    train_data, test_data = cifar10_process(train_data_raw, test_data_raw,
                                            size=(32,32), enrich=True)
    train_data_raw = None
    test_data_raw = None
    print("done\n")
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

    model_filters = 26
    teacher_model = tf.keras.Sequential([
        Input([32,32,3]),
        Resizing(224,224),
        tf.keras.models.load_model('trained_models/trainedEffNet_tuned.h5')
    ])
    new_model = keras_model.resnet_v1_eembc(conv_filters=model_filters)
    print('=== STUDENT MODEL ===')
    new_model.summary()
    print('\n=== TEACHER MODEL ===')
    teacher_model.summary()

    # batch data for the trip to the model
    # train_data_batched, val_data_batched, test_data_batched = \
    #     cifar10_batch(train_data, test_data, batch_size=BS)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(train_data)
    lr_scheduler = WarmUpCosine(learning_rate_base=0.01,
                                total_steps=(EPOCHS * 0.8 * len(train_data) / BS),
                                warmup_learning_rate=0.0,
                                warmup_steps=1500)
    TOTAL_STEPS = EPOCHS * int(len(train_data) * 0.8 / BS)
    print(f"Ending LR: {lr_scheduler(TOTAL_STEPS)}")
    optimizer = tfa.optimizers.AdamW(weight_decay=1e-5,
                                          learning_rate=lr_scheduler,
                                          clipnorm=1.0)
    distiller = keras_model.Distiller(student=new_model, teacher=teacher_model, batch_size=BS)
    distiller.compile(optimizer=optimizer,
                      metrics='accuracy',
                      distillation_loss_fn=kl_divergence,
                      temperature=2)
    new_model.compile(loss='categorical_crossentropy', metrics='accuracy',
                          loss_weights=None, weighted_metrics=None,
                          run_eagerly=None)
    teacher_model.compile(loss='categorical_crossentropy', metrics='accuracy',
                          loss_weights=None, weighted_metrics=None,
                          run_eagerly=None)
    test_metrics = teacher_model.evaluate(x=test_data, y=test_labels,
                                          batch_size=32, verbose=1,
                                          return_dict=True)

    # fits the model on batches with real-time data augmentation:
    History = distiller.fit(datagen.flow(train_data, train_labels, batch_size=BS, subset='training'),
                            steps_per_epoch=len(train_data) * 0.8 / BS, epochs=EPOCHS,
                            validation_data=datagen.flow(train_data, train_labels, batch_size=BS, subset='validation'),
                            validation_steps=len(train_data) * 0.2 / BS
                        )
    
    dt = datetime.datetime.today()
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    timestamp = f"{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}"
    model_name = f"trainedEffNet_distilled_{timestamp}.h5"
    new_model.save("trained_models/" + model_name)
    plt.clf() # clear figure in case the previous figure was not plotted
    plt.plot(np.array(range(EPOCHS)), History.history['distillation_loss'])
    plt.plot(np.array(range(EPOCHS)), History.history['val_distillation_loss'])
    plt.legend(labels=['Training', 'Validation'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f"trained_models/{model_name}_train_val_loss.png")
    plt.clf()
    # plt.plot(np.array(range(EPOCHS)), History.history['accuracy'])
    plt.plot(np.array(range(EPOCHS)), History.history['val_accuracy'])
    plt.legend(labels=['Training', 'Validation'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(f"trained_models/{model_name}_train_val_acc.png")

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
