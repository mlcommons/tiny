'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

keras_model.py: CIFAR10_ResNetv1 from eembc
'''

import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Resizing
from tensorflow.keras.regularizers import l2

#get model
def get_model_name():
    if os.path.exists("trained_models/trainedResnet.h5"):
        return "trainedResnet"
    else:
        return "pretrainedResnet"

def get_quant_model_name():
    if os.path.exists("trained_models/trainedResnet.h5"):
        return "trainedResnet"
    else:
        return "pretrainedResnet"

#define models

# 200k params
def resnet_v1_eembc(conv_filters=26):
    # Resnet parameters
    input_shape=[32,32,3] # default size for cifar10
    num_classes=10 # default class number for cifar10
    num_filters = conv_filters # this should be 64 for an official resnet model

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x) # uncomment this for official resnet model


    # First stack

    # Weight layers
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)


    # Second stack

    # Weight layers
    num_filters = conv_filters * 2 # Filters need to be double for each stack
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)


    # Third stack

    # Weight layers
    num_filters = conv_filters * 4
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)


    # Fourth stack.
    # While the paper uses four stacks, for cifar10 that leads to a large increase in complexity for minor benefits
    # Uncomments to use it

#    # Weight layers
#    num_filters = conv_filters * 8
#    y = Conv2D(num_filters,
#                  kernel_size=3,
#                  strides=2,
#                  padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=l2(1e-4))(x)
#    y = BatchNormalization()(y)
#    y = Activation('relu')(y)
#    y = Conv2D(num_filters,
#                  kernel_size=3,
#                  strides=1,
#                  padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=l2(1e-4))(y)
#    y = BatchNormalization()(y)
#
#    # Adjust for change in dimension due to stride in identity
#    x = Conv2D(num_filters,
#                  kernel_size=1,
#                  strides=2,
#                  padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=l2(1e-4))(x)
#
#    # Overall residual, connect weight layer and identity paths
#    x = tf.keras.layers.add([x, y])
#    x = Activation('relu')(x)


    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

# EffectiveNet V2S
def effnet_v2s(transfer=True):
    # EffNet parameters
    input_shape=[224,224,3] # default size for cifar10
    num_classes=10 # default class number for cifar10

    # Input layer
    inputs = Input(shape=input_shape)
    # x = Resizing(224, 224)(inputs)
    effnet = tf.keras.applications.EfficientNetV2S(include_top=False, weights="imagenet", include_preprocessing=True)
    if transfer:
        for layer in effnet.layers:
            layer.trainable = False
    x = effnet(inputs)

    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

# class Distiller(tf.keras.Model):
#     def __init__(self, student, teacher):
#         super().__init__()
#         self.teacher = teacher
#         self.student = student

#     def compile(
#         self,
#         optimizer,
#         metrics,
#         student_loss_fn,
#         distillation_loss_fn,
#         alpha=0.1,
#         temperature=3,
#     ):
#         """Configure the distiller.

#         Args:
#             optimizer: Keras optimizer for the student weights
#             metrics: Keras metrics for evaluation
#             student_loss_fn: Loss function of difference between student
#                 predictions and ground-truth
#             distillation_loss_fn: Loss function of difference between soft
#                 student predictions and soft teacher predictions
#             alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
#             temperature: Temperature for softening probability distributions.
#                 Larger temperature gives softer distributions.
#         """
#         super().compile(optimizer=optimizer, metrics=metrics)
#         self.student_loss_fn = student_loss_fn
#         self.distillation_loss_fn = distillation_loss_fn
#         self.alpha = alpha
#         self.temperature = temperature

#     def compute_loss(
#         self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
#     ):
#         teacher_pred = self.teacher(x, training=False)
#         student_loss = self.student_loss_fn(y, y_pred)

#         distillation_loss = self.distillation_loss_fn(
#             tf.nn.softmax(teacher_pred / self.temperature, axis=1),
#             tf.nn.softmax(y_pred / self.temperature, axis=1),
#         ) * (self.temperature**2)

#         loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
#         return loss

#     def call(self, x):
#         return self.student(x)

# Reference:
# https://keras.io/examples/vision/knowledge_distillation/
class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, batch_size):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.batch_size = batch_size

    def compile(
        self,
        optimizer,
        metrics,
        distillation_loss_fn,
        temperature=2,
    ):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, _ = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute loss
            distillation_loss = self.distillation_loss_fn(
                teacher_predictions / self.temperature,
                student_predictions / self.temperature
            )
            distillation_loss = tf.nn.compute_average_loss(distillation_loss, 
                                                      global_batch_size=self.batch_size)

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(distillation_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Report progress
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)
        student_predictions = self.student(x, training=False)

        # Calculate the loss
        distillation_loss = self.distillation_loss_fn(
            teacher_predictions / self.temperature,
            student_predictions / self.temperature
        )
        distillation_loss = tf.nn.compute_average_loss(distillation_loss, 
                                                      global_batch_size=self.batch_size)

        # Report progress
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"distillation_loss": distillation_loss}
        )
        return results
