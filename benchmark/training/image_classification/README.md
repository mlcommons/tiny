# MLPerf Tiny image classification reference model

This is the MLPerf Tiny image classification reference model.

A ResNet8 model is trained on the CIFAR10 dataset available at:
https://www.cs.toronto.edu/~kriz/cifar.html

Model: ResNet8
Dataset: Cifar10

## Quick start

Run the following commands to go through the whole training and validation process

Recommend creating one virtual environment before the experiment

``` Bash
# Prepare Python venv (Python 3.9+ and pip>20 required)
./prepare_training_env.sh

# Download dataset
./download_cifar10.sh

# Load the performance subset
./load_performance_subset.sh

# Train and test the model
./train_test_model.sh

# Convert the model to TFlite, and test conversion quality
./convert_to_tflite.sh
```

## Description
The python format CIFAR10 dataset batches are stored in the __/cifar-10-batches-py__ folder.

The performance evaluation test set is made of 200 samples randomly chosen from the CIFAR10.
These samples are stored in the __/perf_samples__ folder.  
Associated to the performance samples is the __y_labels.csv__ ground truth file: the first item of each row is the name of the sample, the second item of each row is the total number of classes (10) and the third item is the target class.

Please, use this performance evaluation test set to evaluate performances of the system to be tested.

