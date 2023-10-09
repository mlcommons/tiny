# MLPerf Tiny image classification PyTorch model

This is the MLPerf Tiny image classification PyTorch model.

A ResNet8 model is trained on the CIFAR10 dataset available at:
https://www.cs.toronto.edu/~kriz/cifar.html

Model: ResNet8
Dataset: Cifar10

## Quick start

Run the following commands to go through the whole training and validation process

```Bash
# Prepare Python venv (Python 3.7+ and pip>20 required)
./prepare_training_env.sh

# Download training, train model, test the model
./download_cifar10_train_resnet.sh
```
