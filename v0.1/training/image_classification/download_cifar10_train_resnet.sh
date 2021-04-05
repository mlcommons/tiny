#!/bin/bash
# Downoad the dataset.
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz

# load performance subset
. venv/bin/activate
python3 perf_samples_loader.py

# train ans test the model
python3 train.py
python3 test.py
