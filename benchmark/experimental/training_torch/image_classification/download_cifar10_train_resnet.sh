#!/bin/bash
. venv/bin/activate

# train ans test the model
python3 train.py
python3 test.py
