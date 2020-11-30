#!/bin/sh

. venv/bin/activate
python 00_train.py --dev
python 01_test.py --dev
python 02_convert.py --dev
python 03_tflite_test.py --dev
