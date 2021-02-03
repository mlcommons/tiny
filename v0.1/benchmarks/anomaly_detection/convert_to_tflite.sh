#!/bin/sh

. venv/bin/activate
python 02_convert.py --dev
python 03_tflite_test.py --dev
