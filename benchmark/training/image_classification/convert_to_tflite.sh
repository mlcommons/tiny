#!/bin/sh

. venv/bin/activate
python3 model_converter.py
python3 tflite_test.py
