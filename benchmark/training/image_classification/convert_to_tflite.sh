#!/bin/sh

python3 model_converter.py
python3 tflite_test.py > Logs/tflite_testing_log.txt
