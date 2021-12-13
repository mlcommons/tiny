#!/bin/sh

. venv/bin/activate
python 00_train.py --dev
python 01_test.py --dev
