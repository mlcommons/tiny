#!/bin/bash
# Downoad the dataset.
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
unzip annotations_trainval2017.zip
unzip train2017.zip

# Preprocess the dataset and train/convert the VWW model.
python3 parse_coco.py annotations/instances_train2017.json
python3 train_vww_96.py
python3 convert_vww.py vww_96.h5
