#!/bin/sh

URL1="https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip?download=1"
ZIPFILE="dev_data_ToyCar.zip"

URL2="https://zenodo.org/record/3727685/files/eval_data_train_ToyCar.zip?download=1"

mkdir -p dev_data

curl $URL1 -o $ZIPFILE || wget $URL1 -O $ZIPFILE
unzip $ZIPFILE -d dev_data
rm $ZIPFILE

curl $URL2 -o $ZIPFILE || wget $URL2 -O $ZIPFILE
unzip $ZIPFILE -d dev_data
rm $ZIPFILE
