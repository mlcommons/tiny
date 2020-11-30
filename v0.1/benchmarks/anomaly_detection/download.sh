#!/bin/sh

URL="https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip?download=1"
ZIPFILE="dev_data_ToyCar.zip"

curl $URL -o $ZIPFILE || wget $URL -O $ZIPFILE
mkdir -p dev_data
unzip $ZIPFILE -d dev_data
