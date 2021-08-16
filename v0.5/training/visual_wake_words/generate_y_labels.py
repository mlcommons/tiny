# Generates y_labels (ground truth class labels)

# This script must be run AFTER the raw images have been processed and saved as .bin

# To save as .bin, use a script like:
# https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/Person_detection/buildPersonDetectionDatabase.py
# but save as .bin instead of .jpg

import os

GROUND_TRUTH_FILENAME='y_labels.csv'
PERSON_DIR = 'vw_coco2014_96_bin/person'
NON_PERSON_DIR = 'vw_coco2014_96_bin/non_person'

def generate_file_contents():
  outfile = open(GROUND_TRUTH_FILENAME, "a")

  for file in os.listdir(PERSON_DIR):
    input_filename = os.fsdecode(file)
    line = '%s, %d, %d\n'%(input_filename, 2, 1)
    outfile.write(line)

  for file in os.listdir(NON_PERSON_DIR):
    input_filename = os.fsdecode(file)
    line = '%s, %d, %d\n'%(input_filename, 2, 0)
    outfile.write(line)

  outfile.flush()
  outfile.close()

generate_file_contents()
