Image Classification reference submission for MLCommons MLPerf Tiny benchmark

Quick Start
===========

Set up build environment with TensorFlow and Mbed-OS

    ./setup_example.sh

Compile using Mbed CLI tools

    mbed compile -m NUCLEO_L4R5ZI -t GCC_ARM

Deploy on Nucleo board

    cp ./BUILD/NUCLEO_L4R5ZI/GCC_ARM/image_classification.bin /Volumes/NODE_L4R5ZI/

### IC02 -- Larger ResNet
To compile the larger image classification model, convert the tflite file to C using `xxd` and 
replace the contents of `ic_model_quant_data.cc` with the resulting data.  Then in `ic/api/submitter_implemented.h` 
change the line

```
#define TH_MODEL_VERSION EE_MODEL_VERSION_IC01
```
to 
```
#define TH_MODEL_VERSION EE_MODEL_VERSION_IC02
```

