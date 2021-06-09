Person detection reference submission for MLCommons MLPerf Tiny benchmark

Quick Start
===========

Set up build environment with TensorFlow and Mbed-OS

    ./setup_example.sh

Compile using Mbed CLI tools

    mbed compile -m NUCLEO_L4R5ZI -t GCC_ARM

Deploy on Nucleo board

    cp ./BUILD/NUCLEO_L4R5ZI/GCC_ARM/anomaly_detection.bin /Volumes/NODE_L4R5ZI/

