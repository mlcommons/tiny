Anomaly detection reference submission for MLCommons MLPerf Tiny benchmark

Quick Start
===========

Set up build environment with TensorFlow and Mbed-OS

    ./setup_example.sh

Compile using Mbed CLI tools

    mbed compile -m NUCLEO_L4R5ZI -t GCC_ARM

Deploy on Nucleo board

    cp ./BUILD/NUCLEO_L4R5ZI/GCC_ARM/anomaly_detection.bin /Volumes/NODE_L4R5ZI/


NOTE: This reference code uses a performance-mode baud rate of 921600 instead of 115200, which requires changing the `~/.eembc.ini` file setting in order to use the DUT. Please see the benchmark runner GitHub readme for information on changing the runner's default baud rate.
The baud rate of 921600 is not working with all OSes, especially on macOS it seems to be problematic. If there are serial errors during the test, please change the baud rate to 115200, making sure it is changed on both sides, i.e., in `submitter_implemented.cpp` for the DUT, and in `~/.eembc.ini` for the runner.
