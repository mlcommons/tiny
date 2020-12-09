# tinyMLPerf Deep Learning Benchmarks for Embedded Devices

The goal of TinyMLPerf is to provide a representative set of deep neural nets
and benchmarking code to compare performance between embedded devices. Embedded
devices include microcontrollers, DSPs, and tiny NN accelerators.  These devices
typically run at between 10MHz and 250MHz, and can perform inference using less
then 50mW of power.

TinyMLPerf submissions will allow device makers and researchers to choose the
best hardware for their use case, and allows hardware vendors to showcase their
offerings.

TinyMLPerf is primarily intended to benchmark hardware rather than new network
architectures, or embedded neural net runtimes. The reference benchmarks are provided using
TensorFlow Lite for Microcontrollers (TFLM). Submitters can directly use the TFLM, although 
submitters are encouraged to use the software stack that works best on their hardware.
