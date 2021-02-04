# This benchmark suite is under construction. 

The TinyMLPerf benchmark suite consists four benchmarks which are found in [benchmarks](https://github.com/mlcommons/tiny/tree/master/v0.1/benchmarks).
Each benchmark has a dataset, model definition, and training script, which can be used as a reference in the open division.
Each benchmark also contains a trained_models folder which contains the fp32 and int8 quantized models used for the closed division.

All four benchmarks have [reference submission](https://github.com/mlcommons/tiny/tree/master/v0.1/reference_submissions) which implement 
the benchmarks using [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) and Mbed on our [reference platform](https://www.st.com/en/microcontrollers-microprocessors/stm32l4r5zi.html).

If you have any questions please email cbanbury@g.harvard.edu
