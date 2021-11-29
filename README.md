# MLPerf™ Tiny Deep Learning Benchmarks for Embedded Devices

The goal of MLPerf Tiny is to provide a representative set of deep neural nets
and benchmarking code to compare performance between embedded devices. Embedded
devices include microcontrollers, DSPs, and tiny NN accelerators.  These devices
typically run at between 10MHz and 250MHz, and can perform inference using less
then 50mW of power.

MLPerf Tiny submissions will allow device makers and researchers to choose the
best hardware for their use case, and allows hardware and software vendors to showcase their
offerings.

The reference benchmarks are provided using TensorFlow Lite for Microcontrollers (TFLM).
Submitters can directly use the TFLM, although submitters are encouraged to use the software stack that works best on their hardware.

For more information, check out v0.5 of the benchmark.

Please see the [MLPerf Tiny Benchmark](https://arxiv.org/pdf/2106.07597.pdf) paper for a detailed description of the motivation and guiding principles behind the benchmark suite. If you use any part of this benchmark (e.g., reference implementations, submissions, etc.) in academic work, please cite the following:

```
@article{banbury2021mlperf,
  title={MLPerf Tiny Benchmark},
  author={Banbury, Colby and Reddi, Vijay Janapa and Torelli, Peter and Holleman, Jeremy and Jeffries, Nat and Kiraly, Csaba and Montino, Pietro and Kanter, David and Ahmed, Sebastian and Pau, Danilo and others},
  journal={Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
  year={2021}
}
```

Join the working group here: [https://groups.google.com/a/mlcommons.org/g/tiny](https://groups.google.com/u/1/a/mlcommons.org/g/tiny)
