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

For the current version of the benchmark under development, please see the [benchmark folder](https://github.com/mlcommons/tiny/tree/master/benchmark).

The **deadline** of the next submission round v1.1 is expected to be May 19, 2023, with publication in June (dates not yet finalized).

Previous versions are frozen using git tags as follows:

| Version | Code                                        | Release Date | Results                                     |
|---------|---------------------------------------------|--------------|---------------------------------------------|
| v0.5    | https://github.com/mlcommons/tiny/tree/v0.5 | Jun 16, 2021 | https://mlcommons.org/en/inference-tiny-05/ |
| v0.7    | https://github.com/mlcommons/tiny/tree/v0.7 |April  6, 2022| https://mlcommons.org/en/inference-tiny-07/ |
| v1.0    | https://github.com/mlcommons/tiny/tree/v1.0 | Nov 9, 2022  | https://mlcommons.org/en/inference-tiny-10/ |
|         |                                             |              |                                             |


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
