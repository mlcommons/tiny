# This benchmark suite is under construction. 

The TinyMLPerf benchmark suite consists four benchmarks:
|       Use Case       |                   Description                   |          Dataset          |       Model      |
|:--------------------:|:-----------------------------------------------:|:-------------------------:|:----------------:|
|   Audio Wake Words   |        Small vocabulary keyword spotting        |      Speech Commands      |      [DS-CNN](https://github.com/mlcommons/tiny/blob/master/v0.1/training/audio_wake_words/keras_model.py)      |
|   Visual Wake Words  |           Binary image classification           | Visual Wake Words Dataset |     [MobileNet](https://github.com/mlcommons/tiny/blob/master/v0.1/training/visual_wake_words/vww_model.py)    |
| Image Classification |            Small image classification           |          Cifar10          |      [ResNet](https://github.com/mlcommons/tiny/blob/master/v0.1/training/image_classification/keras_model.py)      |
|   Anomaly Detection  | Detecting anomalies in machine operating sounds |          ToyADMOS         | [Deep AutoEncoder](https://github.com/mlcommons/tiny/blob/master/v0.1/training/anomaly_detection/keras_model.py) |


There are two division:
* The closed division of the benchmark requires the use of a pre-trained model but allows submitters to use post training quantization and optimizations.
* The open division allows arbitrary selection of pre-processing and model architecture within an accuracy threshold. The open division allows for novel techniques, and a more diverse set of results at the cost of direct comparability.

The structure of the repository is as follows:
* The pre-trained models are found in [training](https://github.com/mlcommons/tiny/tree/master/v0.1/training) along with the dataset, training and quantization scripts.
* The benchmark API is defined in [API](https://github.com/mlcommons/tiny/tree/master/v0.1/api) which includes the required submitter implemented functions.
* All four benchmarks have a [reference submission](https://github.com/mlcommons/tiny/tree/master/v0.1/reference_submissions) which implement 
the benchmarks using [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) and Mbed on the [reference platform](https://www.st.com/en/microcontrollers-microprocessors/stm32l4r5zi.html).

We follow the [MLPerf Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc) unless there is a specific exception related to TinyMLPerf. All benchmarks are single stream only.

If you have any questions or issues with the benchmark please [create a new issue](https://github.com/mlcommons/tiny/issues).

## FAQ
**Q:** How do I submit?

**A:** TinyMLPerf set submission dates that align with MLPerf inference. To submit you must [join the working group](https://groups.google.com/u/4/a/mlcommons.org/g/tiny) and [register](https://forms.gle/GaB9Gc2MftothYpw7). The next submission deadline is March 12th 2021.
## 

**Q:** Is power measurement included in TinyMLPerf?

**A:** We are working with [EEMBC](https://www.eembc.org/) to include power benchmarking in the TinyMLPerf benchmark suite. Power benchmarking will be handeled by EEMBC and use their IoTConnect benchmark framework but will require very minimal modifcations from the standard TinyMLPerf latency benchmark
## 

**Q:** Is power measurement required to submit to TinyMLPerf?

**A:** No. Power measurement is not required to submit latency results to TinyMLPerf.
## 

**Q:** Will accuracy be measured?

**A:** Yes accuracy will be measure on the device under test. This will ensure the validity of results in the closed division. In the open division accuracy must remain within a threshold  of the reference model. The threshold is defined by then benchmark.
## 

**Q:** Do you require that submissions use TFLite for Micro/Mbed for the closed division?

**A:** No. The submitter can use the framework of their choice and implement any optimizations. The reference benchmarks use TFLite for Micro/Mbed for portability.
## 

**Q:** Can I submit with X device?

**A:** Longterm there are no explicit restrictions on what devices can be used for submission. The use cases and models have been selected to target IoT class devices like microcontrollers, DSPs and uNPUs therefore the benchmark is most relavent for these devices. In V0.1 of the benchmark, the LoadGen/runner assumes a Serial connection to transfer the inputs.





