The current version of the MLPerf Tiny benchmark suite consists four benchmarks:
|       Use Case       |                   Description                   |          Dataset          |       Model      | Quality Target<br>(Closed&#160;Division)
|:--------------------:|:-----------------------------------------------:|:-------------------------:|:----------------:|:-------------------:|
|   Keyword Spotting   |        Small vocabulary keyword spotting        |      Speech Commands [[1]](#1)                |      [DS-CNN](https://github.com/mlcommons/tiny/blob/master/benchmark/training/keyword_spotting/keras_model.py)   | 90% (Top 1)
|   Visual Wake Words  |           Binary image classification           | Visual Wake Words Dataset [[2]](#2)           |     [MobileNet](https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual_wake_words/vww_model.py)    | 80% (Top 1)
| Image Classification |            Small image classification           |          Cifar10 [[3]](#3)                    |      [ResNet](https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/keras_model.py)      | 85% (Top 1)
|   Anomaly Detection  | Detecting anomalies in machine operating sounds |          ToyADMOS [[4]](#4)[[5]](#5)[[6]](#6) | [Deep AutoEncoder](https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/keras_model.py) | 0.85 (AUC)

## Divisions

There are two division:
* The closed division of the benchmark requires the use of a pre-trained model but allows submitters to use post training quantization and optimizations.
* The open division allows arbitrary selection of training scheme, pre-processing, and model architecture. The open division allows for novel techniques, and a more diverse set of results at the cost of direct comparability.

The structure of the repository is as follows:
* The pre-trained models are found in [training](https://github.com/mlcommons/tiny/tree/master/benchmark/training) along with the dataset, training and quantization scripts.
* The benchmark API is defined in [API](https://github.com/mlcommons/tiny/tree/master/benchmark/api) which includes the required submitter implemented functions.
* All four benchmarks have a [reference submission](https://github.com/mlcommons/tiny/tree/master/benchmark/reference_submissions) which implement 
the benchmarks using [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) and Mbed on the [reference platform](https://www.st.com/en/microcontrollers-microprocessors/stm32l4r5zi.html).

We use the [EEMBCs EnergyRunner™ benchmark framework](https://github.com/eembc/ulpmark-ml) to connect to the system under test and run the benchmarks.

We follow the [MLPerf Tiny Rules](https://github.com/mlcommons/tiny/blob/master/benchmark/MLPerfTiny_Rules.adoc). All MLPerf Tiny benchmarks are single stream only and we do not support the retraining subdivision.

We additionally follow the MLPerf [submission and run rules](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc).

If you have any questions or issues with the benchmark please check the below FAQ first, and if still unanswered [create a new issue](https://github.com/mlcommons/tiny/issues).

## FAQ
**Q:** How do I submit?

**A:** MLPerf Tiny has set submission dates that align with MLPerf inference. To submit you must [join the working group](https://groups.google.com/u/4/a/mlcommons.org/g/tiny) and [register](https://forms.gle/apopJLhUcYDtmAyw5). The next submission deadline is February 25th 2022.
## 

**Q:** Is power measurement included in MLPerf Tiny?

**A:** We are working with [EEMBC](https://www.eembc.org/) to include power benchmarking in the MLPerf Tiny benchmark suite. Power benchmarking will use [EEMBCs EnergyRunner™ benchmark framework](https://github.com/eembc/energyrunner) but will require very minimal modifcations from the standard MLPerf Tiny latency benchmark.
## 

**Q:** Is power measurement required to submit to MLPerf Tiny?

**A:** No. Power measurement is encouraged but not required to submit latency results to MLPerf Tiny.
## 

**Q:** Will accuracy be measured?

**A:** Yes accuracy will be measured on the device under test. This will ensure the validity of results in the closed division. In the closed division the accuracy must remain within a threshold of the reference model. The threshold is defined by the benchmark.

In the open division the accuracy is measured and reported but does not have to hit the threshold.
## 

**Q:** Do you require that submissions use TFLite for Micro/Mbed for the closed division?

**A:** No. The submitter can use the framework of their choice and implement any post training optimizations including post training quantization. The reference benchmarks use TFLite for Micro/Mbed for portability.
## 

**Q:** Can I submit with X device?

**A:** Longterm there are no explicit restrictions on what devices can be used for submission. The use cases and models have been selected to target IoT class devices like microcontrollers, DSPs and uNPUs therefore the benchmark is most relavent for these devices. In v0.5 and v0.7 of the benchmark, the LoadGen/runner assumes a Serial connection to transfer the inputs.

## References
<a id="1">[1]</a>
Warden, Pete. "Speech commands: A dataset for limited-vocabulary speech recognition." arXiv preprint arXiv:1804.03209 (2018).

<a id="2">[2]</a>
Chowdhery, Aakanksha, Pete Warden, Jonathon Shlens, Andrew Howard, and Rocky Rhodes. "Visual wake words dataset." arXiv preprint arXiv:1906.05721 (2019).

<a id="3">[3]</a>
Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of features from tiny images." (2009).

<a id="4">[4]</a>
Yuma Koizumi, Shoichiro Saito, Noboru Harada, Hisashi Uematsu, and Keisuke Imoto, "ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection," in Proc of Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2019.

<a id="5">[5]</a>
Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” in Proc. 4th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2019.

<a id="6">[6]</a>
Yuma Koizumi, Yohei Kawaguchi, Keisuke Imoto, Toshiki Nakamura, Yuki Nikaido, Ryo Tanabe, Harsh Purohit, Kaori Suefusa, Takashi Endo, Masahiro Yasuda, and Noboru Harada, "Description and Discussion on DCASE2020 Challenge Task2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring," in arXiv e-prints: 2006.05822, 2020.
