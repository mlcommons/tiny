# Trained models and TFLite converted versions

This folder contains pre-trained models for MLPerf Tiny using the ToyCar dataset, including also 
TFLite converted versions, Post-Training Quantization (PTQ) versions and validation logs.

The golden models for the anomaly detection `ad01` benchmarks are included in the current folder.
- `ad01.h5`: the reference model before quantization
- `ad01_fp32`: simple TFLite conversion of the above, without quantization, but with BatchNormalizations folded into FC layers
- `ad01_int8`: the quantized model with INT8 input/output

The latter can be converted into a C source file using
```
xxd -i model_ToyCar_quant_fullint_micro_intio.tflite > model.ccx
```

Models (the same as above and more) are also included under the `ToyCar/baseline_tf23/model` folder, as 
generated during the training and by the conversions scripts. These include:
- `model_ToyCar.hdf5`: the reference model before quantization
- `model_ToyCar.tflite`: simple TFLite conversion, without quantization, but with BatchNormalizations folded into FC layers
- `model_ToyCar_quant.tflite`: PTQ of weights, but not the activations
- `model_ToyCar_quant_fullint.tflite`: PTQ of both weight and activations using a representative dataset
- `model_ToyCar_quant_fullint_micro.tflite`: As previous, enforcing TFLite supported OPs
- `model_ToyCar_quant_fullint_micro_intio.tflite`: the quantized model with INT8 input/output

Finally, the `anomaly_detection/trained_model/result` folder contains the verification results for each model.
