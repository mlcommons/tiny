# Trained models and TFLite converted versions

This folder contains pre-trained models for TinyMLPerf using the ToyCar dataset, including also 
TFLite converted versions, Post-training quantization (PTQ) versions and validation logs.

Several models are included under the `ToyCar/baseline_tf23/model` folder, as generated during the
training and by the conversions scripts. However, only two are used for TinyMLPerf:
- `model_ToyCar.hdf5`: the reference model before quantization
- `model_ToyCar_quant_fullint_micro_intio.tflite`: the quantized model with INT8 input/output

The latter can be converted into a C source file using
```
xxd -i model_ToyCar_quant_fullint_micro_intio.tflite > model.ccx
```

Other versions present in the folder:
- `model_ToyCar.tflite`: simple TFLite conversion, without quantization
- `model_ToyCar_quant.tflite`: PTQ of weights, but not the activations
- `model_ToyCar_quant_fullint.tflite`: PTQ of both weight and activations using a representative dataset
- `model_ToyCar_quant_fullint_micro.tflite`: As previous, enforcing TFLite supported OPs

Finally, the `result` folder contains the verification results for each model.
