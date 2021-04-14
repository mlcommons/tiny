## Steps to Build and Run

```
train.py
```
* Downloads the Google Speech Commands v2 data set to a directory (set by `--data_dir`, defaults to $HOME/data) after checking whether the data already exists there.  The data is structured as a TF dataset, not as individual wav files.
* If `--model_init_path` (can be either a saved_model directory or an h5 file) is set, an initial (partially pretrained) model is loaded as a starting point. Otherwise, a model is constructed by `keras_model.py:get_model()`.
* Trains the model for `--epochs=36` epochs using Keras `model.fit`.
* Training history is plotted in `--plot_dir=plots/`
* Model is saved to `--saved_model_path=trained_models/kws_model.h5`
* Evaluates the trained model for accuracy on the test set. This step can be disabled by setting `--run_test_set=False`.

```
quantize.py
```
* Loads a pre-trained model from `--saved_model_path`
* Uses the indices in `quant_cal_idxs.txt` to select calibration files
  from the validation set.  These files are passed to the TFLite
  converter using `representative_dataset_gen()` to quantize the
  weights and activations to 8b integers.
* Writes TFLite Micro model to `--tfl_file_name`.

```
eval_quantized_model.py
```
* Loads the TFLite model from `--tfl_file_name` and measures its
  accuracy on the test set.


```
./tflm2cc trained_models/kws_model.tflite
```
* Runs xxd on `kws_model.tflite` and makes a few other modifications to create `kws_model_data.cc`.


Move the newly created c file into the directory where you are building the mbed binary.
```
mv kws_model_data.cc ../../reference_submissions/keyword_spotting/kws/ 
```
