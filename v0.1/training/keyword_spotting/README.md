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


In the lines below, edit `TF_DIR` to indicate where you want to clone the TensorFlow github repo and set `LOCAL_ARCH` appropriately for your architecture.  The following commands will clone TF, change into the TF source tree, build the hello_world project for mbed and copy it back into your mbed build tree.  They will then create an mbed project in the `reference_submissions/keyword_spotting` directory and compile it.

```
TF_DIR=/path/to/source/of/tensorflow # replace with your path
git clone https://github.com/tensorflow/tensorflow.git ${TF_DIR}
LOCAL_ARCH=linux_x86_64w  # OR osx_x86_64  OR  osx_arm64_default
pushd $TF_DIR
gmake -f tensorflow/lite/micro/tools/make/Makefile generate_hello_world_mbed_project
popd
mv ${TF_DIR}/tensorflow/lite/micro/tools/make/gen/${LOCAL_ARCH}/prj/hello_world/mbed/* .
mbed config root .                                                                     
mbed deploy
cp -r  ../../api . 
cp ../../main.cpp .
cp -r ../../util . 
rm -rf tensorflow/lite/micro/examples/hello_world
cp -r ../../vww 
mbed compile --target NUCLEO_L4R5ZI --toolchain GCC_ARM -v
```


Copy the compiled executable `BUILD/NUCLEO_L4R5ZI/GCC_ARM/keyword_spotting.bin` onto the mbed board.  On Mac OS, this is `cp BUILD/NUCLEO_L4R5ZI/GCC_ARM/keyword_spotting.bin /Volumes/NODE_L4R5ZI`.  On Linux or Windows, you will use a different command.
