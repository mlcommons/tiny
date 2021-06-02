## Steps to Build and Run

### Train the Model
```
train.py
```
* Downloads the Google Speech Commands v2 data set to a directory (set by `--data_dir`, defaults to $HOME/data) after checking whether the data already exists there.  The data is structured as a TF dataset, not as individual wav files.
* If `--model_init_path` (can be either a saved_model directory or an h5 file) is set, an initial (partially pretrained) model is loaded as a starting point. Otherwise, a model is constructed by `keras_model.py:get_model()`.
* Trains the model for `--epochs=36` epochs using Keras `model.fit`.
* Training history is plotted in `--plot_dir=plots/`
* Model is saved to `--saved_model_path=trained_models/kws_model.h5`
* Evaluates the trained model for accuracy on the test set. This step can be disabled by setting `--run_test_set=False`.

### Quantize and Convert to TFLite
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

### Convert to C++ Code
```
./tflm2cc trained_models/kws_model.tflite
```
* Runs xxd on `kws_model.tflite` and makes a few other modifications to create `kws_model_data.cc`.

Move the newly created c file into the directory where you are building the mbed binary.
```
mv kws_model_data.cc ../../reference_submissions/keyword_spotting/kws/ 
```


### Deploy to the MCU Board
Change to the directory `../../reference_submissions/keyword_spotting/`

Now run the setup script.
```
./setup_example.sh
```

The script will clone TF, change into the TF source tree, build the hello_world project for mbed with the ARM CMSIS-NN kernels and copy it back into your mbed build tree.  It will then create an mbed project in the `reference_submissions/keyword_spotting` directory.


Now you can compile it with this command. Change the target for a different DUT.

```
mbed compile -m NUCLEO_L4R5ZI -t GCC_ARM
```

NOTE: You can install the Mbed CLI tools [here](https://os.mbed.com/docs/mbed-os/v5.15/tools/manual-installation.html)

Copy the compiled executable `BUILD/NUCLEO_L4R5ZI/GCC_ARM/keyword_spotting.bin` onto the mbed board.  On Mac OS, this is `cp BUILD/NUCLEO_L4R5ZI/GCC_ARM/keyword_spotting.bin /Volumes/NODE_L4R5ZI`.  On Linux or Windows, you will use a different command.


### Create Binary Test Files
Binary files for the benchmark can be downloaded from the [EEMBC github](https://github.com/eembc/benchmark-runner-ml/tree/main/datasets), but if you want to create your own, you can use the make_bin_files.py script here.  A pre-defined set of 1000 inputs is converted into features, quantized, and written out to a binary file.  The command should specify the destination for the files in the `bin_file_path` argument.  If features other than MFCCs are desired, that can be specified with `feature_type; 40-D log filterbank energies  (`--feature_type=lfbe`) and the original time-domain waveform (`--feature_type=td_samples`)are supported.  All three sets of bin files can be created with the following three lines.

```
python make_bin_files.py --bin_file_path=${HOME}/kws_bin_files/lfbe --feature_type=lfbe
python make_bin_files.py --bin_file_path=${HOME}/kws_bin_files/mfcc --feature_type=mfcc
python make_bin_files.py --bin_file_path=${HOME}/kws_bin_files/td_samples --feature_type=td_samples
```
The script will also create a file `y_labels.csv` in the same directory that indicates the correct output label for each file.  This file is used by the EEMBC runner.  Each line has the form:
```
<file name>, <number_of_classes>, <true_label>
tst_000002_Right_6.bin, 12, 6
```
If you edit the script and change this line
```
test_tfl_on_bin_files = False
```
to set to `True`, the script will also run the indicated TFLite model on each bin file, calculate accuracy, and create a second csv file `tflm_labels.csv` indicating how each file was classified.

The format of the files are summarized in the following table.

| Feature Type| Description                            | Dimension | Data Type | 
| ----------- | -------------------------------------- | --------- | --------- |
| mfcc        | Mel-Frequency Cepstral Coefficients    | 49 x 10   | INT8      |
| lfbe        | Mel-Scaled Log Filter-bank Energies    | 41x40     | UINT8     |
| td_samples  | Time-domain samples (raw waveform      |  16,000x1 | INT16     | 