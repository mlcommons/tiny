# TinyMLPerf anomaly detection reference model

This is the TinyMLPerf anomaly reference model, based on the baseline system for the
DCASE 2020 Challenge Task 2 "Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring". 

The description of the original challenge is available at:
http://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds

The original code was extended with TFLite conversion and verification.

Besides the training scripts, the repository also contains the pre-trained version of the reference model
and its converted versions.

## Quick start

Run the following commands to go through the whole training and validation process

``` Bash
# Prepare Python venv (Python 3.6+ required)
./prepare.sh

# Download training data from Zenodo
./download.sh

# Train, convert, and test the model
./train_and_convert.sh
```

## Description
The baseline system consists of four main scripts:
- `00_train.py`
  - This script trains models for each Machine Type by using the directory **dev_data/<Machine_Type>/train/** or **eval_data/<Machine_Type>/train/**.
- `01_test.py`
  - This script makes csv files for each Machine ID including the anomaly scores for each wav file in the directory **dev_data/<Machine_Type>/test/** or **eval_data/<Machine_Type>/test/**.
  - The csv files will be stored in the directory **result/**.
  - If the mode is "development", it also makes the csv files including the AUC and pAUC for each Machine ID. 
- `02_convert.py`
  - This script converts the previously generated models to TFLite and quantized TFLite models. For the quantization ,it uses the data in **dev_data/<Machine_Type>/train/**.
- `03_tflite_test.py`
  - This script makes csv files for each Machine ID and each TFLite model version, similar to `01_test.sh`

## Detailed Usage

Either use the scripts described in **Quick Start** instructions, or follow the steps below.  

### 1. Clone repository
Clone this repository from Github.

### 2. Download datasets
We will launch the datasets in three stages. 
So, please download the datasets in each stage:
- Development dataset
  - Download `dev_data_<Machine_Type>.zip` from https://zenodo.org/record/3678171.
- "Additional training dataset", i.e. the evaluation dataset for training
  - After launch, download `eval_data_train_<Machine_Type>.zip` from https://zenodo.org/record/3727685 (not available until April. 1).
- "Evaluation dataset", i.e. the evaluation for test
  - After launch, download `eval_data_test_<Machine_Type>.zip` from https://zenodo.org/record/3841772 (not available until June. 1).

### 3. Unzip dataset
Unzip the downloaded files and make the directory structure as follows:
- ./dcase2020_task2_baseline
    - /dev_data
        - /ToyCar
            - /train (Only normal data for all Machine IDs are included.)
                - /normal_id_01_00000000.wav
                - ...
                - /normal_id_01_00000999.wav
                - /normal_id_02_00000000.wav
                - ...
                - /normal_id_04_00000999.wav
            - /test (Normal and anomaly data for all Machine IDs are included.)
                - /normal_id_01_00000000.wav
                - ...
                - /normal_id_01_00000349.wav
                - /anomaly_id_01_00000000.wav
                - ...
                - /anomaly_id_01_00000263.wav
                - /normal_id_02_00000000.wav
                - ...
                - /anomaly_id_04_00000264.wav
        - /ToyConveyor (The other Machine Types have the same directory structure as ToyCar.)
        - /fan
        - /pump
        - /slider
        - /valve
    - /eval_data (Add this directory after launch)
        - /ToyCar
            - /train (Unzipped "additional training dataset". Only normal data for all Machine IDs are included.)
                - /normal_id_05_00000000.wav
                - ...
                - /normal_id_05_00000999.wav
                - /normal_id_06_00000000.wav
                - ...
                - /normal_id_07_00000999.wav
            - /test (Unzipped "evaluation dataset". Normal and anomaly data for all Machine IDs are included, but there is no label about normal or anomaly.)
                - /id_05_00000000.wav
                - ...
                - /id_05_00000514.wav
                - /id_06_00000000.wav
                - ...
                - /id_07_00000514.wav
        - /ToyConveyor (The other machine types have the same directory structure as ToyCar.)
        - /fan
        - /pump
        - /slider
        - /valve
    - /00_train.py
    - /01_test.py
    - /common.py
    - /keras_model.py
    - /baseline.yaml
    - /readme.md

### 4. Change parameters
You can change the parameters for feature extraction and model definition by editing `baseline.yaml`.

### 5. Run training script (for development dataset)
Run the training script `00_train.py`. 
Use the option `-d` for the development dataset **dev_data/<Machine_Type>/train/**.
```
$ python3.6 00_train.py -d
```
Options:

| Argument                    |                                   | Description                                                  | 
| --------------------------- | --------------------------------- | ------------------------------------------------------------ | 
| `-h`                        | `--help`                          | Application help.                                            | 
| `-v`                        | `--version`                       | Show application version.                                    | 
| `-d`                        | `--dev`                           | Mode for "development"                                       |  
| `-e`                        | `--eval`                          | Mode for "evaluation"                                        | 

`00_train.py` trains the models for each Machine Type and saves the trained models in the directory **model/**.

### 6. Run test script (for development dataset)
Run the test script `01_test.py`.
Use the option `-d` for the development dataset **dev_data/<Machine_Type>/test/**.
```
$ python3.6 01_test.py -d
```
The options for `01_test.py` are the same as those for `00_train.py`.
`01_test.py` calculates the anomaly scores for each wav file in the directory **dev_data/<Machine_Type>/test/**.
The csv files for each Machine ID including the anomaly scores will be stored in the directory **result/**.
If the mode is "development", the script also makes the csv files including the AUCs and pAUCs for each Machine ID. 

### 7. Check results
You can check the anomaly scores in the csv files in the directory **result/**.
Each anomaly score corresponds to a wav file in the directory **dev_data/<Machine_Type>/test/**:

`anomaly_score_ToyCar_id_01.csv`
```  
normal_id_01_00000000.wav	6.95342025
normal_id_01_00000001.wav	6.363580014
normal_id_01_00000002.wav	7.048401741
normal_id_01_00000003.wav	6.151557502
normal_id_01_00000004.wav	6.450118248
normal_id_01_00000005.wav	6.368985477
  ...
```

Also, you can check the AUC and pAUC scores for each Machine ID:

`result.csv`
```  
ToyCar
id		AUC		pAUC
1		0.828474	0.678514
2		0.866307	0.762236
3		0.639396	0.546659
4		0.869477	0.701461
Average		0.800914	0.672218

ToyConveyor	
id		AUC		pAUC
1		0.793497	0.650428
2		0.64243		0.564437
3		0.744579	0.60488
Average		0.726835	0.606582

fan	
id		AUC		pAUC
0		0.539607	0.492435
2		0.721866	0.554171
4		0.622098	0.527072
6		0.72277		0.529961
Average		0.651585	0.52591

pump	
id		AUC		pAUC
0		0.670769	0.57269
2		0.609369	0.58037
4		0.8886		0.676842
6		0.734902	0.570175
Average		0.72591		0.600019

slider	
id		AUC		pAUC
0		0.963315	0.818451
2		0.78809		0.628819
4		0.946011	0.739503
6		0.702247	0.490242
Average		0.849916	0.669254

valve	
id		AUC		pAUC
0		0.687311	0.51349
2		0.674083	0.512281
4		0.744		0.515351
6		0.552833	0.482895
Average		0.664557	0.506004
```

### 8. Run training script for "additional training dataset" (after April 1)
After the "additional training dataset" is launched, download and unzip it.
Move it to **eval_data/<Machine_Type>/train/**.
Run the training script `00_train.py` with the option `-e`. 
```
$ python3.6 00_train.py -e
```
Models are trained by using the "additional training dataset" **eval_data/<Machine_Type>/train/**.

### 9. Run test script for "evaluation dataset" (after June 1)
After the "evaluation dataset" for test is launched, download and unzip it.
Move it to **eval_data/<Machine_Type>/test/**.
Run the test script `01_test.py` with the option `-e`. 
```
$ python3.6 01_test.py -e
```
Anomaly scores are calculated using the "evaluation dataset" **eval_data/<Machine_Type>/test/**.
The anomaly scores are stored as csv files in the directory **result/**.
You can submit the csv files for the challenge.
From the submitted csv files, we will calculate the AUCs, pAUCs, and your ranking.

## Dependency
The original source code was developed on Ubuntu 16.04 LTS and 18.04 LTS. The TinyMLPerf extension was developed on **OS X**.
In addition, we checked performing on **Ubuntu 16.04 LTS**, **18.04 LTS**, **Cent OS 7**, and **Windows 10**.

### Software packages
- p7zip-full, or other zip tool
- Python == 3.6.5, or newer
- FFmpeg

### Python packages

For an up to date list, please refer to `requirements.txt`

