## In progress -- development of streaming wakeword benchmark

## Setup

1. Download and unpack the `speech_commands` dataset. I typically place this in a `~/dataset/` folder.
```
wget http://download.tensorflow.org/~/dataset/speech_commands/validation_list.txtdata/speech_commands_v0.02.tar.gz
mkdir -p ~/dataset/speech_commands/
tar -xzvf speech_commands_v0.02.tar.gz -C ~/dataset/speech_commands/
```
2. Setup a conda environment and install the required packages.
```
conda create -n tiny python=3.11 -y
conda activate tiny
python -m pip install -r requirements.txt
```
3. update the `streaming_config.json` file to point to the `speech_commands` folder. Note, it has to be the absolute path to the dataset.
```
{
  "speech_commands_path":"(/path to user)/dataset/speech_commands/"
}
```
