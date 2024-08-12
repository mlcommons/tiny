## In progress -- development of streaming wakeword benchmark

## Setup

1. Make sure you have enough disk space.  Speech commands and MUSAN together take about 16GB of space, but you'll need an additional 12GB while you untar MUSAN.

2. Download and unpack the `speech_commands` dataset.  You may already have it for the keyword-spotting benchmark. I typically place this under a `~/data/` folder, but you can put it wherever you like, as long as you edit `streaming_config.json` accordingly (below), and replace `~/data/` with the correct path in the commands below.
```
cd ~/data/
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir speech_commands_v0.02
cd speech_commands_v0.02
tar -xzvf speech_commands_v0.02.tar.gz
```

2. Download and unpack the (MUSAN)[https://www.openslr.org/17/] noise dataset. 
```
cd ~/data/
wget https://openslr.elda.org/resources/17/musan.tar.gz
tar -xzvf musan.tar.gz
```

3. Setup a conda environment and install the required packages.
```
conda create -n tiny python=3.11 -y
conda activate tiny
python -m pip install -r requirements.txt
```

4. Copy `streaming_config_template.json` to `streaming_config.json` and edit it to match the paths where you saved the speech commands and musan datasets.
```
cp streaming_config_template.json streaming_config.json
```
Edit `streaming_config.json` to point to the paths where you have the speech commands and MUSAN datasets.
```
{
  "speech_commands_path":"/path/to/data/speech_commands_v0.02/",
  "musan_path":"/path/to/data/musan"
}
```

## Evaluation
To evaluate the pretrained model on the reference model (or at least whatever is in `trained_models/str_ww_ref_model.h5`), run
```
python evaluate.py --saved_model_path=trained_models/str_ww_ref_model.h5
```
The argument `saved_model_path` is required; there is no default. If you want to evaluate another model, replace the argument to the `saved_model_path` flag.

On the reference model, you should see something like this:

```
Input shape = [None, 1, 40]
Long waveform shape = (19200000,), spectrogram shape = (37499, 1, 40)
Building dataset with 2796 targets, 1398 silent, and 9786 other.
time shift should range from -1600 to 1600
280/280 [==============================] - 20s 67ms/step - loss: 0.1984 - categorical_accuracy: 0.9607 - precision: 0.9948 - recall: 0.7500    
Results: false_detections=4, true_detections=40, false_rejections=10,val_loss=0.1984, val_acc=0.9607, val_precision=0.9948, val_recall=0.7500
```

## Quantization
To quantize and convert a trained model into a TFlite model, run this line.
```
python quantize.py --model_init_path=my_model.h5
```
With no `model_init_path`, it will default to `trained_models/str_ww_ref_model.h5`
