## In progress -- development of streaming wakeword benchmark

## Setup

1. Download and unpack the `speech_commands` dataset. I typically place this under a `~/data/` folder, but you can put it wherever you like, as long as you edit `streaming_config.json` accordingly (below).
```
cd ~/data/
wget http://download.tensorflow.org/~/dataset/speech_commands/validation_list.txtdata/speech_commands_v0.02.tar.gz
tar -xzvf speech_commands_v0.02.tar.gz
```

2. Download and unpack the (MUSAN)[https://www.openslr.org/17/] noise dataset. 
```
cd ~/data/
wget https://www.openslr.org/resources/17/
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
python evaluate.py
```
If you want to evaluate another model, use the `model_init_path` flag.
```
python evaluate.py --model_init_path=my_model.h5
```

## Quantization
To quantize and convert a trained model into a TFlite model, run this line.
```
python quantize.py --model_init_path=my_model.h5
```
With no `model_init_path`, it will default to `trained_models/str_ww_ref_model.h5`
