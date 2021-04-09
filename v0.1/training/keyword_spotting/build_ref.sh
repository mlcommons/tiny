# suppress informational messages from TF
export TF_CPP_MIN_LOG_LEVEL=2 

python train.py --saved_model_path=trained_models/kws_ref_model.h5 \
       --epochs=30 --run_test_set=True  
python quantize.py --saved_model_path=trained_models/kws_ref_model.h5 \
       --tfl_file_name=trained_models/kws_ref_model.tflite 
python eval_quantized_model.py --tfl_file_name=trained_models/kws_ref_model.tflite 

