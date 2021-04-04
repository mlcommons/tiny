touch tfout.log
python train.py --saved_model_path=trained_models/kws_ref_model.h5 \
       --epochs=30 --run_test_set=True  2>>& tfout.log
python quantize.py --saved_model_path=trained_models/kws_ref_model.h5 \
       --tfl_file_name=trained_models/kws_ref_model.tflite 2>>& tfout.log
python eval_quantized_model.py --tfl_file_name=trained_models/kws_ref_model.tflite 2>>& tfout.log

