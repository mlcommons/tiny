"""
 @file   03_tflite_test.py
 @brief  Script for testing tflite performance
 @author Csaba Kiraly (based on 01_test.py)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
# from import
from tqdm import tqdm
from sklearn import metrics
# original lib
import common as com
import tensorflow as tf
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
#######################################################################


########################################################################
# def
########################################################################
def predict(interpreter, data):

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on input data.
    input_shape = input_details[0]['shape']

    input_data = numpy.array(data, dtype=numpy.float32)
    output_data = numpy.empty_like(data)

    for i in range(input_data.shape[0]):
        interpreter.set_tensor(input_details[0]['index'], input_data[i:i+1, :])
        interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data[i:i+1, :] = interpreter.get_tensor(output_details[0]['index'])
    return output_data

########################################################################
# main 01_test.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # make output result directory
    os.makedirs(param["result_directory"], exist_ok=True)

    # load base directory
    dirs = com.select_dirs(param=param, mode=mode)

    filename_modifiers = ["",
                          "_quant",
                          "_quant_fullint",
                          "_quant_fullint_micro"]

    for model_postfix in filename_modifiers:

        # initialize lines in csv for AUC and pAUC
        csv_lines = []

        # loop of the base directory
        for idx, target_dir in enumerate(dirs):
            print("\n===========================")
            print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))
            machine_type = os.path.split(target_dir)[1]

            print("============== MODEL LOAD ==============")
            # set model path
            tflite_file = "{model}/model_{machine_type}{model_postfix}.tflite".format(model=param["model_directory"],
                                                                               machine_type=machine_type,
                                                                               model_postfix=model_postfix)

            # load model file
            if not os.path.exists(tflite_file):
                com.logger.error("{} model not found: {} ".format(machine_type, tflite_file))
                sys.exit(-1)

            interpreter = tf.lite.Interpreter(model_path=tflite_file)
            interpreter.allocate_tensors()

            if mode:
                # results by type
                csv_lines.append([machine_type])
                csv_lines.append(["id", "AUC", "pAUC"])
                performance = []

            machine_id_list = com.get_machine_id_list_for_test(target_dir)

            for id_str in machine_id_list:
                # load test file
                test_files, y_true = com.test_file_list_generator(target_dir, id_str, mode)

                # setup anomaly score file path
                anomaly_score_csv = "{result}/tflite{model_postfix}_anomaly_score_{machine_type}_{id_str}.csv".format(
                                                                                         result=param["result_directory"],
                                                                                         model_postfix=model_postfix,
                                                                                         machine_type=machine_type,
                                                                                         id_str=id_str)
                anomaly_score_list = []

                print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
                y_pred = [0. for k in test_files]
                for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
                    try:
                        data = com.file_to_vector_array(file_path,
                                                        n_mels=param["feature"]["n_mels"],
                                                        frames=param["feature"]["frames"],
                                                        n_fft=param["feature"]["n_fft"],
                                                        hop_length=param["feature"]["hop_length"],
                                                        power=param["feature"]["power"])
                        errors = numpy.mean(numpy.square(data - predict(interpreter, data)), axis=1)
                        y_pred[file_idx] = numpy.mean(errors)
                        anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                    except:
                        com.logger.error("file broken!!: {}".format(file_path))

                # save anomaly score
                com.save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
                com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

                if mode:
                    # append AUC and pAUC to lists
                    auc = metrics.roc_auc_score(y_true, y_pred)
                    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
                    csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
                    performance.append([auc, p_auc])
                    com.logger.info("AUC : {}".format(auc))
                    com.logger.info("pAUC : {}".format(p_auc))

                print("\n============ END OF TEST FOR A MACHINE ID ============")

            if mode:
                # calculate averages for AUCs and pAUCs
                averaged_performance = numpy.mean(numpy.array(performance, dtype=float), axis=0)
                csv_lines.append(["Average"] + list(averaged_performance))
                csv_lines.append([])

        if mode:
            # output results
            result_path = "{result}/tflite{model_postfix}_{file_name}".format(
                result=param["result_directory"],
                model_postfix=model_postfix,
                file_name=param["result_file"])
            com.logger.info("AUC and pAUC results -> {}".format(result_path))
            com.save_csv(save_file_path=result_path, save_data=csv_lines)
