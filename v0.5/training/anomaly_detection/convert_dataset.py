"""
 @file   convert_dataset.py
 @brief  Script to convert WAV files to spectrograms
 @author Csaba Kiraly
"""

########################################################################
# import default python-library
########################################################################
import os
import sys
########################################################################


########################################################################
# import additional python-library
########################################################################
# from import
from tqdm import tqdm
# original lib
import common as com
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
#######################################################################


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

    # load base directory
    dirs = com.select_dirs(param=param, mode=mode)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]

        machine_id_list = com.get_machine_id_list_for_test(target_dir)

        for id_str in machine_id_list:
            # load test file
            test_files, y_true = com.test_file_list_generator(target_dir, id_str, mode)

            print("\n============== BEGIN CONVERSION FOR A MACHINE ID ==============")
            y_pred = [0. for k in test_files]
            for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
                try:
                    data = com.file_to_vector_array(file_path,
                                                    n_mels=param["feature"]["n_mels"],
                                                    frames=param["feature"]["frames"],
                                                    n_fft=param["feature"]["n_fft"],
                                                    hop_length=param["feature"]["hop_length"],
                                                    power=param["feature"]["power"],
                                                    save_bin=True)
                except Exception as e:
                    com.logger.error("file broken!!: {}, {}".format(file_path, e))

            print("\n============ END OF CONVERSION FOR A MACHINE ID ============")
