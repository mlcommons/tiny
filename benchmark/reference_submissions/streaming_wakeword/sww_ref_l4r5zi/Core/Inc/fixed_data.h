/*
 * model_test_inputs.h
 *
 *  Created on: Jan 12, 2025
 *      Author: jeremy
 */

#ifndef MODEL_TEST_INPUTS_H_
#define MODEL_TEST_INPUTS_H_
#include <stdint.h>
#include "arm_math.h"

extern const int8_t test_input_class0[1200];
extern const int8_t test_input_class1[1200];
extern const int8_t test_input_class2[1200];

extern const float32_t sine_fs16k_7992[2048];
extern const float32_t sine_fs16k_200[2048];

extern const float32_t hamm_win_1024[1024];
extern const float32_t hamm_win_512[512];

extern const int16_t test_wav_marvin[16000];
extern const int16_t test_wav_backward[16000];

extern const float lin2mel_packed_513x40[987];
extern const int lin2mel_513x40_filter_starts[40];
extern const int lin2mel_513x40_filter_lens[40];

extern const int16_t test_wav_long[];
extern const unsigned int test_wav_long_len;

#endif /* MODEL_TEST_INPUTS_H_ */
