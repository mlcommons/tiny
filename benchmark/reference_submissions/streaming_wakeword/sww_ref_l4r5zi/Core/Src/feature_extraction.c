/*
 * feature_extraction.c
 *
 *  Created on: Jan 12, 2025
 *      Author: jeremy
 */


/* ----------------------------------------------------------------------
* Copyright (C) 2010-2012 ARM Limited. All rights reserved.
*
* $Date:         17. January 2013
* $Revision:     V1.4.0
*
* Project:       CMSIS DSP Library
* Title:       arm_fft_bin_example_f32.c
*
* Description:   Example code demonstrating calculation of Max energy bin of
*                frequency domain of input signal.
*
* Target Processor: Cortex-M4/Cortex-M3
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*   - Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   - Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in
*     the documentation and/or other materials provided with the
*     distribution.
*   - Neither the name of ARM LIMITED nor the names of its contributors
*     may be used to endorse or promote products derived from this
*     software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_const_structs.h"
#include "fixed_data.h"
#include "sww_ref_util.h"
#include <stdio.h>


#define TEST_LENGTH_SAMPLES 2048

/* -------------------------------------------------------------------
* External Input and Output buffer Declarations for FFT Bin Example
* ------------------------------------------------------------------- */

static float32_t testOutput[TEST_LENGTH_SAMPLES/2];


void test_extraction(const float32_t input_signal[])
{
  float32_t maxValue;
  float32_t fftbuff[TEST_LENGTH_SAMPLES];
  uint32_t testIndex = 0;

  uint32_t fftSize = 1024;
  uint32_t ifftFlag = 0;
  uint32_t doBitReverse = 1;


  for(int i=0;i<TEST_LENGTH_SAMPLES;i++){
	  fftbuff[i] = input_signal[i];
  }

  /* Process the data through the CFFT/CIFFT module */
  printf("Before FFT: %3.4f, %3.4f, %3.4f, %3.4f\r\n",
		  fftbuff[0], fftbuff[1], fftbuff[2], fftbuff[3]);
  arm_cfft_f32(&arm_cfft_sR_f32_len1024, fftbuff, ifftFlag, doBitReverse);

  printf("After FFT: %3.4f, %3.4f, %3.4f, %3.4f, %3.4f, %3.4f, %3.4f, %3.4f\r\n",
		  fftbuff[0], fftbuff[1], fftbuff[2], fftbuff[3],
		  fftbuff[4], fftbuff[5], fftbuff[6], fftbuff[7]);

  /* Process the data through the Complex Magnitude Module for
  calculating the magnitude at each bin */
  arm_cmplx_mag_f32(fftbuff, testOutput, fftSize);

//  printf("Mag: %3.4f, %3.4f, %3.4f, %3.4f\r\n",
//		  testOutput[0], testOutput[1], testOutput[2], testOutput[3]);
  printf("Magnitude output\r\n");
  print_vals_float(testOutput, TEST_LENGTH_SAMPLES/2);

  /* Calculates maxValue and returns corresponding BIN value */
  arm_max_f32(testOutput, fftSize, &maxValue, &testIndex);
  printf("Max value = %f at index %lu\r\n", maxValue, testIndex);

}
