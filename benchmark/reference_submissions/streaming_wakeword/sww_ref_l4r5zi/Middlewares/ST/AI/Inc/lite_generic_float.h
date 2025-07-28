/**
  ******************************************************************************
  * @file    lite_conv2d_dqnn.h
  * @author  AIS
  * @brief   header file of AI platform lite conv kernel datatypes
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
#ifndef LITE_GENERIC_FLOAT_H
#define LITE_GENERIC_FLOAT_H

#include "ai_lite_interface.h"

/*****************************************************************************/
/*  Generic Forward Functions Section                                        */
/*****************************************************************************/

/**  Reduce Generic Kernels  *************************************************/
LITE_API_ENTRY
void forward_lite_func_reduce_l1_if32of32( 
  ai_float* out_ptr, const ai_float* in_ptr, 
  const ai_size out_size, const ai_size in_step, 
  const ai_size axis_size, const ai_size axis_step);


LITE_API_ENTRY
void forward_lite_func_reduce_l2_if32of32( 
  ai_float* out_ptr, const ai_float* in_ptr, 
  const ai_size out_size, const ai_size in_step, 
  const ai_size axis_size, const ai_size axis_step);


/**  Split Generic Kernels  **************************************************/

/*!
 * @brief C struct for a generic split layer.
 * @ingroup lite_generic
 * @param outputs_ptr list of pointers for the outputs buffers.
 * @param n_outputs_ptr the number of outputs
 * @param n_outer_elems the number of elements to copy in a single split
 * @param input_ptr the pointer to input buffer to split.
 * @param splits_strides the pointer to array defining outputs split strides.
 * @param splits_step the offset between split strides
 */
typedef struct {
  stai_ptr*         outputs_ptr;
  const stai_size   n_outputs_ptr;
  const stai_size   n_outer_elems;
  const stai_ptr    input_ptr;
  const int32_t*    splits_strides;
  const stai_size   splits_step;
} forward_lite_split_generic_args;


LITE_API_ENTRY
void forward_lite_split_generic(
  forward_lite_split_generic_args* args);


/**  TopK Generic Kernels  ***************************************************/
/*!
 * @brief Handles 2D convolution with binary input, binary output and 
 *        binary weights - with 0 padding (QKeras like) - Lite I/F
 * @ingroup lite_conv2d_dqnn
 */
LITE_API_ENTRY
void forward_lite_topK_axis_0_if32of32(
  const ai_float *pDataIn_init,
  ai_float *pDataOut_values_init,
  ai_i32 *pDataOut_index_init,
  const ai_size height_in,                                        
  const ai_size width_in, 
  const ai_size n_channel_in,
  const ai_size k, ai_i16 largest,
  void (*f)(const ai_float* inputs, ai_float* values, ai_i32* indices, ai_size k, ai_size n_elements, ai_i32 stride, ai_i16 largest)
);

  
/*!
 * @brief Handles 2D convolution with binary input, binary output and 
 *        binary weights - with 0 padding (QKeras like) - Lite I/F 
 *        - Optimized thanks to Optim0 assumptions
 * @ingroup lite_conv2d_dqnn
 */
LITE_API_ENTRY
void forward_lite_topK_axis_1_if32of32(
  const ai_float *pDataIn_init,
  ai_float *pDataOut_values_init,
  ai_i32 *pDataOut_index_init,  
  const ai_size height_in,
  const ai_size width_in,                                 
  const ai_size n_channel_in,
  const ai_size k, ai_i16 largest,
  void (*f)(const ai_float* inputs, ai_float* values, ai_i32* indices, ai_size k, ai_size n_elements, ai_i32 stride, ai_i16 largest)
);

/*!
 * @brief Handles 2D convolution with binary input, 8-bits output and 
 *        binary weights - with 0 padding (QKeras like) - Lite I/F
 * @ingroup lite_conv2d_dqnn
 */
LITE_API_ENTRY
void forward_lite_topK_axis_2_if32of32(
  const ai_float *pDataIn_init,
  ai_float *pDataOut_values_init,
  ai_i32 *pDataOut_index_init,
  const ai_size height_in,                                         
  const ai_size width_in,
  const ai_size n_channel_in, 					  
  const ai_size k, ai_i16 largest,
  void (*f)(const ai_float* inputs, ai_float* values, ai_i32* indices, ai_size k, ai_size n_elements, ai_i32 stride, ai_i16 largest)
);


#endif    /*LITE_GENERIC_FLOAT_H*/
