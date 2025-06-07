/*
Copyright 2020 EEMBC and The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file reflects a modified version of th_lib from EEMBC. The reporting logic
in th_results is copied from the original in EEMBC.
==============================================================================*/
/// \file
/// \brief C++ implementations of submitter_implemented.h

#include "api/submitter_implemented.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "api/internally_implemented.h"
#include "mbed.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "util/quantization_helpers.h"
#include "util/tf_micro_model_runner.h"
//#include "inputs.h"
#include "model.h"
#include "micro_model_settings.h"

UnbufferedSerial pc(USBTX, USBRX);
DigitalOut g_timestampPin(D7);

constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

typedef int8_t model_input_t;
typedef int8_t model_output_t;

float input_float[kInputSize];
int8_t input_quantized[kInputSize];
float result;
 
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;

// copy input into interpreter's buffer
void copy_input() {
  int8_t *model_input_buffer = model_input->data.int8;
  int8_t *feature_buffer_ptr = input_quantized;

  // Copy feature buffer to input tensor
  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer_ptr[i];
  }
}

// calculate |output - input|
void calculate_result(){
  float diffsum = 0;

  TfLiteTensor* output = interpreter->output(0);
  for (size_t i = 0; i < kFeatureElementCount; i++) {
    float converted = DequantizeInt8ToFloat(output->data.int8[i], interpreter->output(0)->params.scale,
                                            interpreter->output(0)->params.zero_point);
    float diff = converted - input_float[i];
    diffsum += diff * diff;
  }
  diffsum /= kFeatureElementCount;

  result = diffsum;
}

// Implement this method to prepare for inference and preprocess inputs.
void th_load_tensor() {
  size_t bytes = ee_get_buffer(reinterpret_cast<uint8_t *>(input_float),
                               kInputSize * sizeof(float));
  if (bytes / sizeof(float) != kInputSize) {
    th_printf("Input db has %d elemented, expected %d\n", bytes / sizeof(float),
              kInputSize);
    return;
  }

  float input_scale = interpreter->input(0)->params.scale;
  int input_zero_point = interpreter->input(0)->params.zero_point;
  for (int i = 0; i < kInputSize; i++) {
    input_quantized[i] = QuantizeFloatToInt8(
        input_float[i], input_scale, input_zero_point);
  }

  // copy input into interpreter's buffer
  copy_input();
}

// Add to this method to return real inference results.
void th_results() {

  // calculate |output - input|
  calculate_result();

  /**
   * The results need to be printed back in exactly this format; if easier
   * to just modify this loop than copy to results[] above, do that.
   */
  th_printf("m-results-[%0.3f]\r\n", result);
}

// Implement this method with the logic to perform one inference cycle.
void th_infer() {

  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }
}

/// \brief optional API.
void th_final_initialize(void) {

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<3> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddQuantize() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddDequantize() != kTfLiteOk) {
    return;
  }
  
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] !=
       (kFeatureSliceCount * kFeatureSliceSize)) ||
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }
  th_printf("Initialized\r\n");
}

void th_pre() {}
void th_post() {}

void th_command_ready(char volatile *p_command) {
  p_command = p_command;
  ee_serial_command_parser_callback((char *)p_command);
}

// th_libc implementations.
int th_strncmp(const char *str1, const char *str2, size_t n) {
  return strncmp(str1, str2, n);
}

char *th_strncpy(char *dest, const char *src, size_t n) {
  return strncpy(dest, src, n);
}

size_t th_strnlen(const char *str, size_t maxlen) {
  return strnlen(str, maxlen);
}

char *th_strcat(char *dest, const char *src) { return strcat(dest, src); }

char *th_strtok(char *str1, const char *sep) { return strtok(str1, sep); }

int th_atoi(const char *str) { return atoi(str); }

void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }

void *th_memcpy(void *dst, const void *src, size_t n) {
  return memcpy(dst, src, n);
}

/* N.B.: Many embedded *printf SDKs do not support all format specifiers. */
int th_vprintf(const char *format, va_list ap) { return vprintf(format, ap); }
void th_printf(const char *p_fmt, ...) {
  va_list args;
  va_start(args, p_fmt);
  (void)th_vprintf(p_fmt, args); /* ignore return */
  va_end(args);
}

char th_getchar() { return getchar(); }

void th_serialport_initialize(void) {
#if EE_CFG_ENERGY_MODE == 1
       pc.baud(9600);
#else
       pc.baud(921600);
#endif
}


void th_timestamp(void) {
#if EE_CFG_ENERGY_MODE == 1
/* USER CODE 1 BEGIN */
/* Step 1. Pull pin low */
       g_timestampPin = 0;
       for (int i=0; i<100000; ++i) {
               asm("nop");
       }
/* Step 2. Hold low for at least 1us */
/* Step 3. Release driver */
       g_timestampPin = 1;

/* USER CODE 1 END */
#else
       unsigned long microSeconds = 0ul;
       /* USER CODE 2 BEGIN */
       microSeconds = us_ticker_read();
       /* USER CODE 2 END */
       /* This message must NOT be changed. */
       th_printf(EE_MSG_TIMESTAMP, microSeconds);
#endif
}

void th_timestamp_initialize(void) {
  /* USER CODE 1 BEGIN */
  // Setting up BOTH perf and energy here
  /* USER CODE 1 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP_MODE);
  /* Always call the timestamp on initialize so that the open-drain output
     is set to "1" (so that we catch a falling edge) */
  th_timestamp();
}
