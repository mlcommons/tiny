/*
Copyright (C) EEMBC(R). All Rights Reserved

All EEMBC Benchmark Software are products of EEMBC and are provided under the
terms of the EEMBC Benchmark License Agreements. The EEMBC Benchmark Software
are proprietary intellectual properties of EEMBC and its Members and is
protected under all applicable laws, including all applicable copyright laws.

If you received this EEMBC Benchmark Software without having a currently
effective EEMBC Benchmark License Agreement, you must discontinue use.

Copyright 2020 The MLPerf Authors. All Rights Reserved.
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
#include "util/quantization_helpers.h"
#include "util/tf_micro_model_runner.h"
//#include "inputs.h"
#include "model.h"
#include "micro_model_settings.h"

UnbufferedSerial pc(USBTX, USBRX, 115200);

constexpr int kTensorArenaSize = 150 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

typedef int8_t model_input_t;
typedef int8_t model_output_t;

float input_float[kInputSize];
int8_t input_quantized[kInputSize];
float results[kFeatureWindows];
 
tflite::MicroModelRunner<model_input_t, model_output_t, 6> *runner;

// Implement this method to prepare for inference and preprocess inputs.
void th_load_tensor() {
  size_t bytes = ee_get_buffer(reinterpret_cast<uint8_t *>(input_float),
                               kInputSize * sizeof(float));
  if (bytes / sizeof(float) != kInputSize) {
    th_printf("Input db has %d elemented, expected %d\n", bytes / sizeof(float),
              kInputSize);
    return;
  }
}

// Add to this method to return real inference results.
void th_results() {
  /**
   * The results need to be printed back in exactly this format; if easier
   * to just modify this loop than copy to results[] above, do that.
   */
  th_printf("m-results-[");
  for (size_t i = 0; i < kFeatureWindows; i++) {
    th_printf("%0.3f", results[i]);
    if (i < (kFeatureWindows - 1)) {
      th_printf(",");
    }
  }
  th_printf("]\r\n");
}

// Implement this method with the logic to perform one inference cycle.
void th_infer() {

  th_printf("quantization %i %i\r\n", int(runner->input_scale()*100), int(runner->input_zero_point()*100));
  th_timestamp();

  float input_scale = runner->input_scale();
  float input_zero_point = runner->input_zero_point();
  for (int i = 0; i < kInputSize; i++) {
    input_quantized[i] = QuantizeFloatToInt8(
        input_float[i], input_scale, input_zero_point);
  }

  th_printf("inference loop\r\n");
  th_timestamp();

  for (int window = 0; window < kFeatureWindows; window++) {
    th_printf("set input\r\n");
    th_timestamp();
    runner->SetInput(input_quantized + window * kFeatureSliceSize);

    th_printf("invoke\r\n");
    th_timestamp();
    runner->Invoke();

    th_printf("postproc\r\n");
    th_timestamp();
    // calculate |output - input|
    float diffsum = 0;

    for (size_t i = 0; i < kFeatureElementCount; i++) {
      float converted = DequantizeInt8ToFloat(runner->GetOutput()[i], runner->output_scale(),
                                              runner->output_zero_point());
      float diff = converted - input_float[i + window * kFeatureSliceSize];
      diffsum += diff * diff;
    }
    diffsum /= kFeatureElementCount;
    th_printf("result*100 %i\r\n", int(diffsum));

    results[window] = diffsum;
  }

}

/// \brief optional API.
void th_final_initialize(void) {
  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddFullyConnected();
  resolver.AddQuantize();
  resolver.AddDequantize();
  static tflite::MicroModelRunner<model_input_t, model_output_t, 6> model_runner(
      g_model, resolver, tensor_arena, kTensorArenaSize);
  runner = &model_runner;
  th_printf("Runner initialized %p\r\n", runner);
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

void th_serialport_initialize(void) { pc.baud(115200); }

void th_timestamp(void) {
  unsigned long microSeconds = 0ul;
  /* USER CODE 2 BEGIN */
  microSeconds = us_ticker_read();
  /* USER CODE 2 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP, microSeconds);
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
