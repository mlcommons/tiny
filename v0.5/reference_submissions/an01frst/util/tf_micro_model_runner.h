/* Copyright 2020 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/// \file
/// \brief Model runner for TF Micro.

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

template <typename inputT, typename outputT, int numOps>
class MicroModelRunner {
 public:
  MicroModelRunner(const uint8_t* model,
                   MicroMutableOpResolver<numOps> &resolver,
                   uint8_t* tensor_arena, int tensor_arena_size)
      : model_(tflite::GetModel(model)),
        reporter_(&micro_reporter_),
        interpreter_(model_, resolver, tensor_arena, tensor_arena_size,
                     reporter_) {
    interpreter_.AllocateTensors();
  }

  void Invoke() {
    // Run the model on this input and make sure it succeeds.
    TfLiteStatus invoke_status = interpreter_.Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(reporter_, "Invoke failed.");
    }
  }

  void SetInput(const inputT* custom_input) {
    // Populate input tensor with an image with no person.
    TfLiteTensor* input = interpreter_.input(0);
    inputT* input_buffer = tflite::GetTensorData<inputT>(input);
    int input_length = input->bytes / sizeof(inputT);
    for (int i = 0; i < input_length; i++) {
      input_buffer[i] = custom_input[i];
    }
  }

  outputT* GetOutput() {
    return tflite::GetTensorData<outputT>(interpreter_.output(0));
  }

  int input_size() { return interpreter_.input(0)->bytes / sizeof(inputT); }

  int output_size() { return interpreter_.output(0)->bytes / sizeof(outputT); }

  float output_scale() { return interpreter_.output(0)->params.scale; }

  int output_zero_point() { return interpreter_.output(0)->params.zero_point; }

  float input_scale() { return interpreter_.input(0)->params.scale; }

  int input_zero_point() { return interpreter_.input(0)->params.zero_point; }

 private:
  const tflite::Model* model_;
  tflite::MicroErrorReporter micro_reporter_;
  tflite::ErrorReporter* reporter_;
  tflite::MicroInterpreter interpreter_;
};

}  // namespace tflite
