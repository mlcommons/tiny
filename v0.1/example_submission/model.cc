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
/// \brief Example of TinyMLPerf submission model.cc file for VWW on TF Micro.

#include "model.h"

#include "tf_micro_model_runner.h"
#include "util/quantization_helpers.h"
#include "vww/vww_inputs.h"
#include "vww/vww_model_data.h"
#include "vww/vww_model_settings.h"

namespace mlperf {
namespace tiny {

namespace {

constexpr int kTensorArenaSize = 150 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroModelRunner<int8_t, int8_t>* runner;

}  // namespace

void Initialize() {
  static tflite::MicroModelRunner<int8_t, int8_t> model_runner(
      g_person_detect_model_data, tensor_arena, kTensorArenaSize);
  runner = &model_runner;
}

TinyMlPerfInputStatus SetInput(int index) {
  if (index >= kNumVwwTestInputs) {
    return kTinyMlPerfInputDone;
  }
  int8_t input[kVwwInputSize];
  for (int i = 0; i < kVwwInputSize; i++) {
    input[i] =
        QuantizeFloatToInt8(g_vww_inputs[index][i], runner->input_scale(),
                            runner->input_zero_point());
  }
  runner->SetInput(input);
  return kTinyMlPerfInputContinue;
}

void Invoke() { runner->Invoke(); }

int GetOutput(float* output_buffer) {
  int8_t* output = runner->GetOutput();
  for (int i = 0; i < runner->output_size(); i++) {
    output_buffer[i] = DequantizeInt8ToFloat(output[i], runner->output_scale(),
                                             runner->output_zero_point());
  }
  return runner->output_size();
}

}  // namespace tiny
}  // namespace mlperf
