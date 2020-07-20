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
/// \brief Submitter-implemented methods required to perform inference.

#ifndef MLPERF_TINY_V0_1_MODEL_H_
#define MLPERF_TINY_V0_1_MODEL_H_

#include <stdint.h>

namespace mlperf {
namespace tiny {

/// \brief Status to indicate the result of setting an input.
typedef enum {
  kTinyMlPerfInputContinue,
  kTinyMlPerfInputDone,
} TinyMlPerfInputStatus;

/// \brief Initialize the model
/// \detail Perform any one-time model initialization. This is not benchmarked.
void Initialize();

/// \brief Provide a model input.
/// \detail Set the model input to prepare for Invoke(). Perform any
/// per-inference initialization within this method. Index is an input counter
/// which may be helpful to index on-device input arrays. This method is not
/// benchmarked. Return kTinyMlPerfInputDone if no inputs are left, otherwise
/// return kTinyMlPerfInputContinue.
TinyMlPerfInputStatus SetInput(int index);

/// \brief Perform one inference cycle.
/// \detail Consume the last input provided by SetInput() and produce an output
/// ready to be fetched by GetOutput(). This is the method measured for
/// the benchmark, so implementations should be limited to work required solely
/// for inference.
void Invoke();

/// \brief Return the model output produced by the last call to Invoke.
int GetOutput(float* output_buffer);

}  // namespace tiny
}  // namespace mlperf

#endif  // MLPERF_TINY_V0_1_MODEL_H_
