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
/// \brief Main function to run benchmark on device.

#include "../logging.h"
#include "../model.h"
#include "../timers.h"

int main() {
  constexpr int64_t kUsecPerSecond = 1000000;
  constexpr int kMaxOutputSize = 10;
  float output_buffer[kMaxOutputSize];
  mlperf::tiny::Initialize();

  int index = 0;
  while (mlperf::tiny::SetInput(index++) !=
         mlperf::tiny::kTinyMlPerfInputDone) {
    int32_t start = mlperf::tiny::CurrentTimeTicks();
    mlperf::tiny::Invoke();
    int32_t end = mlperf::tiny::CurrentTimeTicks();
    int64_t duration_ticks = end - start;
    int64_t duration_usec =
        duration_ticks * kUsecPerSecond / mlperf::tiny::TicksPerSecond();
    mlperf::tiny::LogToHost(
        "invocation %d took %lld cycles %lld microseconds\n", index,
        duration_ticks, duration_usec);
    int len = mlperf::tiny::GetOutput(output_buffer);
    mlperf::tiny::LogToHost("Model output: [");
    for (int i = 0; i < len; i++) {
      mlperf::tiny::LogToHost("%f ", output_buffer[i]);
    }
    mlperf::tiny::LogToHost("]\n");
  }
  return 0;
}
