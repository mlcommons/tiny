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
/// \brief Submitter-provided methods implementing platform-specific timers.

#ifndef MLPERF_TINY_V0_1_TIMERS_H_
#define MLPERF_TINY_V0_1_TIMERS_H_

#include <stdint.h>

namespace mlperf {
namespace tiny {

/// \brief Return the number of timer ticks per second.
int32_t TicksPerSecond();

/// \brief Return time in ticks.
/// \detail The meaning of a tick varies per platform. The returned Ticks value
/// can be converted to seconds using TicksPerSecond().
int32_t CurrentTimeTicks();

}  // namespace tiny
}  // namespace mlperf

#endif  // MLPERF_TINY_V0_1_TIMERS_H_
