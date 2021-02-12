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
/// \brief Input declarations for visual wakewords model.

#ifndef V0_1_VWW_VWW_INPUTS_H_
#define V0_1_VWW_VWW_INPUTS_H_

#include "vww/vww_model_settings.h"

constexpr int kNumVwwTestInputs = 2;
extern const float g_vww_inputs[kNumVwwTestInputs][kVwwInputSize];

#endif  // V0_1_VWW_VWW_INPUTS_H_
