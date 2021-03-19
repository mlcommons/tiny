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
/// \brief Visual wakewords model settings.

#ifndef V0_1_IC_MODEL_SETTINGS_H_
#define V0_1_IC_MODEL_SETTINGS_H_

// All of these values are derived from the values used during model training,
// if you change your model you'll need to update these constants.
constexpr int kNumCols = 32;
constexpr int kNumRows = 32;
constexpr int kNumChannels = 3;

constexpr int kIcInputSize = kNumCols * kNumRows * kNumChannels;

constexpr int kCategoryCount = 10;
constexpr int kAirplaneIndex = 0;
constexpr int kAutomobileIndex = 1;
constexpr int kBirdIndex = 2;
constexpr int kCatIndex = 3;
constexpr int kDeerIndex = 4;
constexpr int kDogIndex = 5;
constexpr int kFrogIndex = 6;
constexpr int kHorseIndex = 7;
constexpr int kShipIndex = 8;
constexpr int kTruckIndex = 9;
extern const char* kCategoryLabels[kCategoryCount];

#endif  // V0_1_IC_MODEL_SETTINGS_H_
