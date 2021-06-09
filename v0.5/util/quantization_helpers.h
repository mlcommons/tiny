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
/// \brief Quantization helpers to convert to and from quantized types.

#include <limits.h>
#include <math.h>

inline float DequantizeInt8ToFloat(int8_t value, float scale, int zero_point) {
  return static_cast<float>(value - zero_point) * scale;
}

inline int8_t QuantizeFloatToInt8(float value, float scale, int zero_point) {
  int32_t result = round(value / scale) + zero_point;
  if (result < INT8_MIN) {
    result = INT8_MIN;
  }
  if (result > INT8_MAX) {
    result = INT8_MAX;
  }
  return static_cast<int8_t>(result);
}
