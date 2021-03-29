/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_ANOMALY_DETECTION_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_ANOMALY_DETECTION_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_

// The following values are derived from values used during model training.
// If you change the way you preprocess the input, update all these constants.
const int kFeatureSliceSize = 128;
const int kFeatureSliceCount = 5;
const int kFeatureElementCount = (kFeatureSliceSize * kFeatureSliceCount);

const int kSpectrogramSliceCount = 200;
const int kInputSize = (kFeatureSliceSize * kFeatureSliceCount);
//const int kFeatureWindows = kSpectrogramSliceCount - kFeatureSliceCount + 1;    // 200 - 5 + 1

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_ANOMALY_DETECTION_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_
