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

#include "api/internally_implemented.h"
#include "api/submitter_implemented.h"

int main(int argc, char *argv[]) {
  th_printf("jhdbg: Just Starting\r\n");
  th_printf("jhdbg: before eebm_init\r\n");
  ee_benchmark_initialize();
  th_printf("jhdbg: after eebm_init\r\n");
  while (1) {
    int c;
    c = th_getchar();
    ee_serial_callback(c);
  }
  return 0;
}
