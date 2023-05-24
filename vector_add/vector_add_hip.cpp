/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "vector_add/util.hpp"


#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif

#define WIDTH  32384
#define HEIGHT 32384
#define NUM    (WIDTH*HEIGHT)  // 32384 * 32384 ~= 1 billion elements

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16
#define THREADS_PER_BLOCK_Z  1

__global__ void
vectoradd_float(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int width, int height)
  {
      int x = blockDim.x * blockIdx.x + threadIdx.x;
      int y = blockDim.y * blockIdx.y + threadIdx.y;
      int i = y * width + x;
      if (i < (width * height)) {
        c[i] = a[i] + b[i];
      }
  }

using namespace std;

int main() {
  hipDeviceProp_t devProp;
  HIP_ASSERT(hipGetDeviceProperties(&devProp, 0));
  cout << "device name: " << devProp.name << endl;

  auto start = chrono::system_clock::now();
  float* hostA = (float*)malloc(NUM * sizeof(float));
  float* hostB = (float*)malloc(NUM * sizeof(float));
  float* hostC = (float*)malloc(NUM * sizeof(float));
  if (!hostA || !hostB || !hostC) {
    cout << "failed allocating host memory\n";
    return -1;
  }
  print_elapsed(&start, "allocate host memory");

  // initialize the input data
  int i;
  for (i = 0; i < NUM; i++) {
    hostA[i] = (float)i;
    hostB[i] = (float)i*100.0f;
  }
  print_elapsed(&start, "initialize host memory");

  float* deviceA;
  float* deviceB;
  float* deviceC;

  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(float)));
  print_elapsed(&start, "allocate device memory");


  HIP_ASSERT(hipMemcpy(deviceA, hostA, NUM*sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM*sizeof(float), hipMemcpyHostToDevice));
  print_elapsed(&start, "copy to device memory");

  hipLaunchKernelGGL(vectoradd_float,
                  dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  deviceA, deviceB, deviceC, WIDTH, HEIGHT);

  HIP_ASSERT(hipMemcpy(hostC, deviceC, NUM*sizeof(float), hipMemcpyDeviceToHost));
  print_elapsed(&start, "run kernel and copy from device memory");

  // verify the results
  int errors = 0;
  for (i = 0; i < NUM; i++) {
      if (hostC[i] != (hostA[i] + hostB[i])) {
        errors++;
        if (errors == 1) {
            printf("Error at index %d: Expected %f, got %f\n", i, hostA[i] + hostB[i], hostC[i]);
        }
      }
  }
  if (errors) {
      printf("FAILED: %d errors\n",errors);
  } else {
      printf("PASSED!\n");
  }

  HIP_ASSERT(hipFree(deviceA));
  HIP_ASSERT(hipFree(deviceB));
  HIP_ASSERT(hipFree(deviceC));

  free(hostA);
  free(hostB);
  free(hostC);

  return errors;
}
