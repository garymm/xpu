#include <cstdint>
#include <iostream>
#include <chrono>
#include "vector_add/util.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

constexpr size_t NUM = 32384 * 32384;

int main()
{
std::cout << "Thrust backend: ";
#if THRUST_DEVICE_SYSTEM == 0
  std::cout << "undefined" << std::endl;
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  std::cout << "CUDA" << std::endl;
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
  std::cout << "TBB" << std::endl;
#endif

  auto start = std::chrono::system_clock::now();
  thrust::device_vector<float> a_device(NUM);
  thrust::device_vector<float> b_device(NUM);

  // initialize the input data
  thrust::sequence(a_device.begin(), a_device.end(), 1);
  thrust::sequence(b_device.begin(), b_device.end(), 100);
  thrust::device_vector<float> c_device(a_device.size());
  print_elapsed(&start, "initialize input memory");

  thrust::transform(a_device.begin(), a_device.end(), b_device.begin(), c_device.begin(), thrust::plus<float>());
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  // copy data from GPU to CPU
  thrust::host_vector<float> a = a_device;
  thrust::host_vector<float> b = b_device;
  thrust::host_vector<float> c = c_device;
#else
  thrust::device_vector<float>& a = a_device;
  thrust::device_vector<float>& b = b_device;
  thrust::device_vector<float>& c = c_device;
#endif
  print_elapsed(&start, "run kernel and copy from device memory");

  // check the results
  int errors = 0;
  for (size_t i = 0; i < NUM; i++) {
      if (c[i] != (a[i] + b[i])) {
        errors++;
        std::cout << "Error at index " << i << ": Expected " << a[i] + b[i] << ", got " << c[i] << "\n";
      }
  }
  print_elapsed(&start, "check results");
  if (errors) {
      printf("FAILED: %d errors\n", errors);
  } else {
      printf("PASSED!\n");
  }
  return errors;
}
