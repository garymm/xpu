#include <cstdint>
#include <iostream>
#include <chrono>
#include <vector>
#include "util.hpp"
#include <CL/sycl.hpp>

constexpr size_t NUM = 32384 * 32384;

int main()
{
  cl::sycl::queue q;
  std::cout << "Running on "
            << q.get_device().get_info<cl::sycl::info::device::name>()
            << "\n";
  auto start = std::chrono::system_clock::now();
  cl::sycl::buffer<float, 1> a(cl::sycl::range<1>{NUM});
  cl::sycl::buffer<float, 1> b(cl::sycl::range<1>{NUM});
  print_elapsed(&start, "allocate input memory");

  cl::sycl::buffer<float, 1> c(cl::sycl::range<1>{NUM});
  print_elapsed(&start, "allocate output memory");

  cl::sycl::range<1> work_items{a.size()};
  q.submit([&](cl::sycl::handler& cgh){
    auto access_a = a.get_access<cl::sycl::access::mode::write>(cgh);
    auto access_b = b.get_access<cl::sycl::access::mode::write>(cgh);
    cgh.parallel_for<class init_input>(work_items,
                                        [=] (cl::sycl::id<1> tid) {
      access_a[tid] = static_cast<float>(tid);
      access_b[tid] = static_cast<float>(tid * 100);
    });
  });
  print_elapsed(&start, "initialize input data");


  q.submit([&](cl::sycl::handler& cgh){
    auto access_a = a.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_b = b.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_c = c.get_access<cl::sycl::access::mode::write>(cgh);

    cgh.parallel_for<class vector_add>(work_items,
                                        [=] (cl::sycl::id<1> tid) {
      access_c[tid] = access_a[tid] + access_b[tid];
    });
  });

  q.wait();
  print_elapsed(&start, "run kernel");

  auto access_a = a.get_host_access();
  auto access_b = b.get_host_access();
  auto access_c = c.get_host_access();
  print_elapsed(&start, "get_host_access()");
  // check the results
  int errors = 0;
  for (size_t i = 0; i < NUM; i++) {
      if (access_c[i] != (access_a[i] + access_b[i])) {
        errors++;
        std::cout << "Error at index " << i << ": Expected " << access_a[i] + access_b[i] << ", got " << access_c[i] << "\n";
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
