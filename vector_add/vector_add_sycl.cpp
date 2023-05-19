#include <cassert>
#include <iostream>
#include <chrono>
#include <vector>

#include <CL/sycl.hpp>

constexpr size_t NUM = 16192 * 8096;

int main()
{
  cl::sycl::queue q;
  auto start = chrono::system_clock::now();
  std::vector<float> a(NUM);
  std::vector<float> b(NUM);
  print_elapsed(&start, "allocate host memory");

  // initialize the input data
  int i;
  for (i = 0; i < NUM; i++) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i*100);
  }
  print_elapsed(&start, "initialize host memory");

  std::vector<float> c(a.size());
  cl::sycl::range<1> work_items{a.size()};

  {
    cl::sycl::buffer<float> buff_a(a.data(), a.size());
    cl::sycl::buffer<float> buff_b(b.data(), b.size());
    cl::sycl::buffer<float> buff_c(c.data(), c.size());

    q.submit([&](cl::sycl::handler& cgh){
      auto access_a = buff_a.get_access<cl::sycl::access::mode::read>(cgh);
      auto access_b = buff_b.get_access<cl::sycl::access::mode::read>(cgh);
      auto access_c = buff_c.get_access<cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class vector_add>(work_items,
                                         [=] (cl::sycl::id<1> tid) {
        access_c[tid] = access_a[tid] + access_b[tid];
      });
    });
  }
  print_elapsed(&start, "run kernel and copy from device memory");

  // verify the results
  int errors = 0;
  for (i = 0; i < NUM; i++) {
      if (c[i] != (a[i] + b[i])) {
        errors++;
        if (errors == 1) {
            printf("Error at index %d: Expected %f, got %f\n", i, a[i] + b[i], c[i]);
        }
      }
  }
  if (errors) {
      printf("FAILED: %d errors\n",errors);
  } else {
      printf("PASSED!\n");
  }
  return errors;
}
