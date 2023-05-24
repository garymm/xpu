# vector addition on CPU and GPU

Adds two vectors of 1 billion floats.

## Thrust

### CPU

```sh
bazel run -c opt --run_under=time //vector_add:vector_add_thrust

Thrust backend: TBB
initialize input memory: 713ms
run kernel and copy from device memory: 246ms
check results: 1317ms
PASSED!

real	0m2.550s
```

### GPU

```sh
bazel run -c opt --cuda --run_under=time //vector_add:vector_add_thrust

Thrust backend: CUDA
initialize input memory: 335ms
run kernel and copy from device memory: 5269ms
check results: 1195ms
PASSED!

real	0m7.137s
```

## Intel DPC++ / SYCL

### CPU

```sh
icpx -O3 -fsycl -Xs "-march=avx512" -fsycl-targets=spir64_x86_64 ~/src/garymm/xpu/vector_add/vector_add_sycl.cpp ~/src/garymm/xpu/vector_add/util.* && mv a.out vector_add_cpu && time ./vector_add_cpu

Running on Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz
allocate input memory: 4µs
allocate output memory: 6µs
initialize input data: 2ms
run kernel: 581ms
get_host_access(): 421µs
check results: 1819ms
PASSED!

2.913 total
```

### GPU

```sh
icpx -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_86 ~/src/garymm/xpu/vector_add/vector_add_sycl.cpp ~/src/garymm/xpu/vector_add/util.* && mv a.out vector_add_cuda && time ./vector_add_cuda

Running on NVIDIA RTX A4000
allocate input memory: 4µs
allocate output memory: 4µs
initialize input data: 5ms
run kernel: 52ms
get_host_access(): 5089ms
check results: 1792ms
PASSED!

7.604 total
```
