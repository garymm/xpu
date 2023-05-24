# XPU

single source that compiles to run on both GPU and CPU

## Set up

Currently only supported on Linux x86_64 / AMD64.

* Install `bazel` or `bazelisk`

* Verify that you can build and test:
```
bazel test //...
```

And with CUDA:

```
bazel test --cuda //...
```


## Status

### SYCL

* Got it building and running with OpenSYCL syclcc on CPU, but not
  through bazel.
* Doesn't work with nvc++ possibly due to an nvc++ bug.
  <https://github.com/OpenSYCL/OpenSYCL/issues/1052>.
* Got it building and running with Intel DPC++ on CPU and GPU, but not through bazel.

### HIP

Working with bazel for both CUDA and CPU.

### Thrust

Working with bazel for both CUDA and CPU.
