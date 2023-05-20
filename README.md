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


## TODO

* Support building on ARM64. Currently using Bootlin
  toolchain which only exists for x86-64. Probably easiest
  to:
  *  wait for libc++ (LLVM's std lib) to implement
     std::execution (AKA PSTL), then we can switch to use
     LLVM toolchain, or
  *  switch to conda environment or docker container
     that installs the toolchain (don't use a bazel
     hermetic toolchain).
