# cuFFT MultiGPU 1D FFT C2C example

## Description

This code demonstrates the usage of single node, multiGPU cuFFT C2C functionality, on a 1D dataset. Use single gpu version as reference. Maximum FFT size limited by single GPU memory.

Note, because cuFFT 10.4.0 cufftSetStream can be used to associate a stream with multi-GPU plan. cufftXtExecDescriptor synchronizes efficiently to the stream before and after execution. Please refer to https://docs.nvidia.com/cuda/cufft/index.html#function-cufftsetstream for more information.

cuFFT by default executes multi-GPU plans in synchronous manner.

The cuFFT library doesn't guarantee that single-GPU and multi-GPU cuFFT plans will perform mathematical operations in same order. Small numerical differences are possible.

## Supported SM Architectures

All GPUs supported by CUDA Toolkit (https://developer.nvidia.com/cuda-gpus)  

## Supported OSes

Linux  
Windows

## Supported CPU Architecture

x86_64  
ppc64le  
arm64-sbsa

## CUDA APIs involved
- [cufftXtSetGPUs API](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftxtsetgpus)
- [cufftMakePlan1d API](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftmakeplan1d)
- [cufftXtExecDescriptor API](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftxtexecdescriptor)
- [cufftXtMalloc API](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftxtmalloc)
- [cufftXtMemcpy API](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftxtmemcpy)

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- [CMake](https://cmake.org/download) version 3.18 minimum

## Build command on Linux
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
Make sure that CMake finds expected CUDA Toolkit. If that is not the case you can add argument `-DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc` to cmake command.

# Usage 1
```
$  ./bin/1d_mgpu_c2c_example
```

Sample example output:

```
PASSED with L2 error = 1.25194e-07
```