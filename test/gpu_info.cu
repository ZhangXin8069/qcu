/*
// 编译
nvcc gpu_info.cpp
// 执行
./a.ou
// 输出 GPU 信息如下
GPU Name = GeForce GTX 1080 Ti
Compute Capability = 6.1
GPU SMs = 28
GPU SM clock rate = 1.683 GHz
GPU Mem clock rate = 5.505 GHz
*/
// gpu_info.cpp 源代码如下
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK_CUDA(x, str)                                                     \
  if ((x) != cudaSuccess) {                                                    \
    fprintf(stderr, str);                                                      \
    exit(EXIT_FAILURE);                                                        \
  }
/*
nvcc gpu_info.cu -o gpu_info.x
Michael Pohoreski
Copyleft {c} 2013
*/
#include <cuda.h>
#include <stdio.h>
/*
    GeForce GTX Titan @ 928 MHz
        SM: 14 * 192 sm/core = 2688 Cores
        384-bit @ 3004 MHz = 288 GB/s
    GeForce GT 750M @ 925 MHz
        2 * 192 Cores/SM = 384 Cores
        128-bit @ 2508 MHz = 80 GB/s
    GeForce GT 330M @ 1100 MHz
        SM: 6 * 8 sm/core = 48 Cores
        128-bit @ 790 MHz = 25 GB/s
*/
int CudaGetCores(int major, int minor) {
  int cores[] = {
      8,   8,  8, 8, 0, 0,   // 1.0  1.1  1.2  1.3  -.-  -.-
      32,  48, 0, 0, 0, 0,   // 2.0  2.1
      192, 0,  0, 0, 0, 192, // 3.0                      3.5
      256, 0,  0, 0, 0, 0    // 4.0
  };
  return cores[6 * (major - 1) + minor];
}
// cudaDeviceProp()
// Reference:
// http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__DEVICE_g5aa4f47938af8276f08074d09b7d520c.html
// https://devblogs.nvidia.com/parallelforall/how-query-device-properties-and-handle-errors-cuda-cc/
// https://devblogs.nvidia.com/parallelforall/how-implement-performance-metrics-cuda-cc/
void main0() {
  int devices;
  cudaError_t error = cudaGetDeviceCount(&devices);
  if (error != cudaSuccess)
    printf("ERROR: Couldn't find any CUDA devices.\n");
  for (int device = 0; device < devices; device++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("\nGPU #%d: \'%s\' @ %d MHz\n", (device + 1), prop.name,
           (prop.clockRate / 1000));
    printf("   Compute: %d.%d\n", prop.major, prop.minor);
    printf("   Multi Processors: %d * %d Cores/SM = %d Cores\n",
           prop.multiProcessorCount, CudaGetCores(prop.major, prop.minor),
           prop.multiProcessorCount * CudaGetCores(prop.major, prop.minor));
    printf("\n=== Memory ===\n");
    printf("   Total Memory : %lu MB (%lu bytes)\n",
           (prop.totalGlobalMem / 1024) / 1024, (size_t)prop.totalGlobalMem);
    printf("   Bus Width    : %u-bit @ %d MHz ==> ", prop.memoryBusWidth,
           prop.memoryClockRate / 1000);
    printf("   Max Bandwidth: %u GB/s\n",
           (prop.memoryClockRate / 1000 * ((prop.memoryBusWidth / 8) * 2)) /
               1000); // DDR2/3/4/5 = *2
    printf("   Const memory : %lu (bytes)\n", prop.totalConstMem);
    printf("   Memory/Block : %lu\n", prop.sharedMemPerBlock);
    printf("   Unified mem  : %d\n", prop.unifiedAddressing);
    printf("\n=== Threads ===\n");
    printf("   Max Threads/SM : %d \n", prop.maxThreadsPerMultiProcessor);
    printf("   Threads / Block: %d\n", prop.maxThreadsPerBlock);
    printf("   Max Thread Size: %d, %d, %d\n", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("   Max Grid size  : %u, %u, %u\n", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("   Registers/Block: %d\n", prop.regsPerBlock);
    printf("\n=== Texture ===\n");
    printf("   Texture Size 1D: %d          \n", prop.maxTexture1D);
    printf("   Texture Size 2D: %d x %d     \n", prop.maxTexture2D[0],
           prop.maxTexture2D[1]);
    printf("   Texture Size 3D: %d x %d x %d\n", prop.maxTexture3D[0],
           prop.maxTexture3D[1], prop.maxTexture3D[2]);
    printf("\n");
  }
  //   return 0;
}
void main1(void) {
  int gpu_index = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, gpu_index),
             "cudaGetDeviceProperties error");
  printf("GPU Name = %s\n", prop.name);
  printf("Compute Capability = %d.%d\n", prop.major,
         prop.minor);                                 // 获得 SM 版本
  printf("GPU SMs = %d\n", prop.multiProcessorCount); // 获得 SM 数目
  printf("GPU SM clock rate = %.3f GHz\n",
         prop.clockRate /
             1e6); // prop.clockRate 单位为 kHz，除以 1e6 之后单位为 GHz
  printf("GPU Mem clock rate = %.3f GHz\n", prop.memoryClockRate / 1e6); // 同上
  if ((prop.major == 8) && (prop.minor == 0)) // SM 8.0，即 A100
  {
    // 根据公式计算峰值吞吐，其中 64、32、256、256 是从表中查到
    printf("-----------CUDA Core Performance------------\n");
    printf("FP32 Peak Performance = %.3f GFLOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 64 * 2);
    printf("FP64 Peak Performance = %.3f GFLOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 32 * 2);
    printf("FP16 Peak Performance = %.3f GFLOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 256 * 2);
    printf("BF16 Peak Performance = %.3f GFLOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 128 * 2);
    printf("INT8 Peak Performance = %.3f GOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 256 * 2);
    printf("-----------Tensor Core Dense Performance------------\n");
    printf("TF32 Peak Performance = %.3f GFLOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 512 * 2);
    printf("FP64 Peak Performance = %.3f GFLOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 64 * 2);
    printf("FP16 Peak Performance = %.3f GFLOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 1024 * 2);
    printf("BF16 Peak Performance = %.3f GFLOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 1024 * 2);
    printf("INT8 Peak Performance = %.3f GOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 2048 * 2);
    printf("INT4 Peak Performance = %.3f GOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 4096 * 2);
    printf("INT1 Peak Performance = %.3f GOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 16384 * 2);
    printf("-----------Tensor Core Sparse Performance------------\n");
    printf("TF32 Peak Performance = %.3f GFLOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 1024 * 2);
    printf("FP16 Peak Performance = %.3f GFLOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 2048 * 2);
    printf("BF16 Peak Performance = %.3f GFLOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 2048 * 2);
    printf("INT8 Peak Performance = %.3f GOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 4096 * 2);
    printf("INT4 Peak Performance = %.3f GOPS.\n",
           prop.multiProcessorCount * (prop.clockRate / 1e6) * 8192 * 2);
  }
  //   return 0;
}
int main() {
  main0();
  main1();
  return 0;
}