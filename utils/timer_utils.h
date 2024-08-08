/*
copy from
https://github.com/NVIDIA/CUDALibrarySamples/cuTENSOR/contraction_jit.cu
*/
#include <cuda_runtime.h>
#include <chrono>

struct GPUTimer {
  GPUTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_, 0);
  }

  ~GPUTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() { cudaEventRecord(start_, 0); }

  float seconds() {
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    float time;
    cudaEventElapsedTime(&time, start_, stop_);
    return time * 1e-3;
  }

private:
  cudaEvent_t start_, stop_;
};

/*
        // Set up timing
        GPUTimer timer;
        timer.start();
        HANDLE_ERROR(cutensorContract(handle,
                                      planJit,
                                      (void*) &alpha, A_d, B_d,
                                      (void*) &beta,  C_d, C_d,
                                      workJit, actualWorkspaceSizeJit, stream))

        // Synchronize and measure timing
        auto time = timer.seconds();
*/