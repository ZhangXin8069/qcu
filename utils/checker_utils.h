/*
copy from
https://github.com/NVIDIA/CUDALibrarySamples.git
*/
#pragma once
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include <curand.h>
// CUDA API error checking
#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);       \
      throw std::runtime_error("CUDA error");                                  \
    }                                                                          \
  } while (0)
// cublas API error checking
#define CUBLAS_CHECK(err)                                                      \
  do {                                                                         \
    cublasStatus_t err_ = (err);                                               \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("cublas error");                                \
    }                                                                          \
  } while (0)
// curand API error checking
#define CURAND_CHECK(err)                                                      \
  do {                                                                         \
    curandStatus_t err_ = (err);                                               \
    if (err_ != CURAND_STATUS_SUCCESS) {                                       \
      std::printf("curand error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("curand error");                                \
    }                                                                          \
  } while (0)
template <typename T> void print_vector(const std::vector<T> &data);
template <> void print_vector(const std::vector<float> &data) {
  for (auto &i : data)
    std::printf("%0.6f\n", i);
}
template <> void print_vector(const std::vector<unsigned int> &data) {
  for (auto &i : data)
    std::printf("%d\n", i);
}