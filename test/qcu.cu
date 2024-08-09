#include "checker_utils.h"
#include "cublas_utils.h"
#include "curand_utils.h"
#include "timer_utils.h"
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
using data_type = cuDoubleComplex;
int main(int argc, char *argv[]) {
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  /*
   *   A = | 1.1 + 1.2j | 2.3 + 2.4j | 3.5 + 3.6j | 4.7 + 4.8j |
   *   B = | 5.1 + 5.2j | 6.3 + 6.4j | 7.5 + 7.6j | 8.7 + 8.8j |
   */
  const std::vector<data_type> A = {
      {1.1, 1.2}, {2.3, 2.4}, {3.5, 3.6}, {4.7, 4.8}};
  std::vector<data_type> B = {{5.1, 5.2}, {6.3, 6.4}, {7.5, 7.6}, {8.7, 8.8}};
  const data_type alpha = {2.1, 1};
  const int incx = 1;
  const int incy = 1;
  data_type *d_A = nullptr;
  data_type *d_B = nullptr;
  printf("A\n");
  print_vector(A.size(), A.data());
  printf("=====\n");
  printf("B\n");
  print_vector(B.size(), B.data());
  printf("=====\n");
  /* step 1: create cublas handle, bind a stream */
  CUBLAS_CHECK(cublasCreate(&cublasH));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));
  /* step 2: copy data to device */
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A),
                        sizeof(data_type) * A.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B),
                        sizeof(data_type) * B.size()));
  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(),
                             cudaMemcpyHostToDevice, stream));
  /* step 3: compute */
  CUBLAS_CHECK(cublasAxpyEx(cublasH, A.size(), &alpha,
                            traits<data_type>::cuda_data_type, d_A,
                            traits<data_type>::cuda_data_type, incx, d_B,
                            traits<data_type>::cuda_data_type, incy,
                            traits<data_type>::cuda_data_type));
  /* step 4: copy data to host */
  CUDA_CHECK(cudaMemcpyAsync(B.data(), d_B, sizeof(data_type) * B.size(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  printf("B\n");
  print_vector(B.size(), B.data());
  printf("=====\n");
  /* free resources */
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUBLAS_CHECK(cublasDestroy(cublasH));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaDeviceReset());
  /*
alpha = 2.1 + 1j
A
1.10 + 1.20j 2.30 + 2.40j 3.50 + 3.60j 4.70 + 4.80j
=====
B
5.10 + 5.20j 6.30 + 6.40j 7.50 + 7.60j 8.70 + 8.80j
=====
B
6.21 + 8.82j 8.73 + 13.74j 11.25 + 18.66j 13.77 + 23.58j
=====
dest(B) = B + alpha*A
  */
  // {
  //   int N = 8388608;
  //   use_host_api(N);
  //   use_device_api(N);
  // }
  return EXIT_SUCCESS;
}
