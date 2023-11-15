
#include "qcu_complex_computation.cuh"
#include "qcu_complex.cuh"
#include "qcu_macro.cuh"
#include <cuda_runtime.h>

#define DEBUG

// DESCRIBE：  x = ax
static __global__ void sclar_multiply_vector_gpu (void* x, void* a, int vol) {
  // a : complex
  int pos = blockIdx.x * blockDim.x + threadIdx.x;  // thread pos in total
  Complex scalar = *(static_cast<Complex*>(a));

  Complex* x_dst = static_cast<Complex*>(x) + pos * Ns * Nc;

  for (int i = 0; i < Ns * Nc; i++) {
    x_dst[i] = scalar * x_dst[i];
  }
}

// FUNCTION: saxpy_gpu
// DESCRIBE：  y = ax + y，vol is Lx * Ly * Lz * Lt
static __global__ void saxpy_gpu (void* y, void* x, void* a, int vol) {
  // a : complex
  int pos = blockIdx.x * blockDim.x + threadIdx.x;  // thread pos in total
  Complex scalar = *(static_cast<Complex*>(a));

  Complex* y_dst = static_cast<Complex*>(y) + pos * Ns * Nc;
  Complex* x_dst = static_cast<Complex*>(x) + pos * Ns * Nc;

// #ifdef DEBUG
//   if (pos == 0) {
//     printf("before real = %lf, imag = %lf\nafter real = %lf imag = %lf\n", y_dst[0].real(), y_dst[0].imag(), (y_dst[0] + x_dst[0] * scalar).real(), (y_dst[0] + x_dst[0] * scalar).imag());
//   }
// #endif

  for (int i = 0; i < Ns * Nc; i++) {
    y_dst[i] += x_dst[i] * scalar;
  }
}

// FUNCTION: inner product
static __global__ void partial_product_kernel(void* x, void* y, void* partial_result, int vol) {
  int thread_in_total = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_in_block = threadIdx.x;
  int stride, last_stride;
  Complex temp;
  temp.clear2Zero();
  __shared__ Complex cache[BLOCK_SIZE];

  for (int i = thread_in_total; i < vol * Ns * Nc; i += vol) {
    // temp += (*(static_cast<Complex*>(x) + i)) * (*(static_cast<Complex*>(y) + i));
    temp += (*(static_cast<Complex*>(x) + i)).conj() * (*(static_cast<Complex*>(y) + i));
  }
  cache[thread_in_block] = temp;
  __syncthreads();
  // reduce in block
  last_stride = BLOCK_SIZE;
  stride = BLOCK_SIZE / 2;
  while (stride > 0) {
    if (thread_in_block < stride && thread_in_block + stride < last_stride) {
      cache[thread_in_block] += cache[thread_in_block + stride];
    }
    stride /= 2;
    last_stride /= 2;
    __syncthreads();
  }
  if (thread_in_block == 0) {
    *(static_cast<Complex*>(partial_result) + blockIdx.x) = cache[0];
  }
}



// when call this function, set gridDim to 1
static __global__ void reduce_partial_result(void* partial_result, int partial_length) {
  int thread_in_block = threadIdx.x;
  int stride, last_stride;

  Complex temp;
  Complex* src = static_cast<Complex*>(partial_result);
  temp.clear2Zero();
  __shared__ Complex cache[BLOCK_SIZE];

  for (int i = thread_in_block; i < partial_length; i+= BLOCK_SIZE) {
    temp += src[i];
  }
  cache[thread_in_block] = temp;
  __syncthreads();
  // reduce in block
  last_stride = BLOCK_SIZE;
  stride = BLOCK_SIZE / 2;
  while (stride > 0) {
    if (thread_in_block < stride && thread_in_block + stride < last_stride) {
      cache[thread_in_block] += cache[thread_in_block + stride];
    }
    stride /= 2;
    last_stride /= 2;
    __syncthreads();
  }
  if (thread_in_block == 0) {
    *(static_cast<Complex*>(partial_result) + thread_in_block) = cache[0];
  }
}


// avoid malloc partial_result
void gpu_inner_product (void* x, void* y, void* result, void* partial_result, int vol) {
  int grid_size = vol / BLOCK_SIZE;
  int block_size = BLOCK_SIZE;

  partial_product_kernel<<<grid_size, block_size>>>(x, y, partial_result, vol);
  checkCudaErrors(cudaDeviceSynchronize());
  reduce_partial_result<<<1, block_size>>>(partial_result, grid_size);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(result, partial_result, sizeof(Complex), cudaMemcpyDeviceToDevice));
}

void gpu_sclar_multiply_vector (void* x, void* scalar, int vol) {
  dim3 gridDim(vol / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  sclar_multiply_vector_gpu<<<gridDim, blockDim>>>(x, scalar, vol);
  checkCudaErrors(cudaDeviceSynchronize());
}


// every point has Ns * Nc dim vector, y <- y + scalar * x
void gpu_saxpy(void* x, void* y, void* scalar, int vol) {
  dim3 gridDim(vol / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  saxpy_gpu<<<gridDim, blockDim>>>(y, x, scalar, vol);
  checkCudaErrors(cudaDeviceSynchronize());
}

