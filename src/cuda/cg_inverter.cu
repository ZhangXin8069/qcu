#include <cstdio>
#include <cmath>
#include <assert.h>
#include <chrono>
#include <mpi.h>
#include "qcu.h"
#include <cuda_runtime.h>
#include "qcu_complex.cuh"
#include "qcu_complex_computation.cuh"
#include "qcu_macro.cuh"
#include "qcu_clover_dslash.cuh"
#include "qcu_communicator.cuh"
// #define DEBUG

extern MPICommunicator *mpi_comm;

static __global__ void clearVectorKernel(void* vec) {
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  Complex* dst = static_cast<Complex*>(vec) + thread_id * Ns * Nc;
  for (int i = 0; i < Ns * Nc; i++) {
    dst[i].clear2Zero();
  }
}

static void clearVector (void* vec, int vol) {
  clearVectorKernel<<<vol / BLOCK_SIZE, BLOCK_SIZE>>> (vec);
  checkCudaErrors(cudaDeviceSynchronize());
}

static __global__ void compare_vec (void* a_vec, void* b_vec, void* partial_result, int vol) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_in_block = threadIdx.x;
  int stride, last_stride;

  __shared__ double cache[BLOCK_SIZE];
  Complex* a_src = static_cast<Complex*>(a_vec);
  Complex* b_src = static_cast<Complex*>(b_vec);
  double temp(0);
  // cache[thread_in_block].clear2Zero();
  for (int i = thread_id; i < vol * Ns * Nc; i += vol) {
    temp += (a_src[i] - b_src[i]).norm2() * (a_src[i] - b_src[i]).norm2();
  }
  cache[thread_in_block] = temp;
  __syncthreads();

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
    *(static_cast<double*>(partial_result) + blockIdx.x) = sqrt(cache[0]);
  }
}
// when call this function, set gridDim to 1
static __global__ void reduce_partial_result(void* partial_result, int partial_length) {
  int thread_in_block = threadIdx.x;
  int stride, last_stride;

  double temp = 0;
  double* src = static_cast<double*>(partial_result);
  __shared__ double cache[BLOCK_SIZE];

  for (int i = thread_in_block; i < partial_length; i+= BLOCK_SIZE) {
    temp += src[i] * src[i];
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
    *(static_cast<double*>(partial_result) + thread_in_block) = sqrt(cache[0]);
  }
}

// device address
static void compare_vectors_kernel(void* a_vec, void* b_vec, void* partial_result, int vol) {
  int grid_size = vol / BLOCK_SIZE;
  int block_size = BLOCK_SIZE;

  compare_vec<<<grid_size, block_size>>>(a_vec, b_vec, partial_result, vol);
  checkCudaErrors(cudaDeviceSynchronize());

  reduce_partial_result<<<1, BLOCK_SIZE>>>(partial_result, grid_size);
  checkCudaErrors(cudaDeviceSynchronize());
}

// res_vec and partial_vec does not store any outside information, 
//    just to avoid repeating memory allocating
// bool if_converge(void* b_vec, void* x_vec, void* res_vec, void* partial_vec, void* gauge, QcuParam *param, void* temporary_vector) {

//   double diff;
//   bool res = false;
//   int vol = param->lattice_size[0] * param->lattice_size[1] * param->lattice_size[2] * param->lattice_size[3];

//   // D dagger D x = b?
//   MmV_one_round (res_vec, x_vec, gauge, param, temporary_vector);// Ap, A = d dagger d
//   // compare b with res_vec
//   compare_vectors_kernel(b_vec, res_vec, partial_vec, vol);
//   checkCudaErrors(cudaMemcpy(&diff, partial_vec, sizeof(double), cudaMemcpyDeviceToHost));  // partial_result[0]就是最终diff
//   if (diff < 1e-23) {
//     res = true;
//   }
// // #ifdef DEBUG
// //   printf("difference = %.9lf\n", diff);
// // #endif

//   return res;
// }

bool if_converge(void* r_vec, int vol) {
  bool res = false;
  double diff;
  Complex inner_prod(10000, 10000);
  Complex* d_inner_prod;
  checkCudaErrors(cudaMalloc(&d_inner_prod, sizeof(Complex)));

  mpi_comm->interprocess_inner_prod_barrier(r_vec, r_vec, d_inner_prod, vol);  // <r, r> --> d_numerator
  checkCudaErrors(cudaMemcpy(&inner_prod, d_inner_prod, sizeof(Complex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_inner_prod));

  // printf("difference = %.9lf\n", inner_prod.norm2());

  diff = inner_prod.norm2();
  if (diff < 1e-23) {
    res = true;
  }
// #ifdef DEBUG
//   printf("difference = %.9lf\n", diff);
// #endif
  return res;
}


double compare_two_vectors (void* a_vec, void* b_vec, void* partial_result, QcuParam *param) {
    double diff = 0;
    int vol = param->lattice_size[0] * param->lattice_size[1] * param->lattice_size[2] * param->lattice_size[3];
    // compare b with res_vec
    compare_vectors_kernel(a_vec, b_vec, partial_result, vol);
    checkCudaErrors(cudaMemcpy(&diff, partial_result, sizeof(double), cudaMemcpyDeviceToHost));  // partial_result[0]就是最终diff
    return diff;
}

double compare_two_vectors_cpu (void* d_a, void* d_b, int vol) {
  Complex* h_a;
  Complex* h_b;
  h_a = new Complex[vol * Ns * Nc];
  h_b = new Complex[vol * Ns * Nc];
  checkCudaErrors(cudaMemcpy(h_a, d_a, sizeof(Complex) * vol * Ns * Nc, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_b, d_b, sizeof(Complex) * vol * Ns * Nc, cudaMemcpyDeviceToHost));

  double diff;
  for (int i = 0; i < vol * Ns * Nc; i++) {
    diff += (h_a[i] - h_b[i]).norm2() * (h_a[i] - h_b[i]).norm2();
  }

  return sqrt(diff);
}

// b_vec doesnot modify, stores result
// r_vec is residential, doesn't take from outside func, only avoid malloc
// p_vec doesn't take from outside func, only avoid malloc
// x_vec stores new result, take zero or initial res from outside
//      temp_vec1, temp_vec2, res_vec, partial_vec: temporary 
// r_vec, p_vec, temp_vec1, temp_vec2, res_vec :  vol * Ns * Nc
// partial_vec : (vol * Ns * Nc + BLOCK_SIZE-1) / BLOCK_SIZE
bool cg(void* b_vec, void* r_vec, void* p_vec, void* x_vec, void* temp_vec1, void* temp_vec2, void* res_vec, void* partial_vec, void *gauge, QcuParam *param, void* temporary_vector) {
  int vol = param->lattice_size[0] * param->lattice_size[1] * param->lattice_size[2] *param->lattice_size[3];
  bool if_end = false;

  Complex alpha;
  Complex beta;
  Complex denominator;
  Complex numerator;
  Complex one(1,0);

  Complex* d_alpha;
  Complex* d_beta;
  Complex* d_denominator;
  Complex* d_numerator;
  Complex* d_one;


  checkCudaErrors(cudaMalloc(&d_alpha, sizeof(Complex)));
  checkCudaErrors(cudaMalloc(&d_beta, sizeof(Complex)));
  checkCudaErrors(cudaMalloc(&d_denominator, sizeof(Complex)));
  checkCudaErrors(cudaMalloc(&d_numerator, sizeof(Complex)));
  checkCudaErrors(cudaMalloc(&d_one, sizeof(Complex)));

  mpi_comm->interprocess_inner_prod_barrier(r_vec, r_vec, d_numerator, vol);  // <r, r> --> d_numerator

  MmV_one_round (temp_vec1, p_vec, gauge, param, temporary_vector); // Ap--->temp_vec1

  mpi_comm->interprocess_inner_prod_barrier (p_vec, temp_vec1, d_denominator, vol);  // <p, Ap>

  checkCudaErrors(cudaMemcpy(&numerator, d_numerator, sizeof(Complex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&denominator, d_denominator, sizeof(Complex), cudaMemcpyDeviceToHost));
  alpha = numerator / denominator;

  checkCudaErrors(cudaMemcpy(d_alpha, &alpha, sizeof(Complex), cudaMemcpyHostToDevice));
  mpi_comm->interprocess_saxpy_barrier(p_vec, x_vec, d_alpha, vol); // x = x + \alpha p

  checkCudaErrors(cudaMemcpy(temp_vec2, r_vec, sizeof(Complex) * vol * Ns * Nc, cudaMemcpyDeviceToDevice)); // copy r to temp_vec2  r'=r
  // r' <- r'- \alpha A p
  alpha = alpha * Complex(-1, 0);
  checkCudaErrors(cudaMemcpy(d_alpha, &alpha, sizeof(Complex), cudaMemcpyHostToDevice));
  mpi_comm->interprocess_saxpy_barrier(temp_vec1, temp_vec2, d_alpha, vol); // temp_vec1 = Ap, r'=r'-\alpha Ap------>temp_vec2


  // : if converge, return x
  if (if_converge(r_vec, vol)) {
    if_end = true;
    goto cg_free;
  }

  // <r, r> is in numerator
  mpi_comm->interprocess_inner_prod_barrier(temp_vec2, temp_vec2, d_denominator, vol);  // <r', r'>
  checkCudaErrors(cudaMemcpy(&denominator, d_denominator, sizeof(Complex), cudaMemcpyDeviceToHost));

  beta = denominator / numerator;
  checkCudaErrors(cudaMemcpy(d_beta, &beta, sizeof(Complex), cudaMemcpyHostToDevice));
  // p = r' + \beta p

  gpu_sclar_multiply_vector (p_vec, d_beta, vol); // p_vec = \beta p_vec


  one = Complex(1, 0);
  checkCudaErrors(cudaMemcpy(d_one, &one, sizeof(Complex), cudaMemcpyHostToDevice));
  mpi_comm->interprocess_saxpy_barrier(temp_vec2, p_vec, d_one, vol); // p <-- r' + \beta p

  checkCudaErrors(cudaMemcpy(r_vec, temp_vec2, sizeof(Complex) * vol * Ns * Nc, cudaMemcpyDeviceToDevice));  // r <--- r'


cg_free:
  checkCudaErrors(cudaFree(d_alpha));
  checkCudaErrors(cudaFree(d_beta));
  checkCudaErrors(cudaFree(d_denominator));
  checkCudaErrors(cudaFree(d_numerator));
  checkCudaErrors(cudaFree(d_one));

  return if_end;
}

// CG inverter
// Ax = b , want x
void cg_inverter(void* b_vector, void* x_vector, void *gauge, QcuParam *param) {

  // Dslash x = b  ----->  Dslash^\dagger Dslash x = Dslash^\dagger b
  int Lx = param->lattice_size[0];
  int Ly = param->lattice_size[1];
  int Lz = param->lattice_size[2];
  int Lt = param->lattice_size[3];
  int vol = Lx * Ly * Lz * Lt;
  int max_iterator = vol;
  Complex coeff(-1, 0);
  void* d_new_b;
  bool cg_res = false;
  void* p_vec;
  void* r_vec;
  void* temp_vec1;
  void* temp_vec2;
  void* res_vec;
  void* d_coeff;
  void* temporary_vector;

// #ifdef DEBUG
//   void* debug_ptr;
//   void* debug_b_ptr;
//   checkCudaErrors(cudaMalloc(&debug_ptr, sizeof(Complex) * vol * Ns * Nc));
//   checkCudaErrors(cudaMalloc(&debug_b_ptr, sizeof(Complex) * vol * Ns * Nc));

//   // use this to debug
//   fullCloverDslashOneRound (debug_b_ptr, b_vector, gauge, param, 0);  // b <- Dslash x
//   printf(CLR"");
// #endif

  checkCudaErrors(cudaMalloc(&d_new_b, sizeof(Complex) * vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&p_vec, sizeof(Complex) * vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&r_vec, sizeof(Complex) * vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&temp_vec1, sizeof(Complex) * vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&temp_vec2, sizeof(Complex) * vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&res_vec, sizeof(Complex) * vol * Ns * Nc));
  checkCudaErrors(cudaMalloc(&d_coeff, sizeof(Complex)));
  checkCudaErrors(cudaMalloc(&temporary_vector, sizeof(Complex) * vol * Ns * Nc));


  checkCudaErrors(cudaMemcpy(d_coeff, &coeff, sizeof(Complex), cudaMemcpyHostToDevice));


  void* partial_result_vector;  // use this to reduce, the size is (vol * Ns * Nc * Ns * Nc + BLOCK_SIZE - 1) / BLOCK_SIZE
  int partial_result_length = (vol * Ns * Nc * Ns * Nc + BLOCK_SIZE - 1) / BLOCK_SIZE;
  checkCudaErrors(cudaMalloc(&partial_result_vector, sizeof(Complex) * partial_result_length));


  clearVector(x_vector, vol);   // x <- 0
  // fullCloverDslashOneRound (d_new_b, b_vector, gauge, param, 1);  // new_b <- Dslash_dagger b
  checkCudaErrors(cudaMemcpy(temporary_vector, b_vector, sizeof(Complex) * vol * Ns * Nc, cudaMemcpyDeviceToDevice));
  invertCloverDslash (temporary_vector, temporary_vector, gauge, param, 0);
  // dagger
  newFullCloverDslashOneRound (d_new_b, temporary_vector, gauge, param, 1);


  // r = b - Ax
  MmV_one_round (temp_vec1, x_vector, gauge, param, temporary_vector);// D dagger D x ---> temp_vec1 (Ax)
  coeff = Complex(-1, 0);
  checkCudaErrors(cudaMemcpy(d_coeff, &coeff, sizeof(Complex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(r_vec, d_new_b, sizeof(Complex) * vol * Ns * Nc, cudaMemcpyDeviceToDevice));     // r <- new_b
  // r = b - Ax
  mpi_comm->interprocess_saxpy_barrier(temp_vec1, r_vec, d_coeff, vol); // r <-- newb - Ax
  // r = b - Ax end

  checkCudaErrors(cudaMemcpy(p_vec, r_vec, sizeof(Complex) * vol * Ns * Nc, cudaMemcpyDeviceToDevice));     // p = r

  //  if converge, return x
  if (if_converge(r_vec, vol)) {
    printf("cg success!!!\n");
    goto cg_inverter_free;
  }

  // iterate
  for (int i = 0; i < max_iterator; i++) {
// #ifdef DEBUG
//     cg_res = cg(d_new_b, r_vec, p_vec, x_vector, temp_vec1, temp_vec2, res_vec, partial_result_vector, gauge, param, debug_ptr);
// #else
    cg_res = cg(d_new_b, r_vec, p_vec, x_vector, temp_vec1, temp_vec2, res_vec, partial_result_vector, gauge, param, temporary_vector);
// #endif
    if (cg_res) {
      printf("number of iteration %d\n", i + 1);
      printf("cg success!!!\n");
      break;
    }
  }
  if (!cg_res) {
    printf("cg_failed!!!\n");
  }



cg_inverter_free:
  checkCudaErrors(cudaFree(d_new_b));

  checkCudaErrors(cudaFree(p_vec));
  checkCudaErrors(cudaFree(r_vec));
  checkCudaErrors(cudaFree(temp_vec1));
  checkCudaErrors(cudaFree(temp_vec2));
  // checkCudaErrors(cudaFree(temp_vec3));
  checkCudaErrors(cudaFree(res_vec));
  checkCudaErrors(cudaFree(partial_result_vector));
  checkCudaErrors(cudaFree(d_coeff));
  checkCudaErrors(cudaFree(temporary_vector));
// #ifdef DEBUG
//   checkCudaErrors(cudaFree(debug_ptr));
//   checkCudaErrors(cudaFree(debug_b_ptr));
// #endif
}

