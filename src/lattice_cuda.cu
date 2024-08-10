#include "../include/qcu.h"
#ifdef LATTICE_CUDA

__global__ void give_random_value(void *device_random_value,
                                  unsigned long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *random_value =
      static_cast<LatticeComplex *>(device_random_value);
  curandState state_real, state_imag;
  curand_init(seed, idx, 0, &state_real);
  curand_init(seed, idx, 1, &state_imag);
  for (int i = 0; i < _LAT_SC_; ++i) {
    random_value[idx * _LAT_SC_ + i].real = curand_uniform(&state_real);
    random_value[idx * _LAT_SC_ + i].imag = curand_uniform(&state_imag);
  }
}
__global__ void give_custom_value(void *device_custom_value, double real,
                                  double imag) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *custom_value =
      static_cast<LatticeComplex *>(device_custom_value);
  for (int i = 0; i < _LAT_SC_; ++i) {
    custom_value[idx * _LAT_SC_ + i].real = real;
    custom_value[idx * _LAT_SC_ + i].imag = imag;
  }
}
__global__ void give_1zero(void *device_vals, const int vals_index) {
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex _(0.0, 0.0);
  origin_vals[vals_index] = _;
}
__global__ void give_1one(void *device_vals, const int vals_index) {
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex _(1.0, 0.0);
  origin_vals[vals_index] = _;
}
__global__ void give_1custom_value(void *device_vals, const int vals_index,
                                   double real, double imag) {
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex _(real, imag);
  origin_vals[vals_index] = _;
}
__global__ void give_1axpy(void *device_vals, const int valA_index,
                           const int valB_index, double real, double imag) {
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex _(real, imag);
  origin_vals[valB_index] =
      origin_vals[valB_index] + origin_vals[valA_index] * _;
}
void LatticeAxpy(void *device_A, void *device_B, void *device_val, int size,
                 cublasHandle_t cublasH) {
  // dest(B) = B + alpha*A
  CUBLAS_CHECK(cublasAxpyEx(
      cublasH, size, device_val, traits<data_type>::cuda_data_type, device_A,
      traits<data_type>::cuda_data_type, 1, device_B,
      traits<data_type>::cuda_data_type, 1, traits<data_type>::cuda_data_type));
}
void LatticeAxpy(void *device_A, void *device_B, double real, double imag,
                 int size, cublasHandle_t cublasH) {
  // dest(B) = B + alpha*A
  LatticeComplex _(real, imag);
  CUBLAS_CHECK(cublasAxpyEx(
      cublasH, size, &_, traits<data_type>::cuda_data_type, device_A,
      traits<data_type>::cuda_data_type, 1, device_B,
      traits<data_type>::cuda_data_type, 1, traits<data_type>::cuda_data_type));
}
void LatticeCopy(void *device_A, void *device_B, int size,
                 cublasHandle_t cublasH) {
  CUBLAS_CHECK(cublasDswap(cublasH, size * sizeof(data_type) / sizeof(double),
                           (double *)device_A, 1, (double *)device_B, 1));
}
void LatticeSwap(void *device_A, void *device_B, int size,
                 cublasHandle_t cublasH) {
  CUBLAS_CHECK(cublasDcopy(cublasH, size * sizeof(data_type) / sizeof(double),
                           (double *)device_A, 1, (double *)device_B, 1));
}
void LatticeDot(void *device_A, void *device_B, void *device_val, int size,
                cublasHandle_t cublasH) {
  // dest(val) = dot(A,B)
  CUBLAS_CHECK(cublasDotcEx(
      cublasH, size, device_A, traits<data_type>::cuda_data_type, 1, device_B,
      traits<data_type>::cuda_data_type, 1, device_val,
      traits<data_type>::cuda_data_type, traits<data_type>::cuda_data_type));
}
#endif