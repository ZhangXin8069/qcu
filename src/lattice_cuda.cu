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

__global__ void fermion_dot(void *device_vec0, void *device_vec1,
                            void *device_vals, const int vals_index) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_vec0 = static_cast<LatticeComplex *>(device_vec0);
  LatticeComplex *origin_vec1 = static_cast<LatticeComplex *>(device_vec1);
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex vec0[_LAT_SC_];
  LatticeComplex vec1[_LAT_SC_];
  give_ptr(vec0, origin_vec0, _LAT_SC_);
  give_ptr(vec1, origin_vec1, _LAT_SC_);
  LatticeComplex _(0.0, 0.0);
  for (int i = 0; i < _LAT_SC_; ++i) {
    _ += (vec0[idx * _LAT_SC_ + i].conj() * vec1[idx * _LAT_SC_ + i]);
  }
  origin_vals[vals_index] += _;
}

__global__ void fermion_diff(void *device_vec0, void *device_vec1,
                             void *device_vals, const int vals_index) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_vec0 = static_cast<LatticeComplex *>(device_vec0);
  LatticeComplex *origin_vec1 = static_cast<LatticeComplex *>(device_vec1);
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex vec0[_LAT_SC_];
  LatticeComplex vec1[_LAT_SC_];
  LatticeComplex vec[_LAT_SC_];
  give_ptr(vec0, origin_vec0, _LAT_SC_);
  give_ptr(vec1, origin_vec1, _LAT_SC_);
  LatticeComplex _(0.0, 0.0);
  for (int i = 0; i < _LAT_SC_; ++i) {
    vec[i] = (vec0[idx * _LAT_SC_ + i] - vec1[idx * _LAT_SC_ + i]);
  }
  for (int i = 0; i < _LAT_SC_; ++i) {
    _ += (vec[i].conj() * vec[i]) / (vec1[i].conj() * vec1[i]);
  }
  origin_vals[vals_index] += _;
}

__global__ void bistabcg_add(void *device_vec0, void *device_vec1,
                             void *device_vals, const int vals_index) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_vec0 = static_cast<LatticeComplex *>(device_vec0);
  LatticeComplex *origin_vec1 = static_cast<LatticeComplex *>(device_vec1);
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex vec0[_LAT_SC_];
  LatticeComplex vec1[_LAT_SC_];
  give_ptr(vec0, origin_vec0, _LAT_SC_);
  give_ptr(vec1, origin_vec1, _LAT_SC_);
  LatticeComplex _(0.0, 0.0);
  for (int i = 0; i < _LAT_SC_; ++i) {
    _ += (vec0[idx * _LAT_SC_ + i] + vec1[idx * _LAT_SC_ + i]);
  }
  origin_vals[vals_index] += _;
}

__global__ void fermion_subt(void *device_vec0, void *device_vec1,
                             void *device_vals, const int vals_index) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_vec0 = static_cast<LatticeComplex *>(device_vec0);
  LatticeComplex *origin_vec1 = static_cast<LatticeComplex *>(device_vec1);
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex vec0[_LAT_SC_];
  LatticeComplex vec1[_LAT_SC_];
  give_ptr(vec0, origin_vec0, _LAT_SC_);
  give_ptr(vec1, origin_vec1, _LAT_SC_);
  LatticeComplex _(0.0, 0.0);
  for (int i = 0; i < _LAT_SC_; ++i) {
    _ += (vec0[idx * _LAT_SC_ + i] - vec1[idx * _LAT_SC_ + i]);
  }
  origin_vals[vals_index] += _;
}

__global__ void fermion_mult(void *device_vec0, void *device_vec1,
                             void *device_vals, const int vals_index) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_vec0 = static_cast<LatticeComplex *>(device_vec0);
  LatticeComplex *origin_vec1 = static_cast<LatticeComplex *>(device_vec1);
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex vec0[_LAT_SC_];
  LatticeComplex vec1[_LAT_SC_];
  give_ptr(vec0, origin_vec0, _LAT_SC_);
  give_ptr(vec1, origin_vec1, _LAT_SC_);
  LatticeComplex _(0.0, 0.0);
  for (int i = 0; i < _LAT_SC_; ++i) {
    _ += (vec0[idx * _LAT_SC_ + i] * vec1[idx * _LAT_SC_ + i]);
  }
  origin_vals[vals_index] += _;
}

__global__ void fermion_divi(void *device_vec0, void *device_vec1,
                             void *device_vals, const int vals_index) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_vec0 = static_cast<LatticeComplex *>(device_vec0);
  LatticeComplex *origin_vec1 = static_cast<LatticeComplex *>(device_vec1);
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex vec0[_LAT_SC_];
  LatticeComplex vec1[_LAT_SC_];
  give_ptr(vec0, origin_vec0, _LAT_SC_);
  give_ptr(vec1, origin_vec1, _LAT_SC_);
  LatticeComplex _(0.0, 0.0);
  for (int i = 0; i < _LAT_SC_; ++i) {
    _ += (vec0[idx * _LAT_SC_ + i] / vec1[idx * _LAT_SC_ + i]);
  }
  origin_vals[vals_index] += _;
}

#endif