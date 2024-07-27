#include "../../include/qcu.h"

__global__ void give_random_value(void *device_random_value,
                                  unsigned long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
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
}

__global__ void give_custom_value(void *device_custom_value, double real,
                                  double imag) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
    LatticeComplex *custom_value =
        static_cast<LatticeComplex *>(device_custom_value);
    for (int i = 0; i < _LAT_SC_; ++i) {
      custom_value[idx * _LAT_SC_ + i].real = real;
      custom_value[idx * _LAT_SC_ + i].imag = imag;
    }
  }
}