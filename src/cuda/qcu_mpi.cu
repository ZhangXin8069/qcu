#pragma optimize(5)
#include "../../include/qcu.h"

__global__ void wilson_bistabcg_part_dot(void *device_dot_tmp,
                                         void *device_val0, void *device_val1) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
    LatticeComplex *device_dot_tmp =
        static_cast<LatticeComplex *>(device_dot_tmp);
    LatticeComplex *val0 = static_cast<LatticeComplex *>(device_val0);
    LatticeComplex *val1 = static_cast<LatticeComplex *>(device_val1);
    device_dot_tmp[idx].real = 0.0;
    device_dot_tmp[idx].imag = 0.0;
    for (int i = 0; i < LAT_SC; ++i) {
      device_dot_tmp[idx] +=
          val0[idx * LAT_SC + i].conj() * val1[idx * LAT_SC + i];
    }
  }
}

__global__ void wilson_bistabcg_part_cut(void *device_latt_tmp0,
                                         void *device_val0, void *device_val1) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
    LatticeComplex *latt_tmp0 = static_cast<LatticeComplex *>(device_latt_tmp0);
    LatticeComplex *val0 = static_cast<LatticeComplex *>(device_val0);
    LatticeComplex *val1 = static_cast<LatticeComplex *>(device_val1);
    for (int i = 0; i < LAT_SC; ++i) {
      latt_tmp0[idx * LAT_SC + i] =
          val0[idx * LAT_SC + i] - val1[idx * LAT_SC + i];
    }
  }
}