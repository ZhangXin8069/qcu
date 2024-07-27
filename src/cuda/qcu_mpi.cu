#include "../../include/qcu.h"

__global__ void wilson_bistabcg_part_dot(void *device_dot_tmp,
                                         void *device_val0, void *device_val1) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
    LatticeComplex *dot_tmp = static_cast<LatticeComplex *>(device_dot_tmp);
    LatticeComplex *val0 = static_cast<LatticeComplex *>(device_val0);
    LatticeComplex *val1 = static_cast<LatticeComplex *>(device_val1);
    dot_tmp[idx].real = 0.0;
    dot_tmp[idx].imag = 0.0;
    for (int i = 0; i < _LAT_SC_; ++i) {
      dot_tmp[idx] += val0[idx * _LAT_SC_ + i].conj() * val1[idx * _LAT_SC_ + i];
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
    for (int i = 0; i < _LAT_SC_; ++i) {
      latt_tmp0[idx * _LAT_SC_ + i] =
          val0[idx * _LAT_SC_ + i] - val1[idx * _LAT_SC_ + i];
    }
  }
}