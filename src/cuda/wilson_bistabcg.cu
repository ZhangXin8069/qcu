#pragma optimize(5)
#include "../../include/qcu.h"
#ifdef WILSON_BISTABCG

//   for (int i = 0; i < lat_4dim12; i++) {
//     b_e[i] =
//         ans_e[i] - device_latt_tmp0[i] * kappa; // b_e=anw_e-kappa*D_eo(ans_o)
//   }
__global__ void wilson_bistabcg_func0(void *device_dest, int device_lat_x,
                                         const int lat_4dim,
                                         const int device_lat_z) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
lat_4dim 
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) +
       t * lat_z * lat_y * lat_x * 12 + z * lat_y * lat_x * 12 +
       y * lat_x * 12 + x * 12);
  host_give_value(origin_dest, zero, 12);
}

#endif