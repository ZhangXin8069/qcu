#pragma optimize(5)
#include "../../include/qcu.h"
#ifdef WILSON_BISTABCG

__global__ void wilson_bistabcg_give_b_e(void *device_b_e, void *device_ans_e,
                                      void *device_latt_tmp0, double kappa) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
    LatticeComplex *b_e =
        static_cast<LatticeComplex *>(device_b_e);
    LatticeComplex *ans_e =
        static_cast<LatticeComplex *>(device_ans_e);
    LatticeComplex *latt_tmp0 =
        static_cast<LatticeComplex *>(device_latt_tmp0);
    for (int i = 0; i < LAT_SC; ++i) {
      b_e[idx * LAT_SC + i] = ans_e[idx * LAT_SC + i] - device_latt_tmp0[idx * LAT_SC + i] * kappa; // b_e=ans_e-kappa*D_eo(ans_o)
    }
  }
}

__global__ void wilson_bistabcg_give_b_o(void *device_b_o, void *device_ans_o,
                                      void *device_latt_tmp1, double kappa) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
    LatticeComplex *b_o =
        static_cast<LatticeComplex *>(device_b_o);
    LatticeComplex *ans_o =
        static_cast<LatticeComplex *>(device_ans_o);
    LatticeComplex *latt_tmp1 =
        static_cast<LatticeComplex *>(device_latt_tmp1);
    for (int i = 0; i < LAT_SC; ++i) {
      b_o[idx * LAT_SC + i]=ans_o[idx * LAT_SC + i] - device_latt_tmp1[idx * LAT_SC + i] * kappa; // b_o=ans_o-kappa*D_oe(ans_e)
    }
  }
}

#endif