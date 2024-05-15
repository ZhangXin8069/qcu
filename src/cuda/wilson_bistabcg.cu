#pragma optimize(5)
#include "../../include/qcu.h"
#ifdef WILSON_BISTABCG

// __global__ void wilson_bistabcg_func0(void *device_b_e, void *device_ans_e,
//                                       void *device_latt_tmp0, double kappa) {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   LatticeComplex *b_e =
//       ((static_cast<LatticeComplex *>(device_b_e)) + index * LAT_SC);
//   LatticeComplex *ans_e =
//       ((static_cast<LatticeComplex *>(device_ans_e)) + index * LAT_SC);
//   LatticeComplex *latt_tmp0 =
//       ((static_cast<LatticeComplex *>(device_latt_tmp0)) + index * LAT_SC);
//   for (int i = 0; i < LAT_SC; i++) {
//     b_e[i] =
//         ans_e[i] - device_latt_tmp0[i] * kappa; // b_e=anw_e-kappa*D_eo(ans_o)
//   }
// }

// __global__ void wilson_bistabcg_func1(void *device_b_e, void *device_ans_e,
//                                       void *device_latt_tmp0, double kappa) {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   LatticeComplex *b_e =
//       ((static_cast<LatticeComplex *>(device_b_e)) + index * LAT_SC);
//   LatticeComplex *ans_e =
//       ((static_cast<LatticeComplex *>(device_ans_e)) + index * LAT_SC);
//   LatticeComplex *latt_tmp0 =
//       ((static_cast<LatticeComplex *>(device_latt_tmp0)) + index * LAT_SC);
//   for (int i = 0; i < LAT_SC; i++) {
//     b_e[i] =
//         ans_e[i] - device_latt_tmp0[i] * kappa; // b_e=anw_e-kappa*D_eo(ans_o)
//   }
// }

#endif