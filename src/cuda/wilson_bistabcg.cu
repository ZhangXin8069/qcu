#pragma optimize(5)
#include "../../include/qcu.h"
#ifdef WILSON_BISTABCG

__global__ void wilson_bistabcg_give_b_e(void *device_b_e, void *device_ans_e,
                                         void *device_latt_tmp0, double kappa) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
    LatticeComplex *b_e = static_cast<LatticeComplex *>(device_b_e);
    LatticeComplex *ans_e = static_cast<LatticeComplex *>(device_ans_e);
    LatticeComplex *latt_tmp0 = static_cast<LatticeComplex *>(device_latt_tmp0);
    for (int i = 0; i < LAT_SC; ++i) {
      b_e[idx * LAT_SC + i] =
          ans_e[idx * LAT_SC + i] -
          latt_tmp0[idx * LAT_SC + i] * kappa; // b_e=ans_e-kappa*D_eo(ans_o)
    }
  }
}

__global__ void wilson_bistabcg_give_b_o(void *device_b_o, void *device_ans_o,
                                         void *device_latt_tmp1, double kappa) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
    LatticeComplex *b_o = static_cast<LatticeComplex *>(device_b_o);
    LatticeComplex *ans_o = static_cast<LatticeComplex *>(device_ans_o);
    LatticeComplex *latt_tmp1 = static_cast<LatticeComplex *>(device_latt_tmp1);
    for (int i = 0; i < LAT_SC; ++i) {
      b_o[idx * LAT_SC + i] =
          ans_o[idx * LAT_SC + i] -
          latt_tmp1[idx * LAT_SC + i] * kappa; // b_o=ans_o-kappa*D_oe(ans_e)
    }
  }
}

__global__ void wilson_bistabcg_give_b__0(void *device_b__o, void *device_b_o,
                                          void *device_latt_tmp0,
                                          double kappa) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
    LatticeComplex *b__o = static_cast<LatticeComplex *>(device_b__o);
    LatticeComplex *b_o = static_cast<LatticeComplex *>(device_b_o);
    LatticeComplex *latt_tmp0 = static_cast<LatticeComplex *>(device_latt_tmp0);
    for (int i = 0; i < LAT_SC; ++i) {
      b__o[idx * LAT_SC + i] =
          b_o[idx * LAT_SC + i] +
          latt_tmp0[idx * LAT_SC + i] * kappa; // b__o=b_o+kappa*D_oe(b_e)
    }
  }
}

__global__ void wilson_bistabcg_give_dest_o(void *device_dest_o,
                                            void *device_src_o,
                                            void *device_latt_tmp1,
                                            double kappa) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
    LatticeComplex *dest_o = static_cast<LatticeComplex *>(device_dest_o);
    LatticeComplex *src_o = static_cast<LatticeComplex *>(device_src_o);
    LatticeComplex *latt_tmp1 = static_cast<LatticeComplex *>(device_latt_tmp1);
    for (int i = 0; i < LAT_SC; ++i) {
      dest_o[idx * LAT_SC + i] =
          src_o[idx * LAT_SC + i] - latt_tmp1[idx * LAT_SC + i] * kappa *
                                        kappa; // dest_o=ans_o-kappa^2*tmp1
    }
  }
}

__global__ void wilson_bistabcg_give_rr(void *device_r, void *device_b__o,
                                        void *device_r_tilde) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
    LatticeComplex *r = static_cast<LatticeComplex *>(device_r);
    LatticeComplex *b__o = static_cast<LatticeComplex *>(device_b__o);
    LatticeComplex *r_tilde = static_cast<LatticeComplex *>(device_r_tilde);
    for (int i = 0; i < LAT_SC; ++i) {
      r[idx * LAT_SC + i] = b__o[idx * LAT_SC + i] - r[idx * LAT_SC + i];
      r_tilde[idx * LAT_SC + i] = r[idx * LAT_SC + i];
    }
  }
}

__global__ void wilson_bistabcg_give_p(void *device_p, void *device_r,
                                       void *device_v, LatticeComplex omega,
                                       LatticeComplex beta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
    LatticeComplex *p = static_cast<LatticeComplex *>(device_p);
    LatticeComplex *r = static_cast<LatticeComplex *>(device_r);
    LatticeComplex *v = static_cast<LatticeComplex *>(device_v);
    for (int i = 0; i < LAT_SC; ++i) {
      p[idx * LAT_SC + i] =
          r[idx * LAT_SC + i] +
          (p[idx * LAT_SC + i] - v[idx * LAT_SC + i] * omega) * beta;
    }
  }
}

__global__ void wilson_bistabcg_give_s(void *device_s, void *device_r,
                                       void *device_v, LatticeComplex alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
    LatticeComplex *s = static_cast<LatticeComplex *>(device_s);
    LatticeComplex *r = static_cast<LatticeComplex *>(device_r);
    LatticeComplex *v = static_cast<LatticeComplex *>(device_v);
    for (int i = 0; i < LAT_SC; ++i) {
      s[idx * LAT_SC + i] = r[idx * LAT_SC + i] - v[idx * LAT_SC + i] * alpha;
    }
  }
}

__global__ void wilson_bistabcg_give_x_o(void *device_x_o, void *device_p,
                                         void *device_s, LatticeComplex alpha,
                                         LatticeComplex omega) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
    LatticeComplex *x_o = static_cast<LatticeComplex *>(device_x_o);
    LatticeComplex *p = static_cast<LatticeComplex *>(device_p);
    LatticeComplex *s = static_cast<LatticeComplex *>(device_s);
    for (int i = 0; i < LAT_SC; ++i) {
      x_o[idx * LAT_SC + i] = x_o[idx * LAT_SC + i] +
                              p[idx * LAT_SC + i] * alpha +
                              s[idx * LAT_SC + i] * omega;
    }
  }
}

__global__ void wilson_bistabcg_give_r(void *device_r, void *device_s,
                                       void *device_tt, LatticeComplex omega) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lat_4dim = gridDim.x * blockDim.x;
  if (idx < lat_4dim) {
    LatticeComplex *r = static_cast<LatticeComplex *>(device_r);
    LatticeComplex *s = static_cast<LatticeComplex *>(device_s);
    LatticeComplex *t = static_cast<LatticeComplex *>(device_tt);
    for (int i = 0; i < LAT_SC; ++i) {
      r[idx * LAT_SC + i] = s[idx * LAT_SC + i] - t[idx * LAT_SC + i] * omega;
    }
  }
}

#endif