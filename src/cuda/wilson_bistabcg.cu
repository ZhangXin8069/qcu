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
    for (int i = 0; i < _LAT_SC_; ++i) {
      b_e[idx * _LAT_SC_ + i] =
          ans_e[idx * _LAT_SC_ + i] -
          latt_tmp0[idx * _LAT_SC_ + i] * kappa; // b_e=ans_e-kappa*D_eo(ans_o)
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
    for (int i = 0; i < _LAT_SC_; ++i) {
      b_o[idx * _LAT_SC_ + i] =
          ans_o[idx * _LAT_SC_ + i] -
          latt_tmp1[idx * _LAT_SC_ + i] * kappa; // b_o=ans_o-kappa*D_oe(ans_e)
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
    for (int i = 0; i < _LAT_SC_; ++i) {
      b__o[idx * _LAT_SC_ + i] =
          b_o[idx * _LAT_SC_ + i] +
          latt_tmp0[idx * _LAT_SC_ + i] * kappa; // b__o=b_o+kappa*D_oe(b_e)
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
    for (int i = 0; i < _LAT_SC_; ++i) {
      dest_o[idx * _LAT_SC_ + i] =
          src_o[idx * _LAT_SC_ + i] - latt_tmp1[idx * _LAT_SC_ + i] * kappa *
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
    for (int i = 0; i < _LAT_SC_; ++i) {
      r[idx * _LAT_SC_ + i] = b__o[idx * _LAT_SC_ + i] - r[idx * _LAT_SC_ + i];
      r_tilde[idx * _LAT_SC_ + i] = r[idx * _LAT_SC_ + i];
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
    for (int i = 0; i < _LAT_SC_; ++i) {
      p[idx * _LAT_SC_ + i] =
          r[idx * _LAT_SC_ + i] +
          (p[idx * _LAT_SC_ + i] - v[idx * _LAT_SC_ + i] * omega) * beta;
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
    for (int i = 0; i < _LAT_SC_; ++i) {
      s[idx * _LAT_SC_ + i] = r[idx * _LAT_SC_ + i] - v[idx * _LAT_SC_ + i] * alpha;
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
    for (int i = 0; i < _LAT_SC_; ++i) {
      x_o[idx * _LAT_SC_ + i] = x_o[idx * _LAT_SC_ + i] +
                              p[idx * _LAT_SC_ + i] * alpha +
                              s[idx * _LAT_SC_ + i] * omega;
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
    for (int i = 0; i < _LAT_SC_; ++i) {
      r[idx * _LAT_SC_ + i] = s[idx * _LAT_SC_ + i] - t[idx * _LAT_SC_ + i] * omega;
    }
  }
}

#endif