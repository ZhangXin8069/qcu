#include "../include/qcu.h"
#ifdef BISTABCG
__global__ void bistabcg_give_1beta(void *device_vals) {
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex rho_prev;
  rho_prev = origin_vals[_rho_prev_];
  LatticeComplex rho;
  rho = origin_vals[_rho_];
  LatticeComplex alpha;
  alpha = origin_vals[_alpha_];
  LatticeComplex beta;
  beta = origin_vals[_beta_];
  LatticeComplex omega;
  omega = origin_vals[_omega_];
  beta = (rho / rho_prev) * (alpha / omega);
  origin_vals[_beta_] = beta;
}

__global__ void bistabcg_give_1rho_prev(void *device_vals) {
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex rho;
  rho = origin_vals[_rho_];
  origin_vals[_rho_prev_] = rho;
}

__global__ void bistabcg_give_1alpha(void *device_vals) {
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex rho;
  rho = origin_vals[_rho_];
  LatticeComplex tmp0;
  tmp0 = origin_vals[_tmp0_];
  origin_vals[_alpha_] = rho / tmp0;
}

__global__ void bistabcg_give_1omega(void *device_vals) {
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex tmp0;
  tmp0 = origin_vals[_tmp0_];
  LatticeComplex tmp1;
  tmp1 = origin_vals[_tmp1_];
  origin_vals[_omega_] = tmp0 / tmp1;
}

__global__ void bistabcg_give_b_e(void *device_b_e, void *device_ans_e,
                                  void *device_vec0, double kappa) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *b_e = static_cast<LatticeComplex *>(device_b_e);
  LatticeComplex *ans_e = static_cast<LatticeComplex *>(device_ans_e);
  LatticeComplex *latt_tmp0 = static_cast<LatticeComplex *>(device_vec0);
  for (int i = 0; i < _LAT_SC_; ++i) {
    b_e[idx * _LAT_SC_ + i] =
        ans_e[idx * _LAT_SC_ + i] -
        latt_tmp0[idx * _LAT_SC_ + i] * kappa; // b_e=ans_e-kappa*D_eo(ans_o)
  }
}

__global__ void bistabcg_give_b_o(void *device_b_o, void *device_ans_o,
                                  void *device_vec1, double kappa) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *b_o = static_cast<LatticeComplex *>(device_b_o);
  LatticeComplex *ans_o = static_cast<LatticeComplex *>(device_ans_o);
  LatticeComplex *latt_tmp1 = static_cast<LatticeComplex *>(device_vec1);
  for (int i = 0; i < _LAT_SC_; ++i) {
    b_o[idx * _LAT_SC_ + i] =
        ans_o[idx * _LAT_SC_ + i] -
        latt_tmp1[idx * _LAT_SC_ + i] * kappa; // b_o=ans_o-kappa*D_oe(ans_e)
  }
}

__global__ void bistabcg_give_b__0(void *device_b__o, void *device_b_o,
                                   void *device_vec0, double kappa) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *b__o = static_cast<LatticeComplex *>(device_b__o);
  LatticeComplex *b_o = static_cast<LatticeComplex *>(device_b_o);
  LatticeComplex *latt_tmp0 = static_cast<LatticeComplex *>(device_vec0);
  for (int i = 0; i < _LAT_SC_; ++i) {
    b__o[idx * _LAT_SC_ + i] =
        b_o[idx * _LAT_SC_ + i] +
        latt_tmp0[idx * _LAT_SC_ + i] * kappa; // b__o=b_o+kappa*D_oe(b_e)
  }
}

__global__ void bistabcg_give_dest_o(void *device_dest_o, void *device_src_o,
                                     void *device_vec1, double kappa) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *dest_o = static_cast<LatticeComplex *>(device_dest_o);
  LatticeComplex *src_o = static_cast<LatticeComplex *>(device_src_o);
  LatticeComplex *latt_tmp1 = static_cast<LatticeComplex *>(device_vec1);
  for (int i = 0; i < _LAT_SC_; ++i) {
    dest_o[idx * _LAT_SC_ + i] =
        src_o[idx * _LAT_SC_ + i] - latt_tmp1[idx * _LAT_SC_ + i] * kappa *
                                        kappa; // dest_o=ans_o-kappa^2*tmp1
  }
}

__global__ void bistabcg_give_rr(void *device_r, void *device_b__o,
                                 void *device_r_tilde) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *r = static_cast<LatticeComplex *>(device_r);
  LatticeComplex *b__o = static_cast<LatticeComplex *>(device_b__o);
  LatticeComplex *r_tilde = static_cast<LatticeComplex *>(device_r_tilde);
  for (int i = 0; i < _LAT_SC_; ++i) {
    r[idx * _LAT_SC_ + i] = b__o[idx * _LAT_SC_ + i] - r[idx * _LAT_SC_ + i];
    r_tilde[idx * _LAT_SC_ + i] = r[idx * _LAT_SC_ + i];
  }
}

__global__ void bistabcg_give_p(void *device_p, void *device_r, void *device_v,
                                void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *p = static_cast<LatticeComplex *>(device_p);
  LatticeComplex *r = static_cast<LatticeComplex *>(device_r);
  LatticeComplex *v = static_cast<LatticeComplex *>(device_v);
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex beta;
  beta = origin_vals[_beta_];
  LatticeComplex omega;
  omega = origin_vals[_omega_];
  for (int i = 0; i < _LAT_SC_; ++i) {
    p[idx * _LAT_SC_ + i] =
        r[idx * _LAT_SC_ + i] +
        (p[idx * _LAT_SC_ + i] - v[idx * _LAT_SC_ + i] * omega) * beta;
  }
}

__global__ void bistabcg_give_s(void *device_s, void *device_r, void *device_v,
                                void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *s = static_cast<LatticeComplex *>(device_s);
  LatticeComplex *r = static_cast<LatticeComplex *>(device_r);
  LatticeComplex *v = static_cast<LatticeComplex *>(device_v);
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex alpha;
  alpha = origin_vals[_alpha_];
  for (int i = 0; i < _LAT_SC_; ++i) {
    s[idx * _LAT_SC_ + i] =
        r[idx * _LAT_SC_ + i] - v[idx * _LAT_SC_ + i] * alpha;
  }
}

__global__ void bistabcg_give_x_o(void *device_x_o, void *device_p,
                                  void *device_s, void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *x_o = static_cast<LatticeComplex *>(device_x_o);
  LatticeComplex *p = static_cast<LatticeComplex *>(device_p);
  LatticeComplex *s = static_cast<LatticeComplex *>(device_s);
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex alpha;
  alpha = origin_vals[_alpha_];
  LatticeComplex omega;
  omega = origin_vals[_omega_];
  for (int i = 0; i < _LAT_SC_; ++i) {
    x_o[idx * _LAT_SC_ + i] = x_o[idx * _LAT_SC_ + i] +
                              p[idx * _LAT_SC_ + i] * alpha +
                              s[idx * _LAT_SC_ + i] * omega;
  }
}

__global__ void bistabcg_give_r(void *device_r, void *device_s, void *device_tt,
                                void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *r = static_cast<LatticeComplex *>(device_r);
  LatticeComplex *s = static_cast<LatticeComplex *>(device_s);
  LatticeComplex *t = static_cast<LatticeComplex *>(device_tt);
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex omega;
  omega = origin_vals[_omega_];
  for (int i = 0; i < _LAT_SC_; ++i) {
    r[idx * _LAT_SC_ + i] =
        s[idx * _LAT_SC_ + i] - t[idx * _LAT_SC_ + i] * omega;
  }
}

#endif