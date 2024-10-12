#include "../include/qcu.h"
#ifdef BISTABCG
__global__ void bistabcg_give_1beta(void *device_vals) {
  LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex rho_prev;
  rho_prev = vals[_rho_prev_];
  LatticeComplex rho;
  rho = vals[_rho_];
  LatticeComplex alpha;
  alpha = vals[_alpha_];
  LatticeComplex beta;
  beta = vals[_beta_];
  LatticeComplex omega;
  omega = vals[_omega_];
  beta = (rho / rho_prev) * (alpha / omega);
  vals[_beta_] = beta;
}
__global__ void bistabcg_give_1rho_prev(void *device_vals) {
  LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex rho;
  rho = vals[_rho_];
  vals[_rho_prev_] = rho;
}
__global__ void bistabcg_give_1alpha(void *device_vals) {
  LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex rho;
  rho = vals[_rho_];
  LatticeComplex tmp0;
  tmp0 = vals[_tmp0_];
  vals[_alpha_] = rho / tmp0;
}
__global__ void bistabcg_give_1omega(void *device_vals) {
  LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex tmp0;
  tmp0 = vals[_tmp0_];
  LatticeComplex tmp1;
  tmp1 = vals[_tmp1_];
  vals[_omega_] = tmp0 / tmp1;
}
__global__ void bistabcg_give_rr(void *device_r, void *device_b__o,
                                 void *device_r_tilde, void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *r = (static_cast<LatticeComplex *>(device_r) + idx);
  LatticeComplex *b__o = (static_cast<LatticeComplex *>(device_b__o) + idx);
  LatticeComplex *r_tilde =
      (static_cast<LatticeComplex *>(device_r_tilde) + idx);
  int _ = int(((LatticeComplex *)device_vals)[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    r[i] = b__o[i] - r[i];
    r_tilde[i] = r[i];
  }
}
__global__ void bistabcg_give_p(void *device_p, void *device_r, void *device_v,
                                void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *p = (static_cast<LatticeComplex *>(device_p) + idx);
  LatticeComplex *r = (static_cast<LatticeComplex *>(device_r) + idx);
  LatticeComplex *v = (static_cast<LatticeComplex *>(device_v) + idx);
  LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex beta;
  beta = vals[_beta_];
  LatticeComplex omega;
  omega = vals[_omega_];
  int _ = int(((LatticeComplex *)device_vals)[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    p[i] = r[i] + (p[i] - v[i] * omega) * beta;
  }
}
__global__ void bistabcg_give_s(void *device_s, void *device_r, void *device_v,
                                void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *s = (static_cast<LatticeComplex *>(device_s) + idx);
  LatticeComplex *r = (static_cast<LatticeComplex *>(device_r) + idx);
  LatticeComplex *v = (static_cast<LatticeComplex *>(device_v) + idx);
  LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex alpha;
  alpha = vals[_alpha_];
  int _ = int(((LatticeComplex *)device_vals)[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    s[i] = r[i] - v[i] * alpha;
  }
}
__global__ void bistabcg_give_x_o(void *device_x_o, void *device_p,
                                  void *device_s, void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *x_o = (static_cast<LatticeComplex *>(device_x_o) + idx);
  LatticeComplex *p = (static_cast<LatticeComplex *>(device_p) + idx);
  LatticeComplex *s = (static_cast<LatticeComplex *>(device_s) + idx);
  LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex alpha;
  alpha = vals[_alpha_];
  LatticeComplex omega;
  omega = vals[_omega_];
  int _ = int(((LatticeComplex *)device_vals)[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    x_o[i] = x_o[i] + p[i] * alpha + s[i] * omega;
  }
}
__global__ void bistabcg_give_r(void *device_r, void *device_s, void *device_tt,
                                void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *r = (static_cast<LatticeComplex *>(device_r) + idx);
  LatticeComplex *s = (static_cast<LatticeComplex *>(device_s) + idx);
  LatticeComplex *t = (static_cast<LatticeComplex *>(device_tt) + idx);
  LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex omega;
  omega = vals[_omega_];
  int _ = int(((LatticeComplex *)device_vals)[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    r[i] = s[i] - t[i] * omega;
  }
}
#endif