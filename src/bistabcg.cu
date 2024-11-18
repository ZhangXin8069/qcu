#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu
{
#ifdef BISTABCG
  __global__ void bistabcg_give_1beta(void *device_vals)
  {
    LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
    vals[_beta_] = (vals[_rho_] / vals[_rho_prev_]) * (vals[_alpha_] / vals[_omega_]);
  }
  __global__ void bistabcg_give_1rho_prev(void *device_vals)
  {
    LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
    vals[_rho_prev_] = vals[_rho_];
  }
  __global__ void bistabcg_give_1alpha(void *device_vals)
  {
    LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
    vals[_alpha_] = vals[_rho_] / vals[_tmp0_];
  }
  __global__ void bistabcg_give_1omega(void *device_vals)
  {
    LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
    vals[_omega_] = vals[_tmp0_] / vals[_tmp1_];
  }
  __global__ void bistabcg_give_1diff(void *device_vals)
  {
    LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
    vals[_diff_tmp_] = vals[_diff_tmp_] / vals[_norm2_tmp_];
  }
  __global__ void bistabcg_give_b_e(void *device_b_e, void *device_ans_e,
                                    void *device_vec0, double kappa,
                                    void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex *b_e = (static_cast<LatticeComplex *>(device_b_e) + idx);
    LatticeComplex *ans_e = (static_cast<LatticeComplex *>(device_ans_e) + idx);
    LatticeComplex *vec0 = (static_cast<LatticeComplex *>(device_vec0) + idx);
    int _ = int(((LatticeComplex *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      b_e[i] = ans_e[i] - vec0[i] * kappa; // b_e=ans_e-kappa*D_eo(ans_o)
    }
  }
  __global__ void bistabcg_give_b_o(void *device_b_o, void *device_ans_o,
                                    void *device_vec1, double kappa,
                                    void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex *b_o = (static_cast<LatticeComplex *>(device_b_o) + idx);
    LatticeComplex *ans_o = (static_cast<LatticeComplex *>(device_ans_o) + idx);
    LatticeComplex *vec1 = (static_cast<LatticeComplex *>(device_vec1) + idx);
    int _ = int(((LatticeComplex *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      b_o[i] = ans_o[i] - vec1[i] * kappa; // b_o=ans_o-kappa*D_oe(ans_e)
    }
  }
  __global__ void bistabcg_give_b__o(void *device_b__o, void *device_b_o,
                                     void *device_vec0, double kappa,
                                     void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex *b__o = (static_cast<LatticeComplex *>(device_b__o) + idx);
    LatticeComplex *b_o = (static_cast<LatticeComplex *>(device_b_o) + idx);
    LatticeComplex *vec0 = (static_cast<LatticeComplex *>(device_vec0) + idx);
    int _ = int(((LatticeComplex *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      b__o[i] = b_o[i] + vec0[i] * kappa; // b__o=b_o+kappa*D_oe(b_e)
    }
  }
  __global__ void bistabcg_give_dest_o(void *device_dest_o, void *device_src_o,
                                       void *device_vec1, double kappa,
                                       void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex *dest_o = (static_cast<LatticeComplex *>(device_dest_o) + idx);
    LatticeComplex *src_o = (static_cast<LatticeComplex *>(device_src_o) + idx);
    LatticeComplex *vec1 = (static_cast<LatticeComplex *>(device_vec1) + idx);
    int _ = int(((LatticeComplex *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      dest_o[i] = src_o[i] - vec1[i] * kappa * kappa; // dest_o=ans_o-kappa^2*tmp1
    }
  }
  __global__ void bistabcg_give_rr(void *device_r, void *device_b__o,
                                   void *device_r_tilde, void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex *r = (static_cast<LatticeComplex *>(device_r) + idx);
    LatticeComplex *b__o = (static_cast<LatticeComplex *>(device_b__o) + idx);
    LatticeComplex *r_tilde =
        (static_cast<LatticeComplex *>(device_r_tilde) + idx);
    int _ = int(((LatticeComplex *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      r[i] = b__o[i] - r[i];
      r_tilde[i] = r[i];
    }
  }
  __global__ void bistabcg_give_p(void *device_p, void *device_r, void *device_v,
                                  void *device_vals)
  {
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
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      p[i] = r[i] + (p[i] - v[i] * omega) * beta;
    }
  }
  __global__ void bistabcg_give_s(void *device_s, void *device_r, void *device_v,
                                  void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex *s = (static_cast<LatticeComplex *>(device_s) + idx);
    LatticeComplex *r = (static_cast<LatticeComplex *>(device_r) + idx);
    LatticeComplex *v = (static_cast<LatticeComplex *>(device_v) + idx);
    LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
    LatticeComplex alpha;
    alpha = vals[_alpha_];
    int _ = int(((LatticeComplex *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      s[i] = r[i] - v[i] * alpha;
    }
  }
  __global__ void bistabcg_give_x_o(void *device_x_o, void *device_p,
                                    void *device_s, void *device_vals)
  {
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
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      x_o[i] = x_o[i] + p[i] * alpha + s[i] * omega;
    }
  }
  __global__ void bistabcg_give_r(void *device_r, void *device_s, void *device_tt,
                                  void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex *r = (static_cast<LatticeComplex *>(device_r) + idx);
    LatticeComplex *s = (static_cast<LatticeComplex *>(device_s) + idx);
    LatticeComplex *t = (static_cast<LatticeComplex *>(device_tt) + idx);
    LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
    LatticeComplex omega;
    omega = vals[_omega_];
    int _ = int(((LatticeComplex *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      r[i] = s[i] - t[i] * omega;
    }
  }
  __global__ void bistabcg_give_diff(void *device_x, void *device_ans,
                                     void *device_vec, void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex *x = (static_cast<LatticeComplex *>(device_x) + idx);
    LatticeComplex *ans = (static_cast<LatticeComplex *>(device_ans) + idx);
    LatticeComplex *vec = (static_cast<LatticeComplex *>(device_vec) + idx);
    int _ = int(((LatticeComplex *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      vec[i] = x[i] - ans[i];
    }
  }
#endif
}
