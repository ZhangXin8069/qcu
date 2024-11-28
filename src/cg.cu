#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu
{
  __global__ void cg_give_b_e(void *device_b_e, void *device_ans_e,
                              void *device_vec0, double kappa,
                              void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *b_e = (static_cast<LatticeComplex<T> *>(device_b_e) + idx);
    LatticeComplex<T> *ans_e = (static_cast<LatticeComplex<T> *>(device_ans_e) + idx);
    LatticeComplex<T> *vec0 = (static_cast<LatticeComplex<T> *>(device_vec0) + idx);
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      b_e[i] = ans_e[i] - vec0[i] * kappa; // b_e=ans_e-kappa*D_eo(ans_o)
    }
  }
  __global__ void cg_give_b_o(void *device_b_o, void *device_ans_o,
                              void *device_vec1, double kappa,
                              void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *b_o = (static_cast<LatticeComplex<T> *>(device_b_o) + idx);
    LatticeComplex<T> *ans_o = (static_cast<LatticeComplex<T> *>(device_ans_o) + idx);
    LatticeComplex<T> *vec1 = (static_cast<LatticeComplex<T> *>(device_vec1) + idx);
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      b_o[i] = ans_o[i] - vec1[i] * kappa; // b_o=ans_o-kappa*D_oe(ans_e)
    }
  }
  __global__ void cg_give_b__o(void *device_b__o, void *device_b_o,
                               void *device_vec0, double kappa,
                               void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *b__o = (static_cast<LatticeComplex<T> *>(device_b__o) + idx);
    LatticeComplex<T> *b_o = (static_cast<LatticeComplex<T> *>(device_b_o) + idx);
    LatticeComplex<T> *vec0 = (static_cast<LatticeComplex<T> *>(device_vec0) + idx);
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      b__o[i] = b_o[i] + vec0[i] * kappa; // b__o=b_o+kappa*D_oe(b_e)
    }
  }
  __global__ void cg_give_r(void *device_r, void *device_b__o, void *device_vec,
                            void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *r = (static_cast<LatticeComplex<T> *>(device_r) + idx);
    LatticeComplex<T> *b__o =
        (static_cast<LatticeComplex<T> *>(device_b__o) + idx);
    LatticeComplex<T> *vec = (static_cast<LatticeComplex<T> *>(device_vec) + idx);
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      r[i] = b__o[i] - vec[i];
    }
  }
  __global__ void cg_give_dest_o(void *device_dest_o, void *device_src_o,
                                 void *device_vec1, double kappa,
                                 void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *dest_o = (static_cast<LatticeComplex<T> *>(device_dest_o) + idx);
    LatticeComplex<T> *src_o = (static_cast<LatticeComplex<T> *>(device_src_o) + idx);
    LatticeComplex<T> *vec1 = (static_cast<LatticeComplex<T> *>(device_vec1) + idx);
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      dest_o[i] = src_o[i] - vec1[i] * kappa * kappa; // dest_o=ans_o-kappa^2*tmp1
    }
  }
  __global__ void cg_give_1diff(void *device_vals)
  {
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
    vals[_diff_tmp_] = vals[_diff_tmp_] / vals[_norm2_tmp_];
  }
  __global__ void cg_give_1beta(void *device_vals)
  {
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
    vals[_beta_] = vals[_rho_] / vals[_rho_prev_];
  }
  __global__ void cg_give_1rho_prev(void *device_vals)
  {
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
    vals[_rho_prev_] = vals[_rho_];
  }
  __global__ void cg_give_1alpha(void *device_vals)
  {
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
    vals[_alpha_] = vals[_rho_prev_] / vals[_tmp0_];
  }
  __global__ void cg_give_p(void *device_p, void *device_r_tilde,
                            void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *p = (static_cast<LatticeComplex<T> *>(device_p) + idx);
    LatticeComplex<T> *r_tilde =
        (static_cast<LatticeComplex<T> *>(device_r_tilde) + idx);
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
    LatticeComplex<T> beta;
    beta = vals[_beta_];
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      p[i] = r_tilde[i] + p[i] * beta;
    }
  }
  __global__ void cg_give_x_o(void *device_x_o, void *device_p,
                              void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *x_o = (static_cast<LatticeComplex<T> *>(device_x_o) + idx);
    LatticeComplex<T> *p = (static_cast<LatticeComplex<T> *>(device_p) + idx);
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
    LatticeComplex<T> alpha;
    alpha = vals[_alpha_];
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      x_o[i] = x_o[i] + p[i] * alpha;
    }
  }
  __global__ void cg_give_r_tilde(void *device_r, void *device_v,
                                  void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *r = (static_cast<LatticeComplex<T> *>(device_r) + idx);
    LatticeComplex<T> *v = (static_cast<LatticeComplex<T> *>(device_v) + idx);
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
    LatticeComplex<T> alpha;
    alpha = vals[_alpha_];
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      r[i] = r[i] - v[i] * alpha;
    }
  }
  __global__ void cg_give_diff(void *device_x, void *device_ans, void *device_vec,
                               void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *x = (static_cast<LatticeComplex<T> *>(device_x) + idx);
    LatticeComplex<T> *ans = (static_cast<LatticeComplex<T> *>(device_ans) + idx);
    LatticeComplex<T> *vec = (static_cast<LatticeComplex<T> *>(device_vec) + idx);
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      vec[i] = x[i] - ans[i];
    }
  }
}
