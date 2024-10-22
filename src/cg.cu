#include "../include/qcu.h"
#ifdef _QCU_LATTICE_CG_
__global__ void cg_give_b_e(void *device_b_e, void *device_ans_e,
                            void *device_vec0, double kappa,
                            void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *b_e = (static_cast<LatticeComplex *>(device_b_e) + idx);
  LatticeComplex *ans_e = (static_cast<LatticeComplex *>(device_ans_e) + idx);
  LatticeComplex *vec0 = (static_cast<LatticeComplex *>(device_vec0) + idx);
  int _ = int(((LatticeComplex *)device_vals)[_qcu_lat_4dim_]._data.x);
  for (int i = 0; i < _QCU_LAT_SC_ * _; i += _) {
    b_e[i] = ans_e[i] - vec0[i] * kappa; // b_e=ans_e-kappa*D_eo(ans_o)
  }
}
__global__ void cg_give_b_o(void *device_b_o, void *device_ans_o,
                            void *device_vec1, double kappa,
                            void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *b_o = (static_cast<LatticeComplex *>(device_b_o) + idx);
  LatticeComplex *ans_o = (static_cast<LatticeComplex *>(device_ans_o) + idx);
  LatticeComplex *vec1 = (static_cast<LatticeComplex *>(device_vec1) + idx);
  int _ = int(((LatticeComplex *)device_vals)[_qcu_lat_4dim_]._data.x);
  for (int i = 0; i < _QCU_LAT_SC_ * _; i += _) {
    b_o[i] = ans_o[i] - vec1[i] * kappa; // b_o=ans_o-kappa*D_oe(ans_e)
  }
}
__global__ void cg_give_b__o(void *device_b__o, void *device_b_o,
                             void *device_vec0, double kappa,
                             void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *b__o = (static_cast<LatticeComplex *>(device_b__o) + idx);
  LatticeComplex *b_o = (static_cast<LatticeComplex *>(device_b_o) + idx);
  LatticeComplex *vec0 = (static_cast<LatticeComplex *>(device_vec0) + idx);
  int _ = int(((LatticeComplex *)device_vals)[_qcu_lat_4dim_]._data.x);
  for (int i = 0; i < _QCU_LAT_SC_ * _; i += _) {
    b__o[i] = b_o[i] + vec0[i] * kappa; // b__o=b_o+kappa*D_oe(b_e)
  }
}
__global__ void cg_give_dest_o(void *device_dest_o, void *device_src_o,
                               void *device_vec1, double kappa,
                               void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *dest_o = (static_cast<LatticeComplex *>(device_dest_o) + idx);
  LatticeComplex *src_o = (static_cast<LatticeComplex *>(device_src_o) + idx);
  LatticeComplex *vec1 = (static_cast<LatticeComplex *>(device_vec1) + idx);
  int _ = int(((LatticeComplex *)device_vals)[_qcu_lat_4dim_]._data.x);
  for (int i = 0; i < _QCU_LAT_SC_ * _; i += _) {
    dest_o[i] = src_o[i] - vec1[i] * kappa * kappa; // dest_o=ans_o-kappa^2*tmp1
  }
}
__global__ void cg_give_1diff(void *device_vals) {
  LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex norm2_tmp;
  norm2_tmp = vals[_qcu_norm2_tmp_];
  LatticeComplex diff_tmp;
  diff_tmp = vals[_qcu_diff_tmp_];
  vals[_qcu_diff_tmp_] = diff_tmp / norm2_tmp;
}
__global__ void cg_give_diff(void *device_x, void *device_ans, void *device_vec,
                             void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *x = (static_cast<LatticeComplex *>(device_x) + idx);
  LatticeComplex *ans = (static_cast<LatticeComplex *>(device_ans) + idx);
  LatticeComplex *vec = (static_cast<LatticeComplex *>(device_vec) + idx);
  int _ = int(((LatticeComplex *)device_vals)[_qcu_lat_4dim_]._data.x);
  for (int i = 0; i < _QCU_LAT_SC_ * _; i += _) {
    vec[i] = x[i] - ans[i];
  }
}
__global__ void cg_give_1beta(void *device_vals) {
  LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex rho_prev;
  rho_prev = vals[_qcu_rho_prev_];
  LatticeComplex rho;
  rho = vals[_qcu_rho_];
  LatticeComplex beta;
  beta = vals[_qcu_beta_];
  beta = rho_prev / rho;
  vals[_qcu_beta_] = beta;
}
__global__ void cg_give_1alpha(void *device_vals) {
  LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex rho;
  rho = vals[_qcu_rho_];
  LatticeComplex tmp0;
  tmp0 = vals[_qcu_tmp0_];
  vals[_qcu_alpha_] = rho / tmp0;
}
__global__ void cg_give_p(void *device_p, void *device_r_tilde,
                          void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *p = (static_cast<LatticeComplex *>(device_p) + idx);
  LatticeComplex *r_tilde =
      (static_cast<LatticeComplex *>(device_r_tilde) + idx);
  LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex beta;
  beta = vals[_qcu_beta_];
  int _ = int(((LatticeComplex *)device_vals)[_qcu_lat_4dim_]._data.x);
  for (int i = 0; i < _QCU_LAT_SC_ * _; i += _) {
    p[i] = r_tilde[i] + p[i] * beta;
  }
}
__global__ void cg_give_x_o(void *device_x_o, void *device_p,
                            void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *x_o = (static_cast<LatticeComplex *>(device_x_o) + idx);
  LatticeComplex *p = (static_cast<LatticeComplex *>(device_p) + idx);
  LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex alpha;
  alpha = vals[_qcu_alpha_];
  int _ = int(((LatticeComplex *)device_vals)[_qcu_lat_4dim_]._data.x);
  for (int i = 0; i < _QCU_LAT_SC_ * _; i += _) {
    x_o[i] = x_o[i] + p[i] * alpha;
  }
}
__global__ void cg_give_rr(void *device_r, void *device_r_tilde, void *device_v,
                           void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *r = (static_cast<LatticeComplex *>(device_r) + idx);
  LatticeComplex *r_tilde =
      (static_cast<LatticeComplex *>(device_r_tilde) + idx);
  LatticeComplex *v = (static_cast<LatticeComplex *>(device_v) + idx);
  LatticeComplex *vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex alpha;
  alpha = vals[_qcu_alpha_];
  int _ = int(((LatticeComplex *)device_vals)[_qcu_lat_4dim_]._data.x);
  for (int i = 0; i < _QCU_LAT_SC_ * _; i += _) {
    r_tilde[i] = r[i] - v[i] * alpha;
    r[i] = r_tilde[i];
  }
}
#endif