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

__global__ void bistabcg_give_1diff(void *device_vals) {
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex norm2_tmp;
  norm2_tmp = origin_vals[_norm2_tmp_];
  LatticeComplex diff_tmp;
  diff_tmp = origin_vals[_diff_tmp_];
  origin_vals[_diff_tmp_] = diff_tmp / norm2_tmp;
}

__global__ void bistabcg_give_b_e(void *device_b_e, void *device_ans_e,
                                  void *device_vec0, double kappa) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_b_e =
      (static_cast<LatticeComplex *>(device_b_e) + idx * _LAT_SC_);
  LatticeComplex *origin_ans_e =
      (static_cast<LatticeComplex *>(device_ans_e) + idx * _LAT_SC_);
  LatticeComplex *origin_vec0 =
      (static_cast<LatticeComplex *>(device_vec0) + idx * _LAT_SC_);
  LatticeComplex b_e[_LAT_SC_];
  LatticeComplex ans_e[_LAT_SC_];
  LatticeComplex vec0[_LAT_SC_];
  give_ptr(ans_e, origin_ans_e, _LAT_SC_);
  give_ptr(vec0, origin_vec0, _LAT_SC_);
  for (int i = 0; i < _LAT_SC_; ++i) {
    b_e[i] = ans_e[i] - vec0[i] * kappa; // b_e=ans_e-kappa*D_eo(ans_o)
  }
  give_ptr(origin_b_e, b_e, _LAT_SC_);
}

__global__ void bistabcg_give_b_o(void *device_b_o, void *device_ans_o,
                                  void *device_vec1, double kappa) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_b_o =
      (static_cast<LatticeComplex *>(device_b_o) + idx * _LAT_SC_);
  LatticeComplex *origin_ans_o =
      (static_cast<LatticeComplex *>(device_ans_o) + idx * _LAT_SC_);
  LatticeComplex *origin_vec1 =
      (static_cast<LatticeComplex *>(device_vec1) + idx * _LAT_SC_);
  LatticeComplex b_o[_LAT_SC_];
  LatticeComplex ans_o[_LAT_SC_];
  LatticeComplex vec1[_LAT_SC_];
  give_ptr(ans_o, origin_ans_o, _LAT_SC_);
  give_ptr(vec1, origin_vec1, _LAT_SC_);
  for (int i = 0; i < _LAT_SC_; ++i) {
    b_o[i] = ans_o[i] - vec1[i] * kappa; // b_o=ans_o-kappa*D_oe(ans_e)
  }
  give_ptr(origin_b_o, b_o, _LAT_SC_);
}

__global__ void bistabcg_give_b__0(void *device_b__o, void *device_b_o,
                                   void *device_vec0, double kappa) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_b__o =
      (static_cast<LatticeComplex *>(device_b__o) + idx * _LAT_SC_);
  LatticeComplex *origin_b_o =
      (static_cast<LatticeComplex *>(device_b_o) + idx * _LAT_SC_);
  LatticeComplex *origin_vec0 =
      (static_cast<LatticeComplex *>(device_vec0) + idx * _LAT_SC_);
  LatticeComplex b__o[_LAT_SC_];
  LatticeComplex b_o[_LAT_SC_];
  LatticeComplex vec0[_LAT_SC_];
  give_ptr(b_o, origin_b_o, _LAT_SC_);
  give_ptr(vec0, origin_vec0, _LAT_SC_);
  for (int i = 0; i < _LAT_SC_; ++i) {
    b__o[i] = b_o[i] + vec0[i] * kappa; // b__o=b_o+kappa*D_oe(b_e)
  }
  give_ptr(origin_b__o, b__o, _LAT_SC_);
}

__global__ void bistabcg_give_dest_o(void *device_dest_o, void *device_src_o,
                                     void *device_vec1, double kappa) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_dest_o =
      (static_cast<LatticeComplex *>(device_dest_o) + idx * _LAT_SC_);
  LatticeComplex *origin_src_o =
      (static_cast<LatticeComplex *>(device_src_o) + idx * _LAT_SC_);
  LatticeComplex *origin_vec1 =
      (static_cast<LatticeComplex *>(device_vec1) + idx * _LAT_SC_);
  LatticeComplex dest_o[_LAT_SC_];
  LatticeComplex src_o[_LAT_SC_];
  LatticeComplex vec1[_LAT_SC_];
  give_ptr(src_o, origin_src_o, _LAT_SC_);
  give_ptr(vec1, origin_vec1, _LAT_SC_);
  for (int i = 0; i < _LAT_SC_; ++i) {
    dest_o[i] = src_o[i] - vec1[i] * kappa * kappa; // dest_o=ans_o-kappa^2*tmp1
  }
  give_ptr(origin_dest_o, dest_o, _LAT_SC_);
}

__global__ void bistabcg_give_rr(void *device_r, void *device_b__o,
                                 void *device_r_tilde) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_r =
      (static_cast<LatticeComplex *>(device_r) + idx * _LAT_SC_);
  LatticeComplex *origin_b__o =
      (static_cast<LatticeComplex *>(device_b__o) + idx * _LAT_SC_);
  LatticeComplex *origin_r_tilde =
      (static_cast<LatticeComplex *>(device_r_tilde) + idx * _LAT_SC_);
  LatticeComplex r[_LAT_SC_];
  LatticeComplex b__o[_LAT_SC_];
  LatticeComplex r_tilde[_LAT_SC_];
  give_ptr(r, origin_r, _LAT_SC_);
  give_ptr(b__o, origin_b__o, _LAT_SC_);
  for (int i = 0; i < _LAT_SC_; ++i) {
    r[i] = b__o[i] - r[i];
    r_tilde[i] = r[i];
  }
  give_ptr(origin_r, r, _LAT_SC_);
  give_ptr(origin_r_tilde, r_tilde, _LAT_SC_);
}

__global__ void bistabcg_give_p(void *device_p, void *device_r, void *device_v,
                                void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_p =
      (static_cast<LatticeComplex *>(device_p) + idx * _LAT_SC_);
  LatticeComplex *origin_r =
      (static_cast<LatticeComplex *>(device_r) + idx * _LAT_SC_);
  LatticeComplex *origin_v =
      (static_cast<LatticeComplex *>(device_v) + idx * _LAT_SC_);
  LatticeComplex p[_LAT_SC_];
  LatticeComplex r[_LAT_SC_];
  LatticeComplex v[_LAT_SC_];
  give_ptr(p, origin_p, _LAT_SC_);
  give_ptr(r, origin_r, _LAT_SC_);
  give_ptr(v, origin_v, _LAT_SC_);
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex beta;
  beta = origin_vals[_beta_];
  LatticeComplex omega;
  omega = origin_vals[_omega_];
  for (int i = 0; i < _LAT_SC_; ++i) {
    p[i] = r[i] + (p[i] - v[i] * omega) * beta;
  }
  give_ptr(origin_p, p, _LAT_SC_);
}

__global__ void bistabcg_give_s(void *device_s, void *device_r, void *device_v,
                                void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_s =
      (static_cast<LatticeComplex *>(device_s) + idx * _LAT_SC_);
  LatticeComplex *origin_r =
      (static_cast<LatticeComplex *>(device_r) + idx * _LAT_SC_);
  LatticeComplex *origin_v =
      (static_cast<LatticeComplex *>(device_v) + idx * _LAT_SC_);
  LatticeComplex s[_LAT_SC_];
  LatticeComplex r[_LAT_SC_];
  LatticeComplex v[_LAT_SC_];
  give_ptr(r, origin_r, _LAT_SC_);
  give_ptr(v, origin_v, _LAT_SC_);
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex alpha;
  alpha = origin_vals[_alpha_];
  for (int i = 0; i < _LAT_SC_; ++i) {
    s[i] = r[i] - v[i] * alpha;
  }
  give_ptr(origin_s, s, _LAT_SC_);
}

__global__ void bistabcg_give_x_o(void *device_x_o, void *device_p,
                                  void *device_s, void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_x_o =
      (static_cast<LatticeComplex *>(device_x_o) + idx * _LAT_SC_);
  LatticeComplex *origin_p =
      (static_cast<LatticeComplex *>(device_p) + idx * _LAT_SC_);
  LatticeComplex *origin_s =
      (static_cast<LatticeComplex *>(device_s) + idx * _LAT_SC_);
  LatticeComplex x_o[_LAT_SC_];
  LatticeComplex p[_LAT_SC_];
  LatticeComplex s[_LAT_SC_];
  give_ptr(x_o, origin_x_o, _LAT_SC_);
  give_ptr(p, origin_p, _LAT_SC_);
  give_ptr(s, origin_s, _LAT_SC_);
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex alpha;
  alpha = origin_vals[_alpha_];
  LatticeComplex omega;
  omega = origin_vals[_omega_];
  for (int i = 0; i < _LAT_SC_; ++i) {
    x_o[i] = x_o[i] + p[i] * alpha + s[i] * omega;
  }
  give_ptr(origin_x_o, x_o, _LAT_SC_);
}

__global__ void bistabcg_give_r(void *device_r, void *device_s, void *device_tt,
                                void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex *origin_r =
      (static_cast<LatticeComplex *>(device_r) + idx * _LAT_SC_);
  LatticeComplex *origin_s =
      (static_cast<LatticeComplex *>(device_s) + idx * _LAT_SC_);
  LatticeComplex *origin_t =
      (static_cast<LatticeComplex *>(device_tt) + idx * _LAT_SC_);
  LatticeComplex r[_LAT_SC_];
  LatticeComplex s[_LAT_SC_];
  LatticeComplex t[_LAT_SC_];
  give_ptr(s, origin_s, _LAT_SC_);
  give_ptr(t, origin_t, _LAT_SC_);
  LatticeComplex *origin_vals = static_cast<LatticeComplex *>(device_vals);
  LatticeComplex omega;
  omega = origin_vals[_omega_];
  for (int i = 0; i < _LAT_SC_; ++i) {
    r[i] = s[i] - t[i] * omega;
  }
  give_ptr(origin_r, r, _LAT_SC_);
}

#endif