// clang-format off
#include "../include/qcu.h"
#ifdef CLOVER_DSLASH
// wait for rebuild
// clang-format on
__global__ void make_clover_all(
    void *device_U, void *device_clover, void *device_lat_xyzt,
    int device_parity, int node_rank, int device_flag,
    void *device_u_b_x_recv_vec, void *device_u_f_x_recv_vec,
    void *device_u_b_y_recv_vec, void *device_u_f_y_recv_vec,
    void *device_u_b_z_recv_vec, void *device_u_f_z_recv_vec,
    void *device_u_b_t_recv_vec, void *device_u_f_t_recv_vec,
    void *device_u_b_x_b_y_recv_vec, void *device_u_f_x_b_y_recv_vec,
    void *device_u_b_x_f_y_recv_vec, void *device_u_f_x_f_y_recv_vec,
    void *device_u_b_x_b_z_recv_vec, void *device_u_f_x_b_z_recv_vec,
    void *device_u_b_x_f_z_recv_vec, void *device_u_f_x_f_z_recv_vec,
    void *device_u_b_x_b_t_recv_vec, void *device_u_f_x_b_t_recv_vec,
    void *device_u_b_x_f_t_recv_vec, void *device_u_f_x_f_t_recv_vec,
    void *device_u_b_y_b_z_recv_vec, void *device_u_f_y_b_z_recv_vec,
    void *device_u_b_y_f_z_recv_vec, void *device_u_f_y_f_z_recv_vec,
    void *device_u_b_y_b_t_recv_vec, void *device_u_f_y_b_t_recv_vec,
    void *device_u_b_y_f_t_recv_vec, void *device_u_f_y_f_t_recv_vec,
    void *device_u_b_z_b_t_recv_vec, void *device_u_f_z_b_t_recv_vec,
    void *device_u_b_z_f_t_recv_vec, void *device_u_f_z_f_t_recv_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int parity = idx;
  int *lat_xyzt = static_cast<int *>(device_lat_xyzt);
  int lat_x = lat_xyzt[_X_];
  int lat_y = lat_xyzt[_Y_];
  int lat_z = lat_xyzt[_Z_];
  int lat_t = lat_xyzt[_T_];
  int lat_tzyx = lat_xyzt[_XYZT_];
  int move0;
  int move1;
  move0 = lat_x * lat_y * lat_z;
  int t = parity / move0;
  parity -= t * move0;
  move0 = lat_x * lat_y;
  int z = parity / move0;
  parity -= z * move0;
  int y = parity / lat_x;
  int x = parity - y * lat_x;
  int eo = (y + z + t) & 0x01; //(y+z+t)%2
  parity = device_parity;
  int move_wards[_WARDS_];
  move_backward_x(move_wards[_B_X_], x, lat_x, eo, parity);
  move_backward(move_wards[_B_Y_], y, lat_y);
  move_backward(move_wards[_B_Z_], z, lat_z);
  move_backward(move_wards[_B_T_], t, lat_t);
  move_forward_x(move_wards[_F_X_], x, lat_x, eo, parity);
  move_forward(move_wards[_F_Y_], y, lat_y);
  move_forward(move_wards[_F_Z_], z, lat_z);
  move_forward(move_wards[_F_T_], t, lat_t);
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex *origin_U = ((static_cast<LatticeComplex *>(device_U)) + idx);
  LatticeComplex *origin_clover =
      ((static_cast<LatticeComplex *>(device_clover)) + idx);
  LatticeComplex *tmp_U;
  LatticeComplex tmp1[_LAT_CC_];
  LatticeComplex tmp2[_LAT_CC_];
  LatticeComplex tmp3[_LAT_CC_];
  LatticeComplex U[_LAT_CC_];
  LatticeComplex clover[_LAT_SCSC_];
  // just all
  int if_b_x = (move_wards[_B_X_] == lat_x - 1);
  int if_b_y = (move_wards[_B_Y_] == lat_y - 1);
  int if_b_z = (move_wards[_B_Z_] == lat_z - 1);
  int if_b_t = (move_wards[_B_T_] == lat_t - 1);
  int if_f_x = (move_wards[_F_X_] == 1 - lat_x);
  int if_f_y = (move_wards[_F_Y_] == 1 - lat_y);
  int if_f_z = (move_wards[_F_Z_] == 1 - lat_z);
  int if_f_t = (move_wards[_F_T_] == 1 - lat_t);
  int if_b_x_b_y =
      (move_wards[_B_X_] == lat_x - 1) * (move_wards[_B_Y_] == lat_y - 1);
  int if_f_x_b_y =
      (move_wards[_F_X_] == 1 - lat_x) * (move_wards[_B_Y_] == lat_y - 1);
  int if_b_x_f_y =
      (move_wards[_B_X_] == lat_x - 1) * (move_wards[_F_Y_] == 1 - lat_y);
  // // int if_f_x_f_y=
  //(move_wards[_F_X_]==1-lat_x)*(move_wards[_F_Y_]==1-lat_y);
  int if_b_x_b_z =
      (move_wards[_B_X_] == lat_x - 1) * (move_wards[_B_Z_] == lat_z - 1);
  int if_f_x_b_z =
      (move_wards[_F_X_] == 1 - lat_x) * (move_wards[_B_Z_] == lat_z - 1);
  int if_b_x_f_z =
      (move_wards[_B_X_] == lat_x - 1) * (move_wards[_F_Z_] == 1 - lat_z);
  // // int if_f_x_f_z=
  //(move_wards[_F_X_]==1-lat_x)*(move_wards[_F_Z_]==1-lat_z);
  int if_b_x_b_t =
      (move_wards[_B_X_] == lat_x - 1) * (move_wards[_B_T_] == lat_t - 1);
  int if_f_x_b_t =
      (move_wards[_F_X_] == 1 - lat_x) * (move_wards[_B_T_] == lat_t - 1);
  int if_b_x_f_t =
      (move_wards[_B_X_] == lat_x - 1) * (move_wards[_F_T_] == 1 - lat_t);
  // // int if_f_x_f_t=
  //(move_wards[_F_X_]==1-lat_x)*(move_wards[_F_T_]==1-lat_t);
  int if_b_y_b_z =
      (move_wards[_B_Y_] == lat_y - 1) * (move_wards[_B_Z_] == lat_z - 1);
  int if_f_y_b_z =
      (move_wards[_F_Y_] == 1 - lat_y) * (move_wards[_B_Z_] == lat_z - 1);
  int if_b_y_f_z =
      (move_wards[_B_Y_] == lat_y - 1) * (move_wards[_F_Z_] == 1 - lat_z);
  // // int if_f_y_f_z=
  //(move_wards[_F_Y_]==1-lat_y)*(move_wards[_F_Z_]==1-lat_z);
  int if_b_y_b_t =
      (move_wards[_B_Y_] == lat_y - 1) * (move_wards[_B_T_] == lat_t - 1);
  int if_f_y_b_t =
      (move_wards[_F_Y_] == 1 - lat_y) * (move_wards[_B_T_] == lat_t - 1);
  int if_b_y_f_t =
      (move_wards[_B_Y_] == lat_y - 1) * (move_wards[_F_T_] == 1 - lat_t);
  // // int if_f_y_f_t=
  //(move_wards[_F_Y_]==1-lat_y)*(move_wards[_F_T_]==1-lat_t);
  int if_b_z_b_t =
      (move_wards[_B_Z_] == lat_z - 1) * (move_wards[_B_T_] == lat_t - 1);
  int if_f_z_b_t =
      (move_wards[_F_Z_] == 1 - lat_z) * (move_wards[_B_T_] == lat_t - 1);
  int if_b_z_f_t =
      (move_wards[_B_Z_] == lat_z - 1) * (move_wards[_F_T_] == 1 - lat_t);
  // // int if_f_z_f_t=
  //(move_wards[_F_Z_]==1-lat_z)*(move_wards[_F_T_]==1-lat_t);
  // sigmaF
  {
    give_vals(clover, zero, _LAT_SCSC_);
    // give_vals(origin_clover,zero,_LAT_SCSC_);//BUG!!!!!!
    give_vals(tmp1, zero, _LAT_CC_);
    give_vals(tmp2, zero, _LAT_CC_);
  }
  // XY
  give_vals(U, zero, _LAT_CC_);
  {
    ////x,y,z,t;x
    tmp_U = (origin_U + (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    ////x+1,y,z,t;y
    if (if_f_x) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_x_recv_vec) +
               ((((_Y_ * lat_t + t) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_x);
    } else {
      move0 = move_wards[_F_X_];
      tmp_U = (origin_U + move0 + (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x,y+1,z,t;x;dag
    if (if_f_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_y_recv_vec) +
               ((((_X_ * lat_t + t) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_y);
    } else {
      move0 = move_wards[_F_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z,t;y;dag
    tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x,y,z,t;y
    tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    ////x-1,y+1,z,t;x;dag
    if (if_b_x_f_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_f_y_recv_vec) +
               ((((_X_ * lat_t + t) * lat_z + z) * 1 + 0) * 1 + 0));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_x / lat_y);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_F_Y_];
      tmp_U = (origin_U + move0 + move1 * lat_x +
               (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x-1,y,z,t;y;dag
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_recv_vec) +
               ((((_Y_ * lat_t + t) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x-1,y,z,t;x
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_recv_vec) +
               ((((_X_ * lat_t + t) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x-1,y,z,t;x;dag
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_recv_vec) +
               ((((_X_ * lat_t + t) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    ////x-1,y-1,z,t;y;dag
    if (if_b_x_b_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_b_y_recv_vec) +
               ((((_Y_ * lat_t + t) * lat_z + z) * 1 + 0) * 1 + 0));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_x / lat_y);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 + move1 * lat_x +
               (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x-1,y-1,z,t;x
    if (if_b_x_b_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_b_y_recv_vec) +
               ((((_X_ * lat_t + t) * lat_z + z) * 1 + 0) * 1 + 0));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_x / lat_y);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 + move1 * lat_x +
               (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y-1,z,t;y
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_t + t) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x,y-1,z,t;y;dag
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_t + t) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    ////x,y-1,z,t;x
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_y_recv_vec) +
               ((((_X_ * lat_t + t) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x+1,y-1,z,t;y
    if (if_f_x_b_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_x_b_y_recv_vec) +
               ((((_Y_ * lat_t + t) * lat_z + z) * 1 + 0) * 1 + 0));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_x / lat_y);
    } else {
      move0 = move_wards[_F_X_];
      move1 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 + move1 * lat_x +
               (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z,t;x;dag
    tmp_U = (origin_U + (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        clover[c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * (-I);
        clover[39 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * I;
        clover[78 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * (-I);
        clover[117 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * I;
      }
    }
  }
  // XZ
  give_vals(U, zero, _LAT_CC_);
  {
    ////x,y,z,t;x
    tmp_U = (origin_U + (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    ////x+1,y,z,t;z
    if (if_f_x) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_x_recv_vec) +
               ((((_Z_ * lat_t + t) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_x);
    } else {
      move0 = move_wards[_F_X_];
      tmp_U = (origin_U + move0 + (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x,y,z+1,t;x;dag
    if (if_f_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_z_recv_vec) +
               ((((_X_ * lat_t + t) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_z);
    } else {
      move0 = move_wards[_F_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z,t;z;dag
    tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x,y,z,t;z
    tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    ////x-1,y,z+1,t;x;dag
    if (if_b_x_f_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_f_z_recv_vec) +
               ((((_X_ * lat_t + t) * 1 + 0) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_x / lat_z);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_F_Z_];
      tmp_U = (origin_U + move0 + move1 * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x-1,y,z,t;z;dag
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_recv_vec) +
               ((((_Z_ * lat_t + t) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x-1,y,z,t;x
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_recv_vec) +
               ((((_X_ * lat_t + t) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x-1,y,z,t;x;dag
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_recv_vec) +
               ((((_X_ * lat_t + t) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    ////x-1,y,z-1,t;z;dag
    if (if_b_x_b_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_b_z_recv_vec) +
               ((((_Z_ * lat_t + t) * 1 + 0) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_x / lat_z);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 + move1 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x-1,y,z-1,t;x
    if (if_b_x_b_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_b_z_recv_vec) +
               ((((_X_ * lat_t + t) * 1 + 0) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_x / lat_z);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 + move1 * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z-1,t;z
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_t + t) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x,y,z-1,t;z;dag
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_t + t) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    ////x,y,z-1,t;x
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_z_recv_vec) +
               ((((_X_ * lat_t + t) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x+1,y,z-1,t;z
    if (if_f_x_b_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_x_b_z_recv_vec) +
               ((((_Z_ * lat_t + t) * 1 + 0) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_x / lat_z);
    } else {
      move0 = move_wards[_F_X_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 + move1 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z,t;x;dag
    tmp_U = (origin_U + (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        clover[_LAT_C_ + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * (-1);
        clover[36 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj());
        clover[81 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * (-1);
        clover[114 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj());
      }
    }
  }
  // XT
  give_vals(U, zero, _LAT_CC_);
  {
    ////x,y,z,t;x
    tmp_U = (origin_U + (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    ////x+1,y,z,t;t
    if (if_f_x) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_x_recv_vec) +
               ((((_T_ * lat_t + t) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_x);
    } else {
      move0 = move_wards[_F_X_];
      tmp_U = (origin_U + move0 + (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x,y,z,t+1;x;dag
    if (if_f_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_t_recv_vec) +
               ((((_X_ * 1 + 0) * lat_z + z) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_t);
    } else {
      move0 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z,t;t;dag
    tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x,y,z,t;t
    tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    ////x-1,y,z,t+1;x;dag
    if (if_b_x_f_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_f_t_recv_vec) +
               ((((_X_ * 1 + 0) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_x / lat_t);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 + move1 * lat_z * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x-1,y,z,t;t;dag
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_recv_vec) +
               ((((_T_ * lat_t + t) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x-1,y,z,t;x
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_recv_vec) +
               ((((_X_ * lat_t + t) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x-1,y,z,t;x;dag
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_recv_vec) +
               ((((_X_ * lat_t + t) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    ////x-1,y,z,t-1;t;dag
    if (if_b_x_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_b_t_recv_vec) +
               ((((_T_ * 1 + 0) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_x / lat_t);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 + move1 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x-1,y,z,t-1;x
    if (if_b_x_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_x_b_t_recv_vec) +
               ((((_X_ * 1 + 0) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_x / lat_t);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 + move1 * lat_z * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z,t-1;t
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_t_recv_vec) +
               ((((_T_ * 1 + 0) * lat_z + z) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x,y,z,t-1;t;dag
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_t_recv_vec) +
               ((((_T_ * 1 + 0) * lat_z + z) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    ////x,y,z,t-1;x
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_t_recv_vec) +
               ((((_X_ * 1 + 0) * lat_z + z) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x+1,y,z,t-1;t
    if (if_f_x_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_x_b_t_recv_vec) +
               ((((_T_ * 1 + 0) * lat_z + z) * lat_y + y) * 1 + 0));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_x / lat_t);
    } else {
      move0 = move_wards[_F_X_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 + move1 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z,t;x;dag
    tmp_U = (origin_U + (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        clover[_LAT_C_ + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * I;
        clover[36 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * I;
        clover[81 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * (-I);
        clover[114 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * (-I);
      }
    }
  }
  // YZ
  give_vals(U, zero, _LAT_CC_);
  {
    ////x,y,z,t;y
    tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    ////x,y+1,z,t;z
    if (if_f_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_y_recv_vec) +
               ((((_Z_ * lat_t + t) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_y);
    } else {
      move0 = move_wards[_F_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x,y,z+1,t;y;dag
    if (if_f_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_z_recv_vec) +
               ((((_Y_ * lat_t + t) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_z);
    } else {
      move0 = move_wards[_F_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z,t;z;dag
    tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x,y,z,t;z
    tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    ////x,y-1,z+1,t;y;dag
    if (if_b_y_f_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_y_f_z_recv_vec) +
               ((((_Y_ * lat_t + t) * 1 + 0) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_y / lat_z);
    } else {
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_F_Z_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x,y-1,z,t;z;dag
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_y_recv_vec) +
               ((((_Z_ * lat_t + t) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y-1,z,t;y
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_t + t) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x,y-1,z,t;y;dag
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_t + t) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    ////x,y-1,z-1,t;z;dag
    if (if_b_y_b_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_y_b_z_recv_vec) +
               ((((_Z_ * lat_t + t) * 1 + 0) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_y / lat_z);
    } else {
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x,y-1,z-1,t;y
    if (if_b_y_b_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_y_b_z_recv_vec) +
               ((((_Y_ * lat_t + t) * 1 + 0) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_y / lat_z);
    } else {
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z-1,t;z
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_t + t) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x,y,z-1,t;z;dag
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_t + t) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    ////x,y,z-1,t;y
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_z_recv_vec) +
               ((((_Y_ * lat_t + t) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x,y+1,z-1,t;z
    if (if_f_y_b_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_y_b_z_recv_vec) +
               ((((_Z_ * lat_t + t) * 1 + 0) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_y / lat_z);
    } else {
      move0 = move_wards[_F_Y_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z,t;y;dag
    tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        clover[_LAT_C_ + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * (-I);
        clover[36 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * (-I);
        clover[81 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * (-I);
        clover[114 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * (-I);
      }
    }
  }
  // YT
  give_vals(U, zero, _LAT_CC_);
  {
    ////x,y,z,t;y
    tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    ////x,y+1,z,t;t
    if (if_f_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_y_recv_vec) +
               ((((_T_ * lat_t + t) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_y);
    } else {
      move0 = move_wards[_F_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x,y,z,t+1;y;dag
    if (if_f_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_t_recv_vec) +
               ((((_Y_ * 1 + 0) * lat_z + z) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_t);
    } else {
      move0 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z,t;t;dag
    tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x,y,z,t;t
    tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    ////x,y-1,z,t+1;y;dag
    if (if_b_y_f_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_y_f_t_recv_vec) +
               ((((_Y_ * 1 + 0) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_y / lat_t);
    } else {
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_z * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x,y-1,z,t;t;dag
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_y_recv_vec) +
               ((((_T_ * lat_t + t) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y-1,z,t;y
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_t + t) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x,y-1,z,t;y;dag
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_t + t) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    ////x,y-1,z,t-1;t;dag
    if (if_b_y_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_y_b_t_recv_vec) +
               ((((_T_ * 1 + 0) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_y / lat_t);
    } else {
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x,y-1,z,t-1;y
    if (if_b_y_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_y_b_t_recv_vec) +
               ((((_Y_ * 1 + 0) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_y / lat_t);
    } else {
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_z * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z,t-1;t
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_t_recv_vec) +
               ((((_T_ * 1 + 0) * lat_z + z) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x,y,z,t-1;t;dag
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_t_recv_vec) +
               ((((_T_ * 1 + 0) * lat_z + z) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    ////x,y,z,t-1;y
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_t_recv_vec) +
               ((((_Y_ * 1 + 0) * lat_z + z) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x,y+1,z,t-1;t
    if (if_f_y_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_y_b_t_recv_vec) +
               ((((_T_ * 1 + 0) * lat_z + z) * 1 + 0) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_y / lat_t);
    } else {
      move0 = move_wards[_F_Y_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z,t;y;dag
    tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        clover[_LAT_C_ + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * (-1);
        clover[36 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj());
        clover[81 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj());
        clover[114 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * (-1);
      }
    }
  }
  // ZT
  give_vals(U, zero, _LAT_CC_);
  {
    ////x,y,z,t;z
    tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    ////x,y,z+1,t;t
    if (if_f_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_z_recv_vec) +
               ((((_T_ * lat_t + t) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_z);
    } else {
      move0 = move_wards[_F_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x,y,z,t+1;z;dag
    if (if_f_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_t_recv_vec) +
               ((((_Z_ * 1 + 0) * lat_z + z) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_t);
    } else {
      move0 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z,t;t;dag
    tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x,y,z,t;t
    tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    ////x,y,z-1,t+1;z;dag
    if (if_b_z_f_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_z_f_t_recv_vec) +
               ((((_Z_ * 1 + 0) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_z / lat_t);
    } else {
      move0 = move_wards[_B_Z_];
      move1 = move_wards[_F_T_];
      tmp_U =
          (origin_U + move0 * lat_y * lat_x + move1 * lat_z * lat_y * lat_x +
           (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x,y,z-1,t;t;dag
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_z_recv_vec) +
               ((((_T_ * lat_t + t) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z-1,t;z
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_t + t) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x,y,z-1,t;z;dag
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_t + t) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    ////x,y,z-1,t-1;t;dag
    if (if_b_z_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_z_b_t_recv_vec) +
               ((((_T_ * 1 + 0) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_z / lat_t);
    } else {
      move0 = move_wards[_B_Z_];
      move1 = move_wards[_B_T_];
      tmp_U =
          (origin_U + move0 * lat_y * lat_x + move1 * lat_z * lat_y * lat_x +
           (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x,y,z-1,t-1;z
    if (if_b_z_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_z_b_t_recv_vec) +
               ((((_Z_ * 1 + 0) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_z / lat_t);
    } else {
      move0 = move_wards[_B_Z_];
      move1 = move_wards[_B_T_];
      tmp_U =
          (origin_U + move0 * lat_y * lat_x + move1 * lat_z * lat_y * lat_x +
           (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    ////x,y,z,t-1;t
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_t_recv_vec) +
               ((((_T_ * 1 + 0) * lat_z + z) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    ////x,y,z,t-1;t;dag
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_t_recv_vec) +
               ((((_T_ * 1 + 0) * lat_z + z) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    ////x,y,z,t-1;z
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_b_t_recv_vec) +
               ((((_Z_ * 1 + 0) * lat_z + z) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp2, tmp_U, lat_tzyx / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
    }
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    ////x,y,z+1,t-1;t
    if (if_f_z_b_t) {
      tmp_U = (static_cast<LatticeComplex *>(device_u_f_z_b_t_recv_vec) +
               ((((_T_ * 1 + 0) * 1 + 0) * lat_y + y) * lat_x + x));
      _give_u_comm(tmp1, tmp_U, lat_tzyx / lat_z / lat_t);
      if (x == 2 && y == 7) {
        // printf("@@@ptr:%p\n", tmp_U);
        for (int cc = 0; cc < _LAT_CC_; cc++) {
          printf("TMP1[%d]:@%d-#x:%d#y:%d#z:%d#t:%d#parity:%d#real:%f\n", cc,
                 node_rank, x, y, z, t, parity,
                 tmp1[cc]._data.x); // test
          printf("TMP1[%d]:@%d-#x:%d#y:%d#z:%d#t:%d#parity:%d#imag:%f\n", cc,
                 node_rank, x, y, z, t, parity,
                 tmp1[cc]._data.y); // test
        }
      }
    } else {
      move0 = move_wards[_F_Z_];
      move1 = move_wards[_B_T_];
      tmp_U =
          (origin_U + move0 * lat_y * lat_x + move1 * lat_z * lat_y * lat_x +
           (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
    if (if_f_z_b_t) {
      if (x == 2 && y == 7) {
        // printf("@@@ptr:%p\n", tmp_U);
        for (int cc = 0; cc < _LAT_CC_; cc++) {
          printf("TMP2[%d]:@%d-#x:%d#y:%d#z:%d#t:%d#parity:%d#real:%f\n", cc,
                 node_rank, x, y, z, t, parity,
                 tmp2[cc]._data.x); // test
          printf("TMP2[%d]:@%d-#x:%d#y:%d#z:%d#t:%d#parity:%d#imag:%f\n", cc,
                 node_rank, x, y, z, t, parity,
                 tmp2[cc]._data.y); // test
        }
      }
    }
  }
  {
    ////x,y,z,t;z;dag
    tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        clover[c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * I;
        clover[39 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * (-I);
        clover[78 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * (-I);
        clover[117 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()) * I;
      }
    }
  }
  {
    // A=1+T
    LatticeComplex one(1.0, 0);
    for (int i = 0; i < _LAT_SCSC_; i++) {
      clover[i] *= -0.125; //-1/8
    }
    for (int i = 0; i < _LAT_SC_; i++) {
      clover[i * 13] += one;
    }
  }
  give_clr(origin_clover, clover, lat_tzyx);
}
#endif
// debug code
/*
{
  // test
  // if_b_x = 0;
  // if_b_y = 0;
  // if_b_z = 0;
  // if_b_t = 0;// BUG!!!
  // if_f_x = 0;
  // if_f_y = 0;
  // if_f_z = 0;
  // if_f_t = 0;
  // if_b_x_b_y = 0;
  // if_f_x_b_y = 0;
  // if_b_x_f_y = 0;
  // // // if_f_x_f_y=0;
  // if_b_x_b_z = 0;
  // if_f_x_b_z = 0;
  // if_b_x_f_z = 0;
  // // // if_f_x_f_z=0;
  // if_b_x_b_t = 0;
  // if_f_x_b_t = 0;
  // if_b_x_f_t = 0;
  // // // if_f_x_f_t=0;
  // if_b_y_b_z = 0;
  // if_f_y_b_z = 0;
  // if_b_y_f_z = 0;
  // // // if_f_y_f_z=0;
  // if_b_y_b_t = 0;
  // if_f_y_b_t = 0;
  // if_b_y_f_t = 0;
  // // // if_f_y_f_t=0;
  // if_b_z_b_t = 0;
  // if_f_z_b_t = 0;
  // if_b_z_f_t = 0;
  // // // if_f_z_f_t=0;
}
if (x == 2 && y == 7 && z == 3) {
  // printf("@@@ptr:%p\n", tmp_U);
  printf("@%d-#x:%d#y:%d#z:%d#t:%d#parity:%d#real:%f\n", node_rank, x, y,
         z, t, parity,
         tmp_U[0]._data.x); // test
  printf("@%d-#x:%d#y:%d#z:%d#t:%d#parity:%d#imag:%f\n", node_rank, x, y,
         z, t, parity,
         tmp_U[0]._data.y); // test
}

*/