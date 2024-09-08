// clang-format off
#include "../include/qcu.h"
#include "define.h"
#ifdef CLOVER_DSLASH
// wait for rebuild
// clang-format on
__global__ void make_clover(void *device_U, void *device_clover,
                            void *device_lat_xyzt, int device_parity) {
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
  int eo = (y + z + t) & 0x01; // (y+z+t)%2
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
  // sigmaF
  {
    give_vals(clover, zero, _LAT_SCSC_);
    give_vals(tmp1, zero, _LAT_CC_);
    give_vals(tmp2, zero, _LAT_CC_);
  }
  // XY
  give_vals(U, zero, _LAT_CC_);
  {
    //// x,y,z,t;x
    tmp_U = (origin_U + (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x+1,y,z,t;y
    move0 = move_wards[_F_X_];
    tmp_U = (origin_U + move0 + (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t;x;dag
    move0 = move_wards[_F_Y_];
    tmp_U = (origin_U + move0 * lat_x +
             (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;y
    tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x-1,y+1,z,t;x;dag
    move0 = move_wards[_B_X_];
    move1 = move_wards[_F_Y_];
    tmp_U = (origin_U + move0 + move1 * lat_x +
             (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;y;dag
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 + (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x-1,y,z,t;x;dag
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x-1,y-1,z,t;y;dag
    move0 = move_wards[_B_X_];
    move1 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 + move1 * lat_x +
             (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y-1,z,t;x
    move0 = move_wards[_B_X_];
    move1 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 + move1 * lat_x +
             (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y-1,z,t;y;dag
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y-1,z,t;x
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_x +
             (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y-1,z,t;y
    move0 = move_wards[_F_X_];
    move1 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 + move1 * lat_x +
             (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
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
    //// x,y,z,t;x
    tmp_U = (origin_U + (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x+1,y,z,t;z
    move0 = move_wards[_F_X_];
    tmp_U = (origin_U + move0 + (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;x;dag
    move0 = move_wards[_F_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;z
    tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x-1,y,z+1,t;x;dag
    move0 = move_wards[_B_X_];
    move1 = move_wards[_F_Z_];
    tmp_U = (origin_U + move0 + move1 * lat_y * lat_x +
             (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;z;dag
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 + (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x-1,y,z,t;x;dag
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x-1,y,z-1,t;z;dag
    move0 = move_wards[_B_X_];
    move1 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 + move1 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z-1,t;x
    move0 = move_wards[_B_X_];
    move1 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 + move1 * lat_y * lat_x +
             (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z-1,t;z;dag
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y,z-1,t;x
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z-1,t;z
    move0 = move_wards[_F_X_];
    move1 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 + move1 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
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
    //// x,y,z,t;x
    tmp_U = (origin_U + (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x+1,y,z,t;t
    move0 = move_wards[_F_X_];
    tmp_U = (origin_U + move0 + (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;x;dag
    move0 = move_wards[_F_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;t
    tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x-1,y,z,t+1;x;dag
    move0 = move_wards[_B_X_];
    move1 = move_wards[_F_T_];
    tmp_U = (origin_U + move0 + move1 * lat_z * lat_y * lat_x +
             (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;t;dag
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 + (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x-1,y,z,t;x;dag
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x-1,y,z,t-1;t;dag
    move0 = move_wards[_B_X_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 + move1 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t-1;x
    move0 = move_wards[_B_X_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 + move1 * lat_z * lat_y * lat_x +
             (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t-1;t;dag
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y,z,t-1;x
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z,t-1;t
    move0 = move_wards[_F_X_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 + move1 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
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
    //// x,y,z,t;y
    tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y+1,z,t;z
    move0 = move_wards[_F_Y_];
    tmp_U = (origin_U + move0 * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;y;dag
    move0 = move_wards[_F_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;z
    tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y-1,z+1,t;y;dag
    move0 = move_wards[_B_Y_];
    move1 = move_wards[_F_Z_];
    tmp_U = (origin_U + move0 * lat_x + move1 * lat_y * lat_x +
             (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;z;dag
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y-1,z,t;y;dag
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y-1,z-1,t;z;dag
    move0 = move_wards[_B_Y_];
    move1 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_x + move1 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z-1,t;y
    move0 = move_wards[_B_Y_];
    move1 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_x + move1 * lat_y * lat_x +
             (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z-1,t;z;dag
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y,z-1,t;y
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z-1,t;z
    move0 = move_wards[_F_Y_];
    move1 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_x + move1 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
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
    //// x,y,z,t;y
    tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y+1,z,t;t
    move0 = move_wards[_F_Y_];
    tmp_U = (origin_U + move0 * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;y;dag
    move0 = move_wards[_F_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;t
    tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y-1,z,t+1;y;dag
    move0 = move_wards[_B_Y_];
    move1 = move_wards[_F_T_];
    tmp_U = (origin_U + move0 * lat_x + move1 * lat_z * lat_y * lat_x +
             (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;t;dag
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y-1,z,t;y;dag
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y-1,z,t-1;t;dag
    move0 = move_wards[_B_Y_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_x + move1 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t-1;y
    move0 = move_wards[_B_Y_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_x + move1 * lat_z * lat_y * lat_x +
             (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t-1;t;dag
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y,z,t-1;y
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t-1;t
    move0 = move_wards[_F_Y_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_x + move1 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
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
    //// x,y,z,t;z
    tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y,z+1,t;t
    move0 = move_wards[_F_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;z;dag
    move0 = move_wards[_F_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;t
    tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y,z-1,t+1;z;dag
    move0 = move_wards[_B_Z_];
    move1 = move_wards[_F_T_];
    tmp_U = (origin_U + move0 * lat_y * lat_x + move1 * lat_z * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t;t;dag
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z-1,t;z;dag
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y,z-1,t-1;t;dag
    move0 = move_wards[_B_Z_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_y * lat_x + move1 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t-1;z
    move0 = move_wards[_B_Z_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_y * lat_x + move1 * lat_z * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t-1;t;dag
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y,z,t-1;z
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t-1;t
    move0 = move_wards[_F_Z_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_y * lat_x + move1 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
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
__global__ void inverse_clover(void *device_clover, void *device_lat_xyzt) {
  LatticeComplex *origin_clover;
  int lat_tzyx = static_cast<int *>(device_lat_xyzt)[_XYZT_];
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    origin_clover = ((static_cast<LatticeComplex *>(device_clover)) + idx);
  }
  {
    LatticeComplex pivot;
    LatticeComplex factor;
    LatticeComplex clover[_LAT_SCSC_];
    LatticeComplex augmented_clover[_LAT_SCSC_ * _BF_];
    get_clr(clover, origin_clover, lat_tzyx);
    _inverse(clover, clover, augmented_clover, pivot, factor, _LAT_SC_);
    give_clr(origin_clover, clover, lat_tzyx);
  }
}
__global__ void give_clover(void *device_clover, void *device_dest,
                            void *device_lat_xyzt) {
  LatticeComplex *origin_clover;
  LatticeComplex *origin_dest;
  int lat_tzyx = static_cast<int *>(device_lat_xyzt)[_XYZT_];
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    origin_clover = ((static_cast<LatticeComplex *>(device_clover)) + idx);
    origin_dest = ((static_cast<LatticeComplex *>(device_dest)) + idx);
  }
  {
    LatticeComplex clover[_LAT_SCSC_];
    LatticeComplex dest[_LAT_SC_];
    LatticeComplex tmp_dest[_LAT_SC_];
    LatticeComplex zero(0.0, 0.0);
    give_vals(tmp_dest, zero, _LAT_SC_);
    give_src(dest, origin_dest, lat_tzyx);
    get_clr(clover, origin_clover, lat_tzyx);
    for (int sc0 = 0; sc0 < _LAT_SC_; sc0++) {
      for (int sc1 = 0; sc1 < _LAT_SC_; sc1++) {
        tmp_dest[sc0] += clover[sc0 * _LAT_SC_ + sc1] * dest[sc1];
      }
    }
    give_dest(origin_dest, tmp_dest, lat_tzyx);
  }
}
#endif