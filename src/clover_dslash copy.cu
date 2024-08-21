#include "../include/qcu.h"
#include "define.h"
__global__ void make_clover_inside(void *device_U, void *device_clover,
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
  // just inside
  int if_inside_x;
  int if_inside_y;
  int if_inside_z;
  int if_inside_t;
  move_backward_x(move0, x, lat_x, eo, parity);
  move_forward_x(move1, x, lat_x, eo, parity);
  if_inside_x = (move0 != lat_x - 1) * (move1 != 1 - lat_x); // even-odd
  move_backward(move0, y, lat_y);
  move_forward(move1, y, lat_y);
  if_inside_y = (move0 == -1) * (move1 == 1);
  move_backward(move0, z, lat_z);
  move_forward(move1, z, lat_z);
  if_inside_z = (move0 == -1) * (move1 == 1);
  move_backward(move0, t, lat_t);
  move_forward(move1, t, lat_t);
  if_inside_t = (move0 == -1) * (move1 == 1);
  // sigmaF
  {
    parity = device_parity;
    give_vals(clover, zero, _LAT_SCSC_);
    // give_vals(origin_clover, zero, _LAT_SCSC_);//BUG!!!!!!
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
    move_forward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 + (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t;x;dag
    move_forward(move0, y, lat_y);
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
    move_backward_x(move0, x, lat_x, eo, parity);
    move_forward(move1, y, lat_y);
    tmp_U = (origin_U + move0 + move1 * lat_x +
             (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;y;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 + (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x-1,y,z,t;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x-1,y-1,z,t;y;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, y, lat_y);
    tmp_U = (origin_U + move0 + move1 * lat_x +
             (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y-1,z,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, y, lat_y);
    tmp_U = (origin_U + move0 + move1 * lat_x +
             (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y-1,z,t;y;dag
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y-1,z,t;x
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_x +
             (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y-1,z,t;y
    move_forward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, y, lat_y);
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
    for (int c0 = 0; c0 < _LAT_C_ * if_inside_x * if_inside_y; c0++) {
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
    move_forward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 + (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;x;dag
    move_forward(move0, z, lat_z);
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
    move_backward_x(move0, x, lat_x, eo, parity);
    move_forward(move1, z, lat_z);
    tmp_U = (origin_U + move0 + move1 * lat_y * lat_x +
             (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;z;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 + (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x-1,y,z,t;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x-1,y,z-1,t;z;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, z, lat_z);
    tmp_U = (origin_U + move0 + move1 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z-1,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, z, lat_z);
    tmp_U = (origin_U + move0 + move1 * lat_y * lat_x +
             (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z-1,t;z;dag
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y,z-1,t;x
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z-1,t;z
    move_forward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, z, lat_z);
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
    for (int c0 = 0; c0 < _LAT_C_ * if_inside_x * if_inside_z; c0++) {
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
    move_forward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 + (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;x;dag
    move_forward(move0, t, lat_t);
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
    move_backward_x(move0, x, lat_x, eo, parity);
    move_forward(move1, t, lat_t);
    tmp_U = (origin_U + move0 + move1 * lat_z * lat_y * lat_x +
             (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;t;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 + (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x-1,y,z,t;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x-1,y,z,t-1;t;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 + move1 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t-1;x
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 + move1 * lat_z * lat_y * lat_x +
             (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t-1;t;dag
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y,z,t-1;x
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z,t-1;t
    move_forward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, t, lat_t);
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
    for (int c0 = 0; c0 < _LAT_C_ * if_inside_x * if_inside_t; c0++) {
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
    move_forward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;y;dag
    move_forward(move0, z, lat_z);
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
    move_backward(move0, y, lat_y);
    move_forward(move1, z, lat_z);
    tmp_U = (origin_U + move0 * lat_x + move1 * lat_y * lat_x +
             (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;z;dag
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y-1,z,t;y;dag
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y-1,z-1,t;z;dag
    move_backward(move0, y, lat_y);
    move_backward(move1, z, lat_z);
    tmp_U = (origin_U + move0 * lat_x + move1 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z-1,t;y
    move_backward(move0, y, lat_y);
    move_backward(move1, z, lat_z);
    tmp_U = (origin_U + move0 * lat_x + move1 * lat_y * lat_x +
             (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z-1,t;z;dag
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y,z-1,t;y
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z-1,t;z
    move_forward(move0, y, lat_y);
    move_backward(move1, z, lat_z);
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
    for (int c0 = 0; c0 < _LAT_C_ * if_inside_y * if_inside_z; c0++) {
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
    move_forward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;y;dag
    move_forward(move0, t, lat_t);
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
    move_backward(move0, y, lat_y);
    move_forward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_x + move1 * lat_z * lat_y * lat_x +
             (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;t;dag
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y-1,z,t;y;dag
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y-1,z,t-1;t;dag
    move_backward(move0, y, lat_y);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_x + move1 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t-1;y
    move_backward(move0, y, lat_y);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_x + move1 * lat_z * lat_y * lat_x +
             (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t-1;t;dag
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y,z,t-1;y
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t-1;t
    move_forward(move0, y, lat_y);
    move_backward(move1, t, lat_t);
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
    for (int c0 = 0; c0 < _LAT_C_ * if_inside_y * if_inside_t; c0++) {
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
    move_forward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;z;dag
    move_forward(move0, t, lat_t);
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
    move_backward(move0, z, lat_z);
    move_forward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_y * lat_x + move1 * lat_z * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t;t;dag
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z-1,t;z;dag
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y,z-1,t-1;t;dag
    move_backward(move0, z, lat_z);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_y * lat_x + move1 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t-1;z
    move_backward(move0, z, lat_z);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_y * lat_x + move1 * lat_z * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t-1;t;dag
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp1, tmp_U, lat_tzyx);
    //// x,y,z,t-1;z
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
             (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
    give_u(tmp2, tmp_U, lat_tzyx);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t-1;t
    move_forward(move0, z, lat_z);
    move_backward(move1, t, lat_t);
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
    for (int c0 = 0; c0 < _LAT_C_ * if_inside_z * if_inside_t; c0++) {
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

__global__ void pick_up_1dim_b_x(void *device_U, void *device_lat_xyzt,
                                 void *device_u_1dim_send_vec,
                                 int device_parity) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int parity = idx;
  int *lat_xyzt = static_cast<int *>(device_lat_xyzt);
  int lat_x = 1; // to 3dim format
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
  parity = device_parity;
  x = 0; // b_x
  LatticeComplex *origin_U = ((static_cast<LatticeComplex *>(device_U)) +
                              (((t)*lat_z + z) * lat_y + y) * lat_x + x +
                              parity * lat_tzyx); // 4dim format
  LatticeComplex *origin_u_1dim_send_vec =
      ((static_cast<LatticeComplex *>(device_u_1dim_send_vec)) +
       idx); // 3dim format
  for (int i = 0; i < _LAT_CC_; i++) {
    origin_u_1dim_send_vec[i * lat_tzyx / lat_xyzt[_X_]] =
        origin_U[i * _LAT_D_ * _EVEN_ODD_ * lat_tzyx];
  }
}