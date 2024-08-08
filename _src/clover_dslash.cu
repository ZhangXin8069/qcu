#include "../include/qcu.h"
#ifdef CLOVER_DSLASH

__global__ void make_clover(void *device_U, void *device_clover,
                            void *device_xyztsc, const int device_parity) {
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  const int lat_x = xyztsc[_X_];
  const int lat_y = xyztsc[_Y_];
  const int lat_z = xyztsc[_Z_];
  const int lat_t = xyztsc[_T_];
  const int lat_xcc = xyztsc[_XCC_];
  const int lat_yxcc = xyztsc[_YXCC_];
  const int lat_zyxcc = xyztsc[_ZYXCC_];
  const int lat_tzyxcc = xyztsc[_TZYXCC_];
  const int lat_xsc = xyztsc[_XSC_];
  const int lat_yxsc = xyztsc[_YXSC_];
  const int lat_zyxsc = xyztsc[_ZYXSC_];
  int move0;
  int move1;
  move0 = lat_x * lat_y * lat_z;
  const int t = parity / move0;
  parity -= t * move0;
  move0 = lat_x * lat_y;
  const int z = parity / move0;
  parity -= z * move0;
  const int y = parity / lat_x;
  const int x = parity - y * lat_x;
  const int eo = (y + z + t) & 0x01; // (y+z+t)%2
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * _LAT_CC_);
  LatticeComplex *origin_clover =
      ((static_cast<LatticeComplex *>(device_clover)) +
       (t * lat_zyxsc + z * lat_yxsc + y * lat_xsc + x * _LAT_SC_) * _LAT_SC_);
  LatticeComplex *tmp_U;
  LatticeComplex tmp1[_LAT_CC_];
  LatticeComplex tmp2[_LAT_CC_];
  LatticeComplex tmp3[_LAT_CC_];
  LatticeComplex U[_LAT_CC_];
  LatticeComplex clover[_LAT_SCSC_];
  // sigmaF
  {
    parity = device_parity;
    host_give_value(clover, zero, _LAT_SCSC_);
    host_give_value(origin_clover, zero, _LAT_SCSC_);
    host_give_value(tmp1, zero, _LAT_CC_);
    host_give_value(tmp2, zero, _LAT_CC_);
  }
  // XY
  host_give_value(U, zero, _LAT_CC_);
  {
    //// x,y,z,t;x
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x+1,y,z,t;y
    move_forward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * _LAT_CC_ + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t;x;dag
    move_forward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;y
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y+1,z,t;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_forward(move1, y, lat_y);
    tmp_U =
        (origin_U + move0 * _LAT_CC_ + move1 * lat_xcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;y;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * _LAT_CC_ + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * _LAT_CC_ + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x-1,y,z,t;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * _LAT_CC_ + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y-1,z,t;y;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, y, lat_y);
    tmp_U = (origin_U + move0 * _LAT_CC_ + move1 * lat_xcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y-1,z,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, y, lat_y);
    tmp_U =
        (origin_U + move0 * _LAT_CC_ + move1 * lat_xcc + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x,y-1,z,t;y;dag
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z,t;x
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y-1,z,t;y
    move_forward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, y, lat_y);
    tmp_U = (origin_U + move0 * _LAT_CC_ + move1 * lat_xcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
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
  host_give_value(U, zero, _LAT_CC_);
  {
    //// x,y,z,t;x
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x+1,y,z,t;z
    move_forward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * _LAT_CC_ + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;x;dag
    move_forward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_yxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;z
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z+1,t;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_forward(move1, z, lat_z);
    tmp_U =
        (origin_U + move0 * _LAT_CC_ + move1 * lat_yxcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;z;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * _LAT_CC_ + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * _LAT_CC_ + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x-1,y,z,t;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * _LAT_CC_ + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z-1,t;z;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, z, lat_z);
    tmp_U = (origin_U + move0 * _LAT_CC_ + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z-1,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, z, lat_z);
    tmp_U =
        (origin_U + move0 * _LAT_CC_ + move1 * lat_yxcc + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x,y,z-1,t;z;dag
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t;x
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_yxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z-1,t;z
    move_forward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, z, lat_z);
    tmp_U = (origin_U + move0 * _LAT_CC_ + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
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
  host_give_value(U, zero, _LAT_CC_);
  {
    //// x,y,z,t;x
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x+1,y,z,t;t
    move_forward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * _LAT_CC_ + lat_tzyxcc * _LAT_HALF_SC_ +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;x;dag
    move_forward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    tmp_U = (origin_U + lat_tzyxcc * _LAT_HALF_SC_ + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;t
    tmp_U = (origin_U + lat_tzyxcc * _LAT_HALF_SC_ + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z,t+1;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_forward(move1, t, lat_t);
    tmp_U =
        (origin_U + move0 * _LAT_CC_ + move1 * lat_zyxcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;t;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * _LAT_CC_ + lat_tzyxcc * _LAT_HALF_SC_ +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * _LAT_CC_ + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x-1,y,z,t;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * _LAT_CC_ + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z,t-1;t;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * _LAT_CC_ + move1 * lat_zyxcc +
             lat_tzyxcc * _LAT_HALF_SC_ + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t-1;x
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, t, lat_t);
    tmp_U =
        (origin_U + move0 * _LAT_CC_ + move1 * lat_zyxcc + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * _LAT_HALF_SC_ +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t-1;t;dag
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * _LAT_HALF_SC_ +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z,t-1;x
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z,t-1;t
    move_forward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * _LAT_CC_ + move1 * lat_zyxcc +
             lat_tzyxcc * _LAT_HALF_SC_ + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
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
  host_give_value(U, zero, _LAT_CC_);
  {
    //// x,y,z,t;y
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y+1,z,t;z
    move_forward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;y;dag
    move_forward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;z
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z+1,t;y;dag
    move_backward(move0, y, lat_y);
    move_forward(move1, z, lat_z);
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;z;dag
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x,y-1,z,t;y;dag
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z-1,t;z;dag
    move_backward(move0, y, lat_y);
    move_backward(move1, z, lat_z);
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z-1,t;y
    move_backward(move0, y, lat_y);
    move_backward(move1, z, lat_z);
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x,y,z-1,t;z;dag
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t;y
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z-1,t;z
    move_forward(move0, y, lat_y);
    move_backward(move1, z, lat_z);
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
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
  host_give_value(U, zero, _LAT_CC_);
  {
    //// x,y,z,t;y
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y+1,z,t;t
    move_forward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * _LAT_HALF_SC_ +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;y;dag
    move_forward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    tmp_U = (origin_U + lat_tzyxcc * _LAT_HALF_SC_ + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;t
    tmp_U = (origin_U + lat_tzyxcc * _LAT_HALF_SC_ + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z,t+1;y;dag
    move_backward(move0, y, lat_y);
    move_forward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;t;dag
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * _LAT_HALF_SC_ +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x,y-1,z,t;y;dag
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z,t-1;t;dag
    move_backward(move0, y, lat_y);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc +
             lat_tzyxcc * _LAT_HALF_SC_ + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t-1;y
    move_backward(move0, y, lat_y);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * _LAT_HALF_SC_ +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t-1;t;dag
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * _LAT_HALF_SC_ +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z,t-1;y
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t-1;t
    move_forward(move0, y, lat_y);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc +
             lat_tzyxcc * _LAT_HALF_SC_ + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
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
  host_give_value(U, zero, _LAT_CC_);
  {
    //// x,y,z,t;z
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z+1,t;t
    move_forward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * _LAT_HALF_SC_ +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;z;dag
    move_forward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    tmp_U = (origin_U + lat_tzyxcc * _LAT_HALF_SC_ + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;t
    tmp_U = (origin_U + lat_tzyxcc * _LAT_HALF_SC_ + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t+1;z;dag
    move_backward(move0, z, lat_z);
    move_forward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t;t;dag
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * _LAT_HALF_SC_ +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x,y,z-1,t;z;dag
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t-1;t;dag
    move_backward(move0, z, lat_z);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc +
             lat_tzyxcc * _LAT_HALF_SC_ + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t-1;z
    move_backward(move0, z, lat_z);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * _LAT_HALF_SC_ +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t-1;t;dag
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * _LAT_HALF_SC_ +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z,t-1;z
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t-1;t
    move_forward(move0, z, lat_z);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc +
             lat_tzyxcc * _LAT_HALF_SC_ + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, _LAT_CC_);
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
  give_ptr(origin_clover, clover, _LAT_SCSC_);
}

__global__ void inverse_clover(void *device_clover, void *device_xyztsc) {
  LatticeComplex *origin_clover;
  {
    int *xyztsc = static_cast<int *>(device_xyztsc);
    const int lat_x = xyztsc[_X_];
    const int lat_y = xyztsc[_Y_];
    const int lat_z = xyztsc[_Z_];
    const int lat_xsc = xyztsc[_XSC_];
    const int lat_yxsc = xyztsc[_YXSC_];
    const int lat_zyxsc = xyztsc[_ZYXSC_];
    int tmp1;
    int tmp2 = blockIdx.x * blockDim.x + threadIdx.x;
    tmp1 = lat_x * lat_y * lat_z;
    const int t = tmp2 / tmp1;
    tmp2 -= t * tmp1;
    tmp1 = lat_x * lat_y;
    const int z = tmp2 / tmp1;
    tmp2 -= z * tmp1;
    const int y = tmp2 / lat_x;
    const int x = tmp2 - y * lat_x;
    origin_clover =
        ((static_cast<LatticeComplex *>(device_clover)) +
         (t * lat_zyxsc + z * lat_yxsc + y * lat_xsc + x * _LAT_SC_) *
             _LAT_SC_);
  }
  {
    LatticeComplex pivot;
    LatticeComplex factor;
    LatticeComplex clover[_LAT_SCSC_];
    LatticeComplex augmented_clover[_LAT_SCSC_ * _BF_];
    give_ptr(clover, origin_clover, _LAT_SCSC_);
    inverse(clover, clover, augmented_clover, pivot, factor, _LAT_SC_);
    give_ptr(origin_clover, clover, _LAT_SCSC_);
  }
}

__global__ void give_clover(void *device_clover, void *device_dest,
                            void *device_xyztsc) {
  LatticeComplex *origin_clover;
  LatticeComplex *origin_dest;
  {
    int *xyztsc = static_cast<int *>(device_xyztsc);
    const int lat_x = xyztsc[_X_];
    const int lat_y = xyztsc[_Y_];
    const int lat_z = xyztsc[_Z_];
    const int lat_xsc = xyztsc[_XSC_];
    const int lat_yxsc = xyztsc[_YXSC_];
    const int lat_zyxsc = xyztsc[_ZYXSC_];
    int tmp1;
    int tmp2 = blockIdx.x * blockDim.x + threadIdx.x;
    tmp1 = lat_x * lat_y * lat_z;
    const int t = tmp2 / tmp1;
    tmp2 -= t * tmp1;
    tmp1 = lat_x * lat_y;
    const int z = tmp2 / tmp1;
    tmp2 -= z * tmp1;
    const int y = tmp2 / lat_x;
    const int x = tmp2 - y * lat_x;
    origin_clover =
        ((static_cast<LatticeComplex *>(device_clover)) +
         (t * lat_zyxsc + z * lat_yxsc + y * lat_xsc + x * _LAT_SC_) *
             _LAT_SC_);
    origin_dest = ((static_cast<LatticeComplex *>(device_dest)) +
                   t * lat_zyxsc + z * lat_yxsc + y * lat_xsc + x * _LAT_SC_);
  }
  {
    LatticeComplex clover[_LAT_SCSC_];
    LatticeComplex dest[_LAT_SC_];
    LatticeComplex tmp_dest[_LAT_SC_];
    give_ptr(dest, origin_dest, _LAT_SC_);
    give_ptr(clover, origin_clover, _LAT_SCSC_);
    for (int sc0 = 0; sc0 < _LAT_SC_; sc0++) {
      for (int sc1 = 0; sc1 < _LAT_SC_; sc1++) {
        tmp_dest[sc0] += clover[sc0 * _LAT_SC_ + sc1] * dest[sc1];
      }
    }
    give_ptr(origin_dest, tmp_dest, _LAT_SC_);
  }
}

void dslashCloverQcu(void *fermion_out, void *fermion_in, void *gauge,
                     QcuParam *param, int parity) {
  // define for nccl_clover_dslash
  LatticeSet _set;
  _set.give(param->lattice_size);
  _set.init();
  LatticeWilsonDslash _dslash;
  _dslash.give(&_set);
  void *clover;
  checkCudaErrors(cudaMalloc(&clover, (_set.lat_4dim * _LAT_SCSC_) *
                                          sizeof(LatticeComplex)));
  cudaError_t err;
  dim3 gridDim(_set.lat_4dim / _BLOCK_SIZE_);
  dim3 blockDim(_BLOCK_SIZE_);
  {
    // wilson dslash
    _dslash.run_test(fermion_out, fermion_in, gauge, parity);
  }
  {
    // make clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    make_clover<<<gridDim, blockDim>>>(gauge, clover, _set.device_xyztsc,
                                       parity);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    err = cudaGetLastError();
    checkCudaErrors(err);
    printf("make clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  {
    // inverse clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    inverse_clover<<<gridDim, blockDim>>>(clover, _set.device_xyztsc);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    err = cudaGetLastError();
    checkCudaErrors(err);
    printf(
        "inverse clover total time: (without malloc free memcpy) :%.9lf sec\n ",
        double(duration) / 1e9);
  }
  {
    // give clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    give_clover<<<gridDim, blockDim>>>(clover, fermion_out, _set.device_xyztsc);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    err = cudaGetLastError();
    checkCudaErrors(err);
    printf("give clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  {
    _set.end();
    // free
    checkCudaErrors(cudaFree(clover));
  }
}

#endif