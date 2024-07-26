#include "../../include/qcu.h"
#ifdef CLOVER_DSLASH
__global__ void make_clover(void *device_U, void *device_clover,
                            int device_lat_x, const int device_lat_y,
                            const int device_lat_z, const int device_lat_t,
                            const int device_parity) {
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  const int lat_xcc = lat_x * 9;
  const int lat_yxcc = lat_y * lat_xcc;
  const int lat_zyxcc = lat_z * lat_yxcc;
  const int lat_tzyxcc = lat_t * lat_zyxcc;
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
       z * lat_yxcc + y * lat_xcc + x * 9);
  LatticeComplex *origin_clover =
      ((static_cast<LatticeComplex *>(device_clover)) + t * lat_zyxcc * 16 +
       z * lat_yxcc * 16 + y * lat_xcc * 16 + x * 144);
  LatticeComplex *tmp_U;
  LatticeComplex tmp1[9];
  LatticeComplex tmp2[9];
  LatticeComplex tmp3[9];
  LatticeComplex U[9];
  LatticeComplex clover[144];
  // sigmaF
  {
    parity = device_parity;
    host_give_value(clover, zero, 144);
    host_give_value(origin_clover, zero, 144);
    host_give_value(tmp1, zero, 9);
    host_give_value(tmp2, zero, 9);
  }
  // XY
  host_give_value(U, zero, 9);
  {
    //// x,y,z,t;x
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x+1,y,z,t;y
    move_forward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 2 + (1 - parity) * lat_tzyxcc);
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
  add_ptr(U, tmp3, 9);
  {
    //// x,y,z,t;y
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y+1,z,t;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_forward(move1, y, lat_y);
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;y;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 2 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, 9);
  {
    //// x-1,y,z,t;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y-1,z,t;y;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, y, lat_y);
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y-1,z,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, y, lat_y);
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + parity * lat_tzyxcc);
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
  add_ptr(U, tmp3, 9);
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
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + lat_tzyxcc * 2 +
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
  add_ptr(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[c0 * 12 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[39 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[78 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[117 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
      }
    }
  }
  // XZ
  host_give_value(U, zero, 9);
  {
    //// x,y,z,t;x
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x+1,y,z,t;z
    move_forward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 4 + (1 - parity) * lat_tzyxcc);
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
  add_ptr(U, tmp3, 9);
  {
    //// x,y,z,t;z
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z+1,t;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_forward(move1, z, lat_z);
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;z;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 4 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, 9);
  {
    //// x-1,y,z,t;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z-1,t;z;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, z, lat_z);
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z-1,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, z, lat_z);
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + parity * lat_tzyxcc);
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
  add_ptr(U, tmp3, 9);
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
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + lat_tzyxcc * 4 +
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
  add_ptr(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[3 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
        clover[36 + c0 * 12 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
        clover[81 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
        clover[114 + c0 * 12 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
      }
    }
  }
  // XT
  host_give_value(U, zero, 9);
  {
    //// x,y,z,t;x
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x+1,y,z,t;t
    move_forward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 6 + (1 - parity) * lat_tzyxcc);
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
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, 9);
  {
    //// x,y,z,t;t
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z,t+1;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_forward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;t;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 6 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, 9);
  {
    //// x-1,y,z,t;x;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z,t-1;t;dag
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t-1;x
    move_backward_x(move0, x, lat_x, eo, parity);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, 9);
  {
    //// x,y,z,t-1;t;dag
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
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
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + lat_tzyxcc * 6 +
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
  add_ptr(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[3 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[36 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[81 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[114 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
      }
    }
  }
  // YZ
  host_give_value(U, zero, 9);
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
  add_ptr(U, tmp3, 9);
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
  add_ptr(U, tmp3, 9);
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
  add_ptr(U, tmp3, 9);
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
  add_ptr(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[3 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[36 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[81 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[114 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
      }
    }
  }
  // YT
  host_give_value(U, zero, 9);
  {
    //// x,y,z,t;y
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y+1,z,t;t
    move_forward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 6 +
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
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, 9);
  {
    //// x,y,z,t;t
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
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
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 6 +
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
  add_ptr(U, tmp3, 9);
  {
    //// x,y-1,z,t;y;dag
    move_backward(move0, y, lat_y);
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z,t-1;t;dag
    move_backward(move0, y, lat_y);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
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
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, 9);
  {
    //// x,y,z,t-1;t;dag
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
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
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
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
  add_ptr(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[3 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
        clover[36 + c0 * 12 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
        clover[81 + c0 * 12 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
        clover[114 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
      }
    }
  }

  // ZT
  host_give_value(U, zero, 9);
  {
    //// x,y,z,t;z
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z+1,t;t
    move_forward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 6 +
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
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, 9);
  {
    //// x,y,z,t;t
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
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
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 6 +
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
  add_ptr(U, tmp3, 9);
  {
    //// x,y,z-1,t;z;dag
    move_backward(move0, z, lat_z);
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t-1;t;dag
    move_backward(move0, z, lat_z);
    move_backward(move1, t, lat_t);
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
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
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, 9);
  {
    //// x,y,z,t-1;t;dag
    move_backward(move0, t, lat_t);
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
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
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_ptr(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[c0 * 12 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[39 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[78 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[117 + c0 * 12 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
      }
    }
  }
  {
    // A=1+T
    LatticeComplex one(1.0, 0);
    for (int i = 0; i < 144; i++) {
      clover[i] *= -0.125; //-1/8
    }
    for (int i = 0; i < 12; i++) {
      clover[i * 13] += one;
    }
  }
  give_ptr(origin_clover, clover, 144);
}

__global__ void inverse_clover(void *device_clover, int device_lat_x,
                               const int device_lat_y, const int device_lat_z) {
  LatticeComplex *origin_clover;
  {
    const int lat_x = device_lat_x;
    const int lat_y = device_lat_y;
    const int lat_z = device_lat_z;
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
    origin_clover = ((static_cast<LatticeComplex *>(device_clover)) +
                     t * lat_z * lat_y * lat_x * 144 + z * lat_y * lat_x * 144 +
                     y * lat_x * 144 + x * 144);
  }
  {
    LatticeComplex pivot;
    LatticeComplex factor;
    LatticeComplex clover[144];
    LatticeComplex augmented_clover[288];
    give_ptr(clover, origin_clover, 144);
    inverse(clover, clover, augmented_clover, pivot, factor, 12);
    give_ptr(origin_clover, clover, 144);
  }
}

__global__ void give_clover(void *device_clover, void *device_dest,
                            int device_lat_x, const int device_lat_y,
                            const int device_lat_z) {
  LatticeComplex *origin_clover;
  LatticeComplex *origin_dest;
  {
    const int lat_x = device_lat_x;
    const int lat_y = device_lat_y;
    const int lat_z = device_lat_z;
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
    origin_clover = ((static_cast<LatticeComplex *>(device_clover)) +
                     t * lat_z * lat_y * lat_x * 144 + z * lat_y * lat_x * 144 +
                     y * lat_x * 144 + x * 144);
    origin_dest = ((static_cast<LatticeComplex *>(device_dest)) +
                   t * lat_z * lat_y * lat_x * 12 + z * lat_y * lat_x * 12 +
                   y * lat_x * 12 + x * 12);
  }
  {
    LatticeComplex clover[144];
    LatticeComplex dest[12];
    LatticeComplex tmp_dest[12];
    give_ptr(dest, origin_dest, 12);
    give_ptr(clover, origin_clover, 144);
    for (int sc0 = 0; sc0 < 12; sc0++) {
      for (int sc1 = 0; sc1 < 12; sc1++) {
        tmp_dest[sc0] += clover[sc0 * 12 + sc1] * dest[sc1];
      }
    }
    give_ptr(origin_dest, tmp_dest, 12);
  }
}

void dslashCloverQcu(void *fermion_out, void *fermion_in, void *gauge,
                     QcuParam *param, int parity) {
  const int lat_x = param->lattice_size[0] >> 1;
  const int lat_y = param->lattice_size[1];
  const int lat_z = param->lattice_size[2];
  const int lat_t = param->lattice_size[3];
  void *clover;
  checkCudaErrors(cudaMalloc(&clover, (lat_t * lat_z * lat_y * lat_x * 144) *
                                          sizeof(LatticeComplex)));
  cudaError_t err;
  dim3 gridDim(lat_x * lat_y * lat_z * lat_t / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  {
    // wilson dslash
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    wilson_dslash<<<gridDim, blockDim>>>(gauge, fermion_in, fermion_out, lat_x,
                                         lat_y, lat_z, lat_t, parity);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    err = cudaGetLastError();
    checkCudaErrors(err);
    printf(
        "wilson dslash total time: (without malloc free memcpy) : %.9lf sec\n",
        double(duration) / 1e9);
  }
  {
    // make clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    make_clover<<<gridDim, blockDim>>>(gauge, clover, lat_x, lat_y, lat_z,
                                       lat_t, parity);
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
    inverse_clover<<<gridDim, blockDim>>>(clover, lat_x, lat_y, lat_z);
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
    give_clover<<<gridDim, blockDim>>>(clover, fermion_out, lat_x, lat_y,
                                       lat_z);
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
    // free
    checkCudaErrors(cudaFree(clover));
  }
}
#endif