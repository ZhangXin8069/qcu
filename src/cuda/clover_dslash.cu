#pragma optimize(5)
#include "../../include/qcu_cuda.h"

__global__ void make_clover(void *device_U, void *device_clover,
                            int device_lat_x, const int device_lat_y,
                            const int device_lat_z, const int device_lat_t,
                            const int device_parity) {
  register int parity = blockIdx.x * blockDim.x + threadIdx.x;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  const int lat_xcc = lat_x * 9;
  const int lat_yxcc = lat_y * lat_xcc;
  const int lat_zyxcc = lat_z * lat_yxcc;
  const int lat_tzyxcc = lat_t * lat_zyxcc;
  register int move0;
  register int move1;
  move0 = lat_x * lat_y * lat_z;
  const int t = parity / move0;
  parity -= t * move0;
  move0 = lat_x * lat_y;
  const int z = parity / move0;
  parity -= z * move0;
  const int y = parity / lat_x;
  const int x = parity - y * lat_x;
  const int oe = (y + z + t) % 2;
  register LatticeComplex I(0.0, 1.0);
  register LatticeComplex zero(0.0, 0.0);
  register LatticeComplex tmp0(0.0, 0.0);
  register LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * 9);
  register LatticeComplex *origin_clover =
      ((static_cast<LatticeComplex *>(device_clover)) + t * lat_zyxcc * 16 +
       z * lat_yxcc * 16 + y * lat_xcc * 16 + x * 144);
  register LatticeComplex *tmp_U;
  register LatticeComplex tmp1[9];
  register LatticeComplex tmp2[9];
  register LatticeComplex tmp3[9];
  register LatticeComplex U[9];
  register LatticeComplex clover[144];
  // sigmaF
  {
    parity = device_parity;
    give_value(clover, zero, 144);
    give_value(origin_clover, zero, 144);
    give_value(tmp1, zero, 9);
    give_value(tmp2, zero, 9);
  }
  // XY
  give_value(U, zero, 9);
  {
    //// x,y,z,t;x
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x+1,y,z,t;y
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe != parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 2 + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t;x;dag
    move0 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;y
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y+1,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;y;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 2 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x-1,y,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y-1,z,t;y;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y-1,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y-1,z,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z,t;x
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y-1,z,t;y
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe == parity);
    move1 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[45 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[90 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[135 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
      }
    }
  }
  // XZ
  give_value(U, zero, 9);
  {
    //// x,y,z,t;x
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x+1,y,z,t;z
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe != parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 4 + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;x;dag
    move0 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;z
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z+1,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;z;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 4 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x-1,y,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z-1,t;z;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z-1,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z-1,t;z;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t;x
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z-1,t;z
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe == parity);
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[9 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
        clover[36 + c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
        clover[99 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
        clover[126 + c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
      }
    }
  }
  // XT
  give_value(U, zero, 9);
  {
    //// x,y,z,t;x
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x+1,y,z,t;t
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe != parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 6 + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;x;dag
    move0 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;t
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z,t+1;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;t;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 6 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x-1,y,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z,t-1;t;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t-1;x
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t-1;t;dag
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z,t-1;x
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z,t-1;t
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe == parity);
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[9 + c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[36 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[99 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[126 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
      }
    }
  }
  // YZ
  give_value(U, zero, 9);
  {
    //// x,y,z,t;y
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y+1,z,t;z
    move0 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;y;dag
    move0 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;z
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z+1,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;z;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y-1,z,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z-1,t;z;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z-1,t;y
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z-1,t;z;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t;y
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z-1,t;z
    move0 = 1 - (y == lat_y - 1) * lat_y;
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[9 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[36 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[99 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[126 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
      }
    }
  }
  // YT
  give_value(U, zero, 9);
  {
    //// x,y,z,t;y
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y+1,z,t;t
    move0 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;y;dag
    move0 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;t
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z,t+1;y;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;t;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y-1,z,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z,t-1;t;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t-1;y
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t-1;t;dag
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z,t-1;y
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t-1;t
    move0 = 1 - (y == lat_y - 1) * lat_y;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[9 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
        clover[36 + c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
        clover[99 + c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
        clover[126 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
      }
    }
  }

  // ZT
  give_value(U, zero, 9);
  {
    //// x,y,z,t;z
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z+1,t;t
    move0 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;z;dag
    move0 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;t
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t+1;z;dag
    move0 = -1 + (z == 0) * lat_z;
    move1 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t;t;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z-1,t;z;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t-1;t;dag
    move0 = -1 + (z == 0) * lat_z;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t-1;z
    move0 = -1 + (z == 0) * lat_z;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t-1;t;dag
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z,t-1;z
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t-1;t
    move0 = 1 - (z == lat_z - 1) * lat_z;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[45 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[90 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[135 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
      }
    }
  }
  {
    // A=1+T
    LatticeComplex one(1.0, 0);
    for (int i = 0; i < 144; i++) {
      clover[i] *= -0.125;
    }
    for (int s = 0; s < 4; s++) {
      for (int c = 0; c < 3; c++) {
        clover[s * 45 + c * 4] += one;
      }
    }
    for (int i = 0; i < 144; i++) {
      origin_clover[i] = clover[i];
    }
  }
}

__global__ void give_clover(void *device_clover, void *device_dest,
                            int device_lat_x, const int device_lat_y,
                            const int device_lat_z) {
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  register LatticeComplex zero(0.0, 0.0);
  register LatticeComplex I(0.0, 1.0);
  register LatticeComplex tmp0;
  register int tmp1;
  register int tmp2 = blockIdx.x * blockDim.x + threadIdx.x;
  tmp1 = lat_x * lat_y * lat_z;
  const int t = tmp2 / tmp1;
  tmp2 -= t * tmp1;
  tmp1 = lat_x * lat_y;
  const int z = tmp2 / tmp1;
  tmp2 -= z * tmp1;
  const int y = tmp2 / lat_x;
  const int x = tmp2 - y * lat_x;
  LatticeComplex pivot;
  LatticeComplex factor;
  LatticeComplex *origin_clover =
      ((static_cast<LatticeComplex *>(device_clover)) +
       t * lat_z * lat_y * lat_x * 144 + z * lat_y * lat_x * 144 +
       y * lat_x * 144 + x * 144);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) +
       t * lat_z * lat_y * lat_x * 12 + z * lat_y * lat_x * 12 +
       y * lat_x * 12 + x * 12);
  LatticeComplex clover[144];
  LatticeComplex augmented_clover[288];
  LatticeComplex dest[12];
  LatticeComplex tmp_dest[12];
  give_ptr(dest, origin_dest, 12);
  give_ptr(clover, origin_clover, 144);
  inverse_clover(clover, clover, augmented_clover, pivot, factor);
  {
    for (int s0 = 0; s0 < 4; s0++) {
      for (int c0 = 0; c0 < 3; c0++) {
        tmp0 = zero;
        for (int s1 = 0; s1 < 4; s1++) {
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += clover[s0 * 36 + s1 * 9 + c0 * 3 + c1] * dest[s1 * 3 + c1];
          }
        }
        tmp_dest[s0 * 3 + c0] = tmp0;
      }
    }
    give_ptr(origin_dest, tmp_dest, 12);
  }
}
