#pragma optimize(5)
#include "../../include/qcu_cuda.h"

#ifdef JOD457

__global__ void mpi_make_clover_part1(
    void *device_U, void *device_clover, void *device_b_mu_vec,
    void *device_b_nu_vec, int device_lat_x, const int device_lat_y,
    const int device_lat_z, const int device_lat_t, const int device_parity) {
  register int parity = blockIdx.x * blockDim.x + threadIdx.x;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  const int lat_xcc = lat_x * 9;
  const int lat_yxcc = lat_y * lat_xcc;
  const int lat_zyxcc = lat_z * lat_yxcc;
  const int lat_tzyxcc = lat_t * lat_zyxcc;
  const int lat_xsc = lat_x * 12;
  const int lat_yxsc = lat_y * lat_xsc;
  const int lat_zyxsc = lat_z * lat_yxsc;
  register int move;
  move = lat_x * lat_y * lat_z;
  const int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  const int z = parity / move;
  parity -= z * move;
  const int y = parity / lat_x;
  const int x = parity - y * lat_x;
  parity = device_parity;
  const int eo = (y + z + t) & 0x01; // (y+z+t)%2
  register LatticeComplex I(0.0, 1.0);
  register LatticeComplex zero(0.0, 0.0);
  register LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * 9);
  register LatticeComplex *origin_clover =
      ((static_cast<LatticeComplex *>(device_clover)) + t * lat_zyxcc * 16 +
       z * lat_yxcc * 16 + y * lat_xcc * 16 + x * 144);
  register LatticeComplex *origin_b_mu_send_vec =
      ((static_cast<LatticeComplex *>(device_b_mu_send_vec)) +
       (t * lat_z * lat_y + z * lat_y + y) * 6);
  register LatticeComplex *origin_b_nu_send_vec =
      ((static_cast<LatticeComplex *>(device_b_nu_send_vec)) +
       (t * lat_z * lat_y + z * lat_y + y) * 6);
  register LatticeComplex *tmp_U;
  register LatticeComplex tmp0(0.0, 0.0);
  register LatticeComplex tmp1(0.0, 0.0);
  register LatticeComplex U[9];
  register LatticeComplex b_mu_send_vec[36];
  register LatticeComplex b_nu_send_vec[36];
  give_value(dest, zero, 12);
  {
    // x-1
    move_backward_x(move, x, lat_x, eo, parity);
    if (move != lat_x - 1) {
      tmp_U = (origin_U + move * 9 + (1 - parity) * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * 12);
      give_ptr(src, tmp_src, 12);
      {
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += (src[c1] + src[c1 + 9] * I) * U[c1 * 3 + c0].conj();
            tmp1 += (src[c1 + 3] + src[c1 + 6] * I) * U[c1 * 3 + c0].conj();
          }
          dest[c0] += tmp0;
          dest[c0 + 3] += tmp1;
          dest[c0 + 6] -= tmp1 * I;
          dest[c0 + 9] -= tmp0 * I;
        }
      }
    }
    if (x == 0 && move == 0) { // even-odd
      // send in x+1 way
      give_ptr(src, origin_src, 12);
      {
        // sigma src
        for (int c1 = 0; c1 < 3; c1++) {
          b_x_send_vec[c1] = src[c1] - src[c1 + 9] * I;
          b_x_send_vec[c1 + 3] = src[c1 + 3] - src[c1 + 6] * I;
        }
        give_ptr(origin_b_x_send_vec, b_x_send_vec, 6);
      }
    }
  }
  {
    // x+1
    move_forward_x(move, x, lat_x, eo, parity);
    if (move != 1 - lat_x) {
      tmp_U = (origin_U + parity * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * 12);
      give_ptr(src, tmp_src, 12);
      {
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += (src[c1] - src[c1 + 9] * I) * U[c0 * 3 + c1];
            tmp1 += (src[c1 + 3] - src[c1 + 6] * I) * U[c0 * 3 + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + 3] += tmp1;
          dest[c0 + 6] += tmp1 * I;
          dest[c0 + 9] += tmp0 * I;
        }
      }
    }
    if (x == lat_x - 1 && move == 0) { // even-odd
      // send in x-1 way
      tmp_U = (origin_U + (1 - parity) * lat_tzyxcc); // even-odd
      give_u(U, tmp_U);
      give_ptr(src, origin_src, 12);
      {
        // just tmp
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += (src[c1] + src[c1 + 9] * I) * U[c1 * 3 + c0].conj();
            tmp1 += (src[c1 + 3] + src[c1 + 6] * I) * U[c1 * 3 + c0].conj();
          }
          f_x_send_vec[c0] = tmp0;
          f_x_send_vec[c0 + 3] = tmp1;
        }
        give_ptr(origin_f_x_send_vec, f_x_send_vec, 6);
      }
    }
  }
  // just add
  add_ptr(origin_dest, dest, 12);
  // XY
  give_value(U, zero, 9);
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

__global__ void mpi_make_clover_part2(void *device_U, void *device_clover,
                                      int device_lat_x, const int device_lat_y,
                                      const int device_lat_z,
                                      const int device_lat_t,
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
  const int eo = (y + z + t) & 0x01; // (y+z+t)%2
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

#endif