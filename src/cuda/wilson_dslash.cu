#pragma optimize(5)
#include "../../include/qcu.h"
#ifdef WILSON_DSLASH
__global__ void wilson_dslash(void *device_U, void *device_src,
                              void *device_dest, int device_lat_x,
                              const int device_lat_y, const int device_lat_z,
                              const int device_lat_t, const int device_parity) {
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
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
  int move;
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
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * 9);
  LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  LatticeComplex *tmp_U;
  LatticeComplex *tmp_src;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[9];
  LatticeComplex src[12];
  LatticeComplex dest[12];
  // just wilson(Sum part)
  host_give_value(dest, zero, 12);
  {
    // x-1
    move_backward_x(move, x, lat_x, eo, parity);
    tmp_U = (origin_U + move * 9 + (1 - parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * 12);
    give_ptr(src, tmp_src, 12);
  }
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
  {
    // x+1
    move_forward_x(move, x, lat_x, eo, parity);
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * 12);
    give_ptr(src, tmp_src, 12);
  }
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
  {
    // y-1
    move_backward(move, y, lat_y);
    tmp_U = (origin_U + move * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_xsc);
    give_ptr(src, tmp_src, 12);
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] - src[c1 + 9]) * U[c1 * 3 + c0].conj();
        tmp1 += (src[c1 + 3] + src[c1 + 6]) * U[c1 * 3 + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] += tmp1;
      dest[c0 + 9] -= tmp0;
    }
  }
  {
    // y+1
    move_forward(move, y, lat_y);
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_xsc);
    give_ptr(src, tmp_src, 12);
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] + src[c1 + 9]) * U[c0 * 3 + c1];
        tmp1 += (src[c1 + 3] - src[c1 + 6]) * U[c0 * 3 + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] -= tmp1;
      dest[c0 + 9] += tmp0;
    }
  }
  {
    // z-1
    move_backward(move, z, lat_z);
    tmp_U = (origin_U + move * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_yxsc);
    give_ptr(src, tmp_src, 12);
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] + src[c1 + 6] * I) * U[c1 * 3 + c0].conj();
        tmp1 += (src[c1 + 3] - src[c1 + 9] * I) * U[c1 * 3 + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] -= tmp0 * I;
      dest[c0 + 9] += tmp1 * I;
    }
  }
  {
    // z+1
    move_forward(move, z, lat_z);
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_yxsc);
    give_ptr(src, tmp_src, 12);
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] - src[c1 + 6] * I) * U[c0 * 3 + c1];
        tmp1 += (src[c1 + 3] + src[c1 + 9] * I) * U[c0 * 3 + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] += tmp0 * I;
      dest[c0 + 9] -= tmp1 * I;
    }
  }
  {
    // t-1
    move_backward(move, t, lat_t);
    tmp_U = (origin_U + move * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_zyxsc);
    give_ptr(src, tmp_src, 12);
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] + src[c1 + 6]) * U[c1 * 3 + c0].conj();
        tmp1 += (src[c1 + 3] + src[c1 + 9]) * U[c1 * 3 + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] += tmp0;
      dest[c0 + 9] += tmp1;
    }
  }
  {
    // t+1
    move_forward(move, t, lat_t);
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_zyxsc);
    give_ptr(src, tmp_src, 12);
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] - src[c1 + 6]) * U[c0 * 3 + c1];
        tmp1 += (src[c1 + 3] - src[c1 + 9]) * U[c0 * 3 + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] -= tmp0;
      dest[c0 + 9] -= tmp1;
    }
  }
  give_ptr(origin_dest, dest, 12);
}

__global__ void wilson_dslash_clear_dest(void *device_dest, int device_lat_x,
                                         const int device_lat_y,
                                         const int device_lat_z) {
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  int move;
  move = lat_x * lat_y * lat_z;
  const int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  const int z = parity / move;
  parity -= z * move;
  const int y = parity / lat_x;
  const int x = parity - y * lat_x;
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) +
       t * lat_z * lat_y * lat_x * 12 + z * lat_y * lat_x * 12 +
       y * lat_x * 12 + x * 12);
<<<<<<< HEAD
  host_give_value(origin_dest, zero, 12);
=======
  give_value(origin_dest, zero, 12);
>>>>>>> 13cba993c1f80d9ba1169f680f01d04c9ed1f482
}

__global__ void
wilson_dslash_x_send(void *device_U, void *device_src, void *device_dest,
                     int device_lat_x, const int device_lat_y,
                     const int device_lat_z, const int device_lat_t,
                     const int device_parity, void *device_b_x_send_vec,
                     void *device_f_x_send_vec) {
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
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
  int move;
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
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * 9);
  LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  LatticeComplex *origin_b_x_send_vec =
      ((static_cast<LatticeComplex *>(device_b_x_send_vec)) +
       (t * lat_z * lat_y + z * lat_y + y) * 6);
  LatticeComplex *origin_f_x_send_vec =
      ((static_cast<LatticeComplex *>(device_f_x_send_vec)) +
       (t * lat_z * lat_y + z * lat_y + y) * 6);
  LatticeComplex *tmp_U;
  LatticeComplex *tmp_src;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[9];
  LatticeComplex src[12];
  LatticeComplex dest[12];
  LatticeComplex b_x_send_vec[6];
  LatticeComplex f_x_send_vec[6];
<<<<<<< HEAD
  host_give_value(dest, zero, 12);
=======
  give_value(dest, zero, 12);
>>>>>>> 13cba993c1f80d9ba1169f680f01d04c9ed1f482
  { // x-1
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
      { // sigma src
        for (int c1 = 0; c1 < 3; c1++) {
          b_x_send_vec[c1] = src[c1] - src[c1 + 9] * I;
          b_x_send_vec[c1 + 3] = src[c1 + 3] - src[c1 + 6] * I;
        }
        give_ptr(origin_b_x_send_vec, b_x_send_vec, 6);
      }
    }
  }
  { // x+1
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
      { // just tmp
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
  } // just add
  add_ptr(origin_dest, dest, 12);
}

__global__ void
wilson_dslash_x_recv(void *device_U, void *device_dest, int device_lat_x,
                     const int device_lat_y, const int device_lat_z,
                     const int device_lat_t, const int device_parity,
                     void *device_b_x_recv_vec, void *device_f_x_recv_vec) {
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  int move;
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
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *origin_U = ((static_cast<LatticeComplex *>(device_U)) +
                              t * lat_z * lat_y * lat_x * 9 +
                              z * lat_y * lat_x * 9 + y * lat_x * 9 + x * 9);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) +
       t * lat_z * lat_y * lat_x * 12 + z * lat_y * lat_x * 12 +
       y * lat_x * 12 + x * 12);
  LatticeComplex *origin_b_x_recv_vec =
      ((static_cast<LatticeComplex *>(device_b_x_recv_vec)) +
       (t * lat_z * lat_y + z * lat_y + y) * 6);
  LatticeComplex *origin_f_x_recv_vec =
      ((static_cast<LatticeComplex *>(device_f_x_recv_vec)) +
       (t * lat_z * lat_y + z * lat_y + y) * 6);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[9];
  LatticeComplex dest[12];
  LatticeComplex b_x_recv_vec[6];
  LatticeComplex f_x_recv_vec[6]; // needed
<<<<<<< HEAD
  host_give_value(dest, zero, 12);
=======
  give_value(dest, zero, 12);
>>>>>>> 13cba993c1f80d9ba1169f680f01d04c9ed1f482
  { // x-1
    move_backward_x(move, x, lat_x, eo, parity);
    if (move == lat_x - 1) { // recv in x-1 way
      give_ptr(b_x_recv_vec, origin_b_x_recv_vec, 6);
      for (int c0 = 0; c0 < 3; c0++) {
        dest[c0] += b_x_recv_vec[c0];
        dest[c0 + 3] += b_x_recv_vec[c0 + 3];
        dest[c0 + 6] -= b_x_recv_vec[c0 + 3] * I;
        dest[c0 + 9] -= b_x_recv_vec[c0] * I;
      }
    }
  }
  { // x+1
    move_forward_x(move, x, lat_x, eo, parity);
    if (move == 1 - lat_x) { // recv in x+1 way
      give_ptr(f_x_recv_vec, origin_f_x_recv_vec, 6);
      tmp_U = (origin_U + parity * lat_t * lat_z * lat_y * lat_x * 9);
      give_u(U, tmp_U);
      {
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += f_x_recv_vec[c1] * U[c0 * 3 + c1];
            tmp1 += f_x_recv_vec[c1 + 3] * U[c0 * 3 + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + 3] += tmp1;
          dest[c0 + 6] += tmp1 * I;
          dest[c0 + 9] += tmp0 * I;
        }
      }
    }
  } // just add
  add_ptr(origin_dest, dest, 12);
}
__global__ void
wilson_dslash_y_send(void *device_U, void *device_src, void *device_dest,
                     int device_lat_x, const int device_lat_y,
                     const int device_lat_z, const int device_lat_t,
                     const int device_parity, void *device_b_y_send_vec,
                     void *device_f_y_send_vec) {
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
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
  int move;
  move = lat_x * lat_y * lat_z;
  const int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  const int z = parity / move;
  parity -= z * move;
  const int y = parity / lat_x;
  const int x = parity - y * lat_x;
  parity = device_parity;
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * 9);
  LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  LatticeComplex *origin_b_y_send_vec =
      ((static_cast<LatticeComplex *>(device_b_y_send_vec)) +
       (t * lat_z * lat_x + z * lat_x + x) * 6);
  LatticeComplex *origin_f_y_send_vec =
      ((static_cast<LatticeComplex *>(device_f_y_send_vec)) +
       (t * lat_z * lat_x + z * lat_x + x) * 6);
  LatticeComplex *tmp_U;
  LatticeComplex *tmp_src;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[9];
  LatticeComplex src[12];
  LatticeComplex dest[12];
  LatticeComplex b_y_send_vec[6];
  LatticeComplex f_y_send_vec[6];
<<<<<<< HEAD
  host_give_value(dest, zero, 12);
=======
  give_value(dest, zero, 12);
>>>>>>> 13cba993c1f80d9ba1169f680f01d04c9ed1f482
  { // y-1
    move_backward(move, y, lat_y);
    if (move == -1) {
      tmp_U = (origin_U + move * lat_xcc + lat_tzyxcc * 2 +
               (1 - parity) * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * lat_xsc);
      give_ptr(src, tmp_src, 12);
      {
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += (src[c1] - src[c1 + 9]) * U[c1 * 3 + c0].conj();
            tmp1 += (src[c1 + 3] + src[c1 + 6]) * U[c1 * 3 + c0].conj();
          }
          dest[c0] += tmp0;
          dest[c0 + 3] += tmp1;
          dest[c0 + 6] += tmp1;
          dest[c0 + 9] -= tmp0;
        }
      }
    } else { // send in y+1 way
      give_ptr(src, origin_src, 12);
      { // sigma src
        for (int c1 = 0; c1 < 3; c1++) {
          b_y_send_vec[c1] = src[c1] + src[c1 + 9];
          b_y_send_vec[c1 + 3] = src[c1 + 3] - src[c1 + 6];
        }
        give_ptr(origin_b_y_send_vec, b_y_send_vec, 6);
      }
    }
  }
  { // y+1
    move_forward(move, y, lat_y);
    if (move == 1) {
      tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * lat_xsc);
      give_ptr(src, tmp_src, 12);
      {
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += (src[c1] + src[c1 + 9]) * U[c0 * 3 + c1];
            tmp1 += (src[c1 + 3] - src[c1 + 6]) * U[c0 * 3 + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + 3] += tmp1;
          dest[c0 + 6] -= tmp1;
          dest[c0 + 9] += tmp0;
        }
      }
    } else { // send in y-1 way
      tmp_U =
          (origin_U + +lat_tzyxcc * 2 + (1 - parity) * lat_tzyxcc); // even-odd
      give_u(U, tmp_U);
      give_ptr(src, origin_src, 12);
      { // just tmp
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += (src[c1] - src[c1 + 9]) * U[c1 * 3 + c0].conj();
            tmp1 += (src[c1 + 3] + src[c1 + 6]) * U[c1 * 3 + c0].conj();
          }
          f_y_send_vec[c0] = tmp0;
          f_y_send_vec[c0 + 3] = tmp1;
        }
        give_ptr(origin_f_y_send_vec, f_y_send_vec, 6);
      }
    }
  } // just add
  add_ptr(origin_dest, dest, 12);
}

__global__ void
wilson_dslash_y_recv(void *device_U, void *device_dest, int device_lat_x,
                     const int device_lat_y, const int device_lat_z,
                     const int device_lat_t, const int device_parity,
                     void *device_b_y_recv_vec, void *device_f_y_recv_vec) {
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  int move;
  move = lat_x * lat_y * lat_z;
  const int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  const int z = parity / move;
  parity -= z * move;
  const int y = parity / lat_x;
  const int x = parity - y * lat_x;
  parity = device_parity;
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *origin_U = ((static_cast<LatticeComplex *>(device_U)) +
                              t * lat_z * lat_y * lat_x * 9 +
                              z * lat_y * lat_x * 9 + y * lat_x * 9 + x * 9);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) +
       t * lat_z * lat_y * lat_x * 12 + z * lat_y * lat_x * 12 +
       y * lat_x * 12 + x * 12);
  LatticeComplex *origin_b_y_recv_vec =
      ((static_cast<LatticeComplex *>(device_b_y_recv_vec)) +
       (t * lat_z * lat_x + z * lat_x + x) * 6);
  LatticeComplex *origin_f_y_recv_vec =
      ((static_cast<LatticeComplex *>(device_f_y_recv_vec)) +
       (t * lat_z * lat_x + z * lat_x + x) * 6);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[9];
  LatticeComplex dest[12];
  LatticeComplex b_y_recv_vec[6];
  LatticeComplex f_y_recv_vec[6]; // needed
<<<<<<< HEAD
  host_give_value(dest, zero, 12);
=======
  give_value(dest, zero, 12);
>>>>>>> 13cba993c1f80d9ba1169f680f01d04c9ed1f482
  { // y-1
    move_backward(move, y, lat_y);
    if (move != -1) { // recv in y-1 way
      give_ptr(b_y_recv_vec, origin_b_y_recv_vec, 6);
      for (int c0 = 0; c0 < 3; c0++) {
        dest[c0] += b_y_recv_vec[c0];
        dest[c0 + 3] += b_y_recv_vec[c0 + 3];
        dest[c0 + 6] += b_y_recv_vec[c0 + 3];
        dest[c0 + 9] -= b_y_recv_vec[c0];
      }
    }
  }
  { // y+1
    move_forward(move, y, lat_y);
    if (move != 1) { // recv in y+1 way
      give_ptr(f_y_recv_vec, origin_f_y_recv_vec, 6);
      tmp_U = (origin_U + (parity + 2) * lat_t * lat_z * lat_y * lat_x * 9);
      give_u(U, tmp_U);
      {
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += f_y_recv_vec[c1] * U[c0 * 3 + c1];
            tmp1 += f_y_recv_vec[c1 + 3] * U[c0 * 3 + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + 3] += tmp1;
          dest[c0 + 6] -= tmp1;
          dest[c0 + 9] += tmp0;
        }
      }
    }
  } // just add
  add_ptr(origin_dest, dest, 12);
}

__global__ void
wilson_dslash_z_send(void *device_U, void *device_src, void *device_dest,
                     int device_lat_x, const int device_lat_y,
                     const int device_lat_z, const int device_lat_t,
                     const int device_parity, void *device_b_z_send_vec,
                     void *device_f_z_send_vec) {
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
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
  int move;
  move = lat_x * lat_y * lat_z;
  const int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  const int z = parity / move;
  parity -= z * move;
  const int y = parity / lat_x;
  const int x = parity - y * lat_x;
  parity = device_parity;
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * 9);
  LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  LatticeComplex *origin_b_z_send_vec =
      ((static_cast<LatticeComplex *>(device_b_z_send_vec)) +
       (t * lat_y * lat_x + y * lat_x + x) * 6);
  LatticeComplex *origin_f_z_send_vec =
      ((static_cast<LatticeComplex *>(device_f_z_send_vec)) +
       (t * lat_y * lat_x + y * lat_x + x) * 6);
  LatticeComplex *tmp_U;
  LatticeComplex *tmp_src;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[9];
  LatticeComplex src[12];
  LatticeComplex dest[12];
  LatticeComplex b_z_send_vec[6];
  LatticeComplex f_z_send_vec[6];
<<<<<<< HEAD
  host_give_value(dest, zero, 12);
=======
  give_value(dest, zero, 12);
>>>>>>> 13cba993c1f80d9ba1169f680f01d04c9ed1f482
  { // z-1
    move_backward(move, z, lat_z);
    if (move == -1) {
      tmp_U = (origin_U + move * lat_yxcc + lat_tzyxcc * 4 +
               (1 - parity) * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * lat_yxsc);
      give_ptr(src, tmp_src, 12);
      {
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += (src[c1] + src[c1 + 6] * I) * U[c1 * 3 + c0].conj();
            tmp1 += (src[c1 + 3] - src[c1 + 9] * I) * U[c1 * 3 + c0].conj();
          }
          dest[c0] += tmp0;
          dest[c0 + 3] += tmp1;
          dest[c0 + 6] -= tmp0 * I;
          dest[c0 + 9] += tmp1 * I;
        }
      }
    } else { // send in z+1 way
      give_ptr(src, origin_src, 12);
      { // sigma src
        for (int c1 = 0; c1 < 3; c1++) {
          b_z_send_vec[c1] = src[c1] - src[c1 + 6] * I;
          b_z_send_vec[c1 + 3] = src[c1 + 3] + src[c1 + 9] * I;
        }
        give_ptr(origin_b_z_send_vec, b_z_send_vec, 6);
      }
    }
  }
  { // z+1
    move_forward(move, z, lat_z);
    if (move == 1) {
      tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * lat_yxsc);
      give_ptr(src, tmp_src, 12);
      {
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += (src[c1] - src[c1 + 6] * I) * U[c0 * 3 + c1];
            tmp1 += (src[c1 + 3] + src[c1 + 9] * I) * U[c0 * 3 + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + 3] += tmp1;
          dest[c0 + 6] += tmp0 * I;
          dest[c0 + 9] -= tmp1 * I;
        }
      }
    } else { // send in z-1 way
      tmp_U =
          (origin_U + 4 * lat_tzyxcc + (1 - parity) * lat_tzyxcc); // even-odd
      give_u(U, tmp_U);
      give_ptr(src, origin_src, 12);
      { // just tmp
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += (src[c1] + src[c1 + 6] * I) * U[c1 * 3 + c0].conj();
            tmp1 += (src[c1 + 3] - src[c1 + 9] * I) * U[c1 * 3 + c0].conj();
          }
          f_z_send_vec[c0] = tmp0;
          f_z_send_vec[c0 + 3] = tmp1;
        }
        give_ptr(origin_f_z_send_vec, f_z_send_vec, 6);
      }
    }
  } // just add
  add_ptr(origin_dest, dest, 12);
}

__global__ void
wilson_dslash_z_recv(void *device_U, void *device_dest, int device_lat_x,
                     const int device_lat_y, const int device_lat_z,
                     const int device_lat_t, const int device_parity,
                     void *device_b_z_recv_vec, void *device_f_z_recv_vec) {
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  int move;
  move = lat_x * lat_y * lat_z;
  const int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  const int z = parity / move;
  parity -= z * move;
  const int y = parity / lat_x;
  const int x = parity - y * lat_x;
  parity = device_parity;
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *origin_U = ((static_cast<LatticeComplex *>(device_U)) +
                              t * lat_z * lat_y * lat_x * 9 +
                              z * lat_y * lat_x * 9 + y * lat_x * 9 + x * 9);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) +
       t * lat_z * lat_y * lat_x * 12 + z * lat_y * lat_x * 12 +
       y * lat_x * 12 + x * 12);
  LatticeComplex *origin_b_z_recv_vec =
      ((static_cast<LatticeComplex *>(device_b_z_recv_vec)) +
       (t * lat_y * lat_x + y * lat_x + x) * 6);
  LatticeComplex *origin_f_z_recv_vec =
      ((static_cast<LatticeComplex *>(device_f_z_recv_vec)) +
       (t * lat_y * lat_x + y * lat_x + x) * 6);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[9];
  LatticeComplex dest[12];
  LatticeComplex b_z_recv_vec[6];
  LatticeComplex f_z_recv_vec[6]; // needed
<<<<<<< HEAD
  host_give_value(dest, zero, 12);
=======
  give_value(dest, zero, 12);
>>>>>>> 13cba993c1f80d9ba1169f680f01d04c9ed1f482
  { // z-1
    move_backward(move, z, lat_z);
    if (move != -1) { // recv in z-1 way
      give_ptr(b_z_recv_vec, origin_b_z_recv_vec, 6);
      for (int c0 = 0; c0 < 3; c0++) {
        dest[c0] += b_z_recv_vec[c0];
        dest[c0 + 3] += b_z_recv_vec[c0 + 3];
        dest[c0 + 6] -= b_z_recv_vec[c0] * I;
        dest[c0 + 9] += b_z_recv_vec[c0 + 3] * I;
      }
    }
  }
  { // z+1
    move_forward(move, z, lat_z);
    if (move != 1) { // recv in z+1 way
      give_ptr(f_z_recv_vec, origin_f_z_recv_vec, 6);
      tmp_U = (origin_U + (parity + 4) * lat_t * lat_z * lat_y * lat_x * 9);
      give_u(U, tmp_U);
      {
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += f_z_recv_vec[c1] * U[c0 * 3 + c1];
            tmp1 += f_z_recv_vec[c1 + 3] * U[c0 * 3 + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + 3] += tmp1;
          dest[c0 + 6] += tmp0 * I;
          dest[c0 + 9] -= tmp1 * I;
        }
      }
    }
  } // just add
  add_ptr(origin_dest, dest, 12);
}

__global__ void
wilson_dslash_t_send(void *device_U, void *device_src, void *device_dest,
                     int device_lat_x, const int device_lat_y,
                     const int device_lat_z, const int device_lat_t,
                     const int device_parity, void *device_b_t_send_vec,
                     void *device_f_t_send_vec) {
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
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
  int move;
  move = lat_x * lat_y * lat_z;
  const int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  const int z = parity / move;
  parity -= z * move;
  const int y = parity / lat_x;
  const int x = parity - y * lat_x;
  parity = device_parity;
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * 9);
  LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  LatticeComplex *origin_b_t_send_vec =
      ((static_cast<LatticeComplex *>(device_b_t_send_vec)) +
       (z * lat_y * lat_x + y * lat_x + x) * 6);
  LatticeComplex *origin_f_t_send_vec =
      ((static_cast<LatticeComplex *>(device_f_t_send_vec)) +
       (z * lat_y * lat_x + y * lat_x + x) * 6);
  LatticeComplex *tmp_U;
  LatticeComplex *tmp_src;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[9];
  LatticeComplex src[12];
  LatticeComplex dest[12];
  LatticeComplex b_t_send_vec[6];
  LatticeComplex f_t_send_vec[6];
<<<<<<< HEAD
  host_give_value(dest, zero, 12);
=======
  give_value(dest, zero, 12);
>>>>>>> 13cba993c1f80d9ba1169f680f01d04c9ed1f482
  { // t-1
    move_backward(move, t, lat_t);
    if (move == -1) {
      tmp_U = (origin_U + move * lat_zyxcc + lat_tzyxcc * 6 +
               (1 - parity) * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * lat_zyxsc);
      give_ptr(src, tmp_src, 12);
      {
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += (src[c1] + src[c1 + 6]) * U[c1 * 3 + c0].conj();
            tmp1 += (src[c1 + 3] + src[c1 + 9]) * U[c1 * 3 + c0].conj();
          }
          dest[c0] += tmp0;
          dest[c0 + 3] += tmp1;
          dest[c0 + 6] += tmp0;
          dest[c0 + 9] += tmp1;
        }
      }
    } else { // send in t+1 way
      give_ptr(src, origin_src, 12);
      { // sigma src
        for (int c1 = 0; c1 < 3; c1++) {
          b_t_send_vec[c1] = src[c1] - src[c1 + 6];
          b_t_send_vec[c1 + 3] = src[c1 + 3] - src[c1 + 9];
        }
        give_ptr(origin_b_t_send_vec, b_t_send_vec, 6);
      }
    }
  }
  { // t+1
    move_forward(move, t, lat_t);
    if (move == 1) {
      tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * lat_zyxsc);
      give_ptr(src, tmp_src, 12);
      {
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += (src[c1] - src[c1 + 6]) * U[c0 * 3 + c1];
            tmp1 += (src[c1 + 3] - src[c1 + 9]) * U[c0 * 3 + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + 3] += tmp1;
          dest[c0 + 6] -= tmp0;
          dest[c0 + 9] -= tmp1;
        }
      }
    } else { // send in t-1 way
      tmp_U =
          (origin_U + lat_tzyxcc * 6 + (1 - parity) * lat_tzyxcc); // even-odd
      give_u(U, tmp_U);
      give_ptr(src, origin_src, 12);
      { // just tmp
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += (src[c1] + src[c1 + 6]) * U[c1 * 3 + c0].conj();
            tmp1 += (src[c1 + 3] + src[c1 + 9]) * U[c1 * 3 + c0].conj();
          }
          f_t_send_vec[c0] = tmp0;
          f_t_send_vec[c0 + 3] = tmp1;
        }
        give_ptr(origin_f_t_send_vec, f_t_send_vec, 6);
      }
    }
  } // just add
  add_ptr(origin_dest, dest, 12);
}

__global__ void
wilson_dslash_t_recv(void *device_U, void *device_dest, int device_lat_x,
                     const int device_lat_y, const int device_lat_z,
                     const int device_lat_t, const int device_parity,
                     void *device_b_t_recv_vec, void *device_f_t_recv_vec) {
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  int move;
  move = lat_x * lat_y * lat_z;
  const int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  const int z = parity / move;
  parity -= z * move;
  const int y = parity / lat_x;
  const int x = parity - y * lat_x;
  parity = device_parity;
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *origin_U = ((static_cast<LatticeComplex *>(device_U)) +
                              t * lat_z * lat_y * lat_x * 9 +
                              z * lat_y * lat_x * 9 + y * lat_x * 9 + x * 9);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) +
       t * lat_z * lat_y * lat_x * 12 + z * lat_y * lat_x * 12 +
       y * lat_x * 12 + x * 12);
  LatticeComplex *origin_b_t_recv_vec =
      ((static_cast<LatticeComplex *>(device_b_t_recv_vec)) +
       (z * lat_y * lat_x + y * lat_x + x) * 6);
  LatticeComplex *origin_f_t_recv_vec =
      ((static_cast<LatticeComplex *>(device_f_t_recv_vec)) +
       (z * lat_y * lat_x + y * lat_x + x) * 6);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[9];
  LatticeComplex dest[12];
  LatticeComplex b_t_recv_vec[6];
  LatticeComplex f_t_recv_vec[6]; // needed
<<<<<<< HEAD
  host_give_value(dest, zero, 12);
=======
  give_value(dest, zero, 12);
>>>>>>> 13cba993c1f80d9ba1169f680f01d04c9ed1f482
  { // t-1
    move_backward(move, t, lat_t);
    if (move != -1) { // recv in t-1 way
      give_ptr(b_t_recv_vec, origin_b_t_recv_vec, 6);
      for (int c0 = 0; c0 < 3; c0++) {
        dest[c0] += b_t_recv_vec[c0];
        dest[c0 + 3] += b_t_recv_vec[c0 + 3];
        dest[c0 + 6] += b_t_recv_vec[c0];
        dest[c0 + 9] += b_t_recv_vec[c0 + 3];
      }
    }
  }
  { // t+1
    move_forward(move, t, lat_t);
    if (move != 1) { // recv in t+1 way
      give_ptr(f_t_recv_vec, origin_f_t_recv_vec, 6);
      tmp_U = (origin_U + (parity + 6) * lat_t * lat_z * lat_y * lat_x * 9);
      give_u(U, tmp_U);
      {
        for (int c0 = 0; c0 < 3; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < 3; c1++) {
            tmp0 += f_t_recv_vec[c1] * U[c0 * 3 + c1];
            tmp1 += f_t_recv_vec[c1 + 3] * U[c0 * 3 + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + 3] += tmp1;
          dest[c0 + 6] -= tmp0;
          dest[c0 + 9] -= tmp1;
        }
      }
    }
  } // just add
  add_ptr(origin_dest, dest, 12);
}
#endif