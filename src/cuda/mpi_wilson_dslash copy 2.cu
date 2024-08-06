#pragma optimize(5)
#include "../../include/qcu.h"

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
  give_value(origin_dest, zero, 12);
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
  LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) +
       t * lat_z * lat_y * lat_x * 9 + z * lat_y * lat_x * 9 + y * lat_x * 9 +
       x * 9);
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
  LatticeComplex f_x_recv_vec[6];
  // needed
  give_value(dest, zero, 12);
  {
    // x-1
    move_backward_x(move, x, lat_x, eo, parity);
    if (move == lat_x - 1) {
      // recv in x-1 way
      give_ptr(b_x_recv_vec, origin_b_x_recv_vec, 6);
      for (int c0 = 0; c0 < 3; c0++) {
        dest[c0] += b_x_recv_vec[c0];
        dest[c0 + 3] += b_x_recv_vec[c0 + 3];
        dest[c0 + 6] -= b_x_recv_vec[c0 + 3] * I;
        dest[c0 + 9] -= b_x_recv_vec[c0] * I;
      }
    }
  }
  {
    // x+1
    move_forward_x(move, x, lat_x, eo, parity);
    if (move == 1 - lat_x) {
      // recv in x+1 way
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
  }
  // just add
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
  give_value(dest, zero, 12);
  {
    // y-1
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
    } else {
      // send in y+1 way
      give_ptr(src, origin_src, 12);
      {
        // sigma src
        for (int c1 = 0; c1 < 3; c1++) {
          b_y_send_vec[c1] = src[c1] + src[c1 + 9];
          b_y_send_vec[c1 + 3] = src[c1 + 3] - src[c1 + 6];
        }
        give_ptr(origin_b_y_send_vec, b_y_send_vec, 6);
      }
    }
  }
  {
    // y+1
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
    } else {
      // send in y-1 way
      tmp_U =
          (origin_U + +lat_tzyxcc * 2 + (1 - parity) * lat_tzyxcc); // even-odd
      give_u(U, tmp_U);
      give_ptr(src, origin_src, 12);
      {
        // just tmp
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
  }
  // just add
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
  LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) +
       t * lat_z * lat_y * lat_x * 9 + z * lat_y * lat_x * 9 + y * lat_x * 9 +
       x * 9);
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
  LatticeComplex f_y_recv_vec[6];
  // needed
  give_value(dest, zero, 12);
  {
    // y-1
    move_backward(move, y, lat_y);
    if (move != -1) {
      // recv in y-1 way
      give_ptr(b_y_recv_vec, origin_b_y_recv_vec, 6);
      for (int c0 = 0; c0 < 3; c0++) {
        dest[c0] += b_y_recv_vec[c0];
        dest[c0 + 3] += b_y_recv_vec[c0 + 3];
        dest[c0 + 6] += b_y_recv_vec[c0 + 3];
        dest[c0 + 9] -= b_y_recv_vec[c0];
      }
    }
  }
  {
    // y+1
    move_forward(move, y, lat_y);
    if (move != 1) {
      // recv in y+1 way
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
  }
  // just add
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
  give_value(dest, zero, 12);
  {
    // z-1
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
    } else {
      // send in z+1 way
      give_ptr(src, origin_src, 12);
      {
        // sigma src
        for (int c1 = 0; c1 < 3; c1++) {
          b_z_send_vec[c1] = src[c1] - src[c1 + 6] * I;
          b_z_send_vec[c1 + 3] = src[c1 + 3] + src[c1 + 9] * I;
        }
        give_ptr(origin_b_z_send_vec, b_z_send_vec, 6);
      }
    }
  }
  {
    // z+1
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
    } else {
      // send in z-1 way
      tmp_U =
          (origin_U + 4 * lat_tzyxcc + (1 - parity) * lat_tzyxcc); // even-odd
      give_u(U, tmp_U);
      give_ptr(src, origin_src, 12);
      {
        // just tmp
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
  }
  // just add
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
  LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) +
       t * lat_z * lat_y * lat_x * 9 + z * lat_y * lat_x * 9 + y * lat_x * 9 +
       x * 9);
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
  LatticeComplex f_z_recv_vec[6];
  // needed
  give_value(dest, zero, 12);
  {
    // z-1
    move_backward(move, z, lat_z);
    if (move != -1) {
      // recv in z-1 way
      give_ptr(b_z_recv_vec, origin_b_z_recv_vec, 6);
      for (int c0 = 0; c0 < 3; c0++) {
        dest[c0] += b_z_recv_vec[c0];
        dest[c0 + 3] += b_z_recv_vec[c0 + 3];
        dest[c0 + 6] -= b_z_recv_vec[c0] * I;
        dest[c0 + 9] += b_z_recv_vec[c0 + 3] * I;
      }
    }
  }
  {
    // z+1
    move_forward(move, z, lat_z);
    if (move != 1) {
      // recv in z+1 way
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
  }
  // just add
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
  give_value(dest, zero, 12);
  {
    // t-1
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
    } else {
      // send in t+1 way
      give_ptr(src, origin_src, 12);
      {
        // sigma src
        for (int c1 = 0; c1 < 3; c1++) {
          b_t_send_vec[c1] = src[c1] - src[c1 + 6];
          b_t_send_vec[c1 + 3] = src[c1 + 3] - src[c1 + 9];
        }
        give_ptr(origin_b_t_send_vec, b_t_send_vec, 6);
      }
    }
  }
  {
    // t+1
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
    } else {
      // send in t-1 way
      tmp_U =
          (origin_U + lat_tzyxcc * 6 + (1 - parity) * lat_tzyxcc); // even-odd
      give_u(U, tmp_U);
      give_ptr(src, origin_src, 12);
      {
        // just tmp
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
  }
  // just add
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
  LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) +
       t * lat_z * lat_y * lat_x * 9 + z * lat_y * lat_x * 9 + y * lat_x * 9 +
       x * 9);
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
  LatticeComplex f_t_recv_vec[6];
  // needed
  give_value(dest, zero, 12);
  {
    // t-1
    move_backward(move, t, lat_t);
    if (move != -1) {
      // recv in t-1 way
      give_ptr(b_t_recv_vec, origin_b_t_recv_vec, 6);
      for (int c0 = 0; c0 < 3; c0++) {
        dest[c0] += b_t_recv_vec[c0];
        dest[c0 + 3] += b_t_recv_vec[c0 + 3];
        dest[c0 + 6] += b_t_recv_vec[c0];
        dest[c0 + 9] += b_t_recv_vec[c0 + 3];
      }
    }
  }
  {
    // t+1
    move_forward(move, t, lat_t);
    if (move != 1) {
      // recv in t+1 way
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
  }
  // just add
  add_ptr(origin_dest, dest, 12);
}

#ifdef MPI_WILSON_DSLASH
void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                  QcuParam *param, int parity, QcuParam *grid) {
  const int lat_x = param->lattice_size[0] >> 1;
  const int lat_y = param->lattice_size[1];
  const int lat_z = param->lattice_size[2];
  const int lat_t = param->lattice_size[3];
  const int lat_yzt6 = lat_y * lat_z * lat_t * 6;
  const int lat_xzt6 = lat_x * lat_z * lat_t * 6;
  const int lat_xyt6 = lat_x * lat_y * lat_t * 6;
  const int lat_xyz6 = lat_x * lat_y * lat_z * 6;
  const int lat_yzt12 = lat_yzt6 * 2;
  const int lat_xzt12 = lat_xzt6 * 2;
  const int lat_xyt12 = lat_xyt6 * 2;
  const int lat_xyz12 = lat_xyz6 * 2;
  cudaError_t err;
  dim3 gridDim(lat_x * lat_y * lat_z * lat_t / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  {
    // mpi wilson dslash
    int node_size, node_rank, move_b, move_f;
    MPI_Comm_size(MPI_COMM_WORLD, &node_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_rank);
    const int grid_x = grid->lattice_size[0];
    const int grid_y = grid->lattice_size[1];
    const int grid_z = grid->lattice_size[2];
    const int grid_t = grid->lattice_size[3];
    const int grid_index_x = node_rank / grid_t / grid_z / grid_y;
    const int grid_index_y = node_rank / grid_t / grid_z % grid_y;
    const int grid_index_z = node_rank / grid_t % grid_z;
    const int grid_index_t = node_rank % grid_t;
    MPI_Request b_x_send_request, b_x_recv_request;
    MPI_Request f_x_send_request, f_x_recv_request;
    MPI_Request b_y_send_request, b_y_recv_request;
    MPI_Request f_y_send_request, f_y_recv_request;
    MPI_Request b_z_send_request, b_z_recv_request;
    MPI_Request f_z_send_request, f_z_recv_request;
    MPI_Request b_t_send_request, b_t_recv_request;
    MPI_Request f_t_send_request, f_t_recv_request;
    void *b_x_send_vec, *b_x_recv_vec;
    void *f_x_send_vec, *f_x_recv_vec;
    void *b_y_send_vec, *b_y_recv_vec;
    void *f_y_send_vec, *f_y_recv_vec;
    void *b_z_send_vec, *b_z_recv_vec;
    void *f_z_send_vec, *f_z_recv_vec;
    void *b_t_send_vec, *b_t_recv_vec;
    void *f_t_send_vec, *f_t_recv_vec;
    cudaMallocManaged(&b_x_send_vec, lat_yzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_x_send_vec, lat_yzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_y_send_vec, lat_xzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_y_send_vec, lat_xzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_z_send_vec, lat_xyt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_z_send_vec, lat_xyt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_t_send_vec, lat_xyz6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_t_send_vec, lat_xyz6 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_x_recv_vec, lat_yzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_x_recv_vec, lat_yzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_y_recv_vec, lat_xzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_y_recv_vec, lat_xzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_z_recv_vec, lat_xyt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_z_recv_vec, lat_xyt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_t_recv_vec, lat_xyz6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_t_recv_vec, lat_xyz6 * sizeof(LatticeComplex));
    auto start = std::chrono::high_resolution_clock::now();
    // clean
    wilson_dslash_clear_dest<<<gridDim, blockDim>>>(fermion_out, lat_x, lat_y,
                                                    lat_z);
    // send x
    wilson_dslash_x_send<<<gridDim, blockDim>>>(
        gauge, fermion_in, fermion_out, lat_x, lat_y, lat_z, lat_t, parity,
        b_x_send_vec, f_x_send_vec);
    if (grid_x != 1) {
      checkCudaErrors(cudaDeviceSynchronize());
      move_backward(move_b, grid_index_x, grid_x);
      move_forward(move_f, grid_index_x, grid_x);
      move_b = node_rank + move_b * grid_y * grid_z * grid_t;
      move_f = node_rank + move_f * grid_y * grid_z * grid_t;
      MPI_Irecv(b_x_recv_vec, lat_yzt12, MPI_DOUBLE, move_b, 1, MPI_COMM_WORLD,
                &b_x_recv_request);
      MPI_Irecv(f_x_recv_vec, lat_yzt12, MPI_DOUBLE, move_f, 0, MPI_COMM_WORLD,
                &f_x_recv_request);
      MPI_Isend(b_x_send_vec, lat_yzt12, MPI_DOUBLE, move_b, 0, MPI_COMM_WORLD,
                &b_x_send_request);
      MPI_Isend(f_x_send_vec, lat_yzt12, MPI_DOUBLE, move_f, 1, MPI_COMM_WORLD,
                &f_x_send_request);
    }
    // send y
    wilson_dslash_y_send<<<gridDim, blockDim>>>(
        gauge, fermion_in, fermion_out, lat_x, lat_y, lat_z, lat_t, parity,
        b_y_send_vec, f_y_send_vec);
    if (grid_y != 1) {
      checkCudaErrors(cudaDeviceSynchronize());
      move_backward(move_b, grid_index_y, grid_y);
      move_forward(move_f, grid_index_y, grid_y);
      move_b = node_rank + move_b * grid_z * grid_t;
      move_f = node_rank + move_f * grid_z * grid_t;
      MPI_Irecv(b_y_recv_vec, lat_xzt12, MPI_DOUBLE, move_b, 3, MPI_COMM_WORLD,
                &b_y_recv_request);
      MPI_Irecv(f_y_recv_vec, lat_xzt12, MPI_DOUBLE, move_f, 2, MPI_COMM_WORLD,
                &f_y_recv_request);
      MPI_Isend(b_y_send_vec, lat_xzt12, MPI_DOUBLE, move_b, 2, MPI_COMM_WORLD,
                &b_y_send_request);
      MPI_Isend(f_y_send_vec, lat_xzt12, MPI_DOUBLE, move_f, 3, MPI_COMM_WORLD,
                &f_y_send_request);
    }
    // send z
    wilson_dslash_z_send<<<gridDim, blockDim>>>(
        gauge, fermion_in, fermion_out, lat_x, lat_y, lat_z, lat_t, parity,
        b_z_send_vec, f_z_send_vec);
    if (grid_z != 1) {
      checkCudaErrors(cudaDeviceSynchronize());
      move_backward(move_b, grid_index_z, grid_z);
      move_forward(move_f, grid_index_z, grid_z);
      move_b = node_rank + move_b * grid_t;
      move_f = node_rank + move_f * grid_t;
      MPI_Irecv(b_z_recv_vec, lat_xyt12, MPI_DOUBLE, move_b, 5, MPI_COMM_WORLD,
                &b_z_recv_request);
      MPI_Irecv(f_z_recv_vec, lat_xyt12, MPI_DOUBLE, move_f, 4, MPI_COMM_WORLD,
                &f_z_recv_request);
      MPI_Isend(b_z_send_vec, lat_xyt12, MPI_DOUBLE, move_b, 4, MPI_COMM_WORLD,
                &b_z_send_request);
      MPI_Isend(f_z_send_vec, lat_xyt12, MPI_DOUBLE, move_f, 5, MPI_COMM_WORLD,
                &f_z_send_request);
    }
    // send t
    wilson_dslash_t_send<<<gridDim, blockDim>>>(
        gauge, fermion_in, fermion_out, lat_x, lat_y, lat_z, lat_t, parity,
        b_t_send_vec, f_t_send_vec);
    if (grid_t != 1) {
      checkCudaErrors(cudaDeviceSynchronize());
      move_backward(move_b, grid_index_t, grid_t);
      move_forward(move_f, grid_index_t, grid_t);
      move_b = node_rank + move_b;
      move_f = node_rank + move_f;
      MPI_Irecv(b_t_recv_vec, lat_xyz12, MPI_DOUBLE, move_b, 7, MPI_COMM_WORLD,
                &b_t_recv_request);
      MPI_Irecv(f_t_recv_vec, lat_xyz12, MPI_DOUBLE, move_f, 6, MPI_COMM_WORLD,
                &f_t_recv_request);
      MPI_Isend(b_t_send_vec, lat_xyz12, MPI_DOUBLE, move_b, 6, MPI_COMM_WORLD,
                &b_t_send_request);
      MPI_Isend(f_t_send_vec, lat_xyz12, MPI_DOUBLE, move_f, 7, MPI_COMM_WORLD,
                &f_t_send_request);
    }
    // recv x
    if (grid_x != 1) {
      MPI_Wait(&b_x_recv_request, MPI_STATUS_IGNORE);
      MPI_Wait(&f_x_recv_request, MPI_STATUS_IGNORE);
      wilson_dslash_x_recv<<<gridDim, blockDim>>>(gauge, fermion_out, lat_x,
                                                  lat_y, lat_z, lat_t, parity,
                                                  b_x_recv_vec, f_x_recv_vec);
    } else {
      checkCudaErrors(cudaDeviceSynchronize());
      wilson_dslash_x_recv<<<gridDim, blockDim>>>(gauge, fermion_out, lat_x,
                                                  lat_y, lat_z, lat_t, parity,
                                                  f_x_send_vec, b_x_send_vec);
    }
    // recv y
    if (grid_y != 1) {
      MPI_Wait(&b_y_recv_request, MPI_STATUS_IGNORE);
      MPI_Wait(&f_y_recv_request, MPI_STATUS_IGNORE);
      wilson_dslash_y_recv<<<gridDim, blockDim>>>(gauge, fermion_out, lat_x,
                                                  lat_y, lat_z, lat_t, parity,
                                                  b_y_recv_vec, f_y_recv_vec);
    } else {
      checkCudaErrors(cudaDeviceSynchronize());
      wilson_dslash_y_recv<<<gridDim, blockDim>>>(gauge, fermion_out, lat_x,
                                                  lat_y, lat_z, lat_t, parity,
                                                  f_y_send_vec, b_y_send_vec);
    }
    // recv z
    if (grid_z != 1) {
      MPI_Wait(&b_z_recv_request, MPI_STATUS_IGNORE);
      MPI_Wait(&f_z_recv_request, MPI_STATUS_IGNORE);
      wilson_dslash_z_recv<<<gridDim, blockDim>>>(gauge, fermion_out, lat_x,
                                                  lat_y, lat_z, lat_t, parity,
                                                  b_z_recv_vec, f_z_recv_vec);
    } else {
      checkCudaErrors(cudaDeviceSynchronize());
      wilson_dslash_z_recv<<<gridDim, blockDim>>>(gauge, fermion_out, lat_x,
                                                  lat_y, lat_z, lat_t, parity,
                                                  f_z_send_vec, b_z_send_vec);
    }
    // recv t
    if (grid_t != 1) {
      MPI_Wait(&b_t_recv_request, MPI_STATUS_IGNORE);
      MPI_Wait(&f_t_recv_request, MPI_STATUS_IGNORE);
      wilson_dslash_t_recv<<<gridDim, blockDim>>>(gauge, fermion_out, lat_x,
                                                  lat_y, lat_z, lat_t, parity,
                                                  b_t_recv_vec, f_t_recv_vec);
    } else {
      checkCudaErrors(cudaDeviceSynchronize());
      wilson_dslash_t_recv<<<gridDim, blockDim>>>(gauge, fermion_out, lat_x,
                                                  lat_y, lat_z, lat_t, parity,
                                                  f_t_send_vec, b_t_send_vec);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    {
      checkCudaErrors(cudaDeviceSynchronize());
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      err = cudaGetLastError();
      checkCudaErrors(err);
      printf(
          "mpi wilson dslash total time: (without malloc free memcpy) :%.9lf "
          "sec\n",
          double(duration) / 1e9);
    }
    {
      // free
      checkCudaErrors(cudaFree(b_x_send_vec));
      checkCudaErrors(cudaFree(f_x_send_vec));
      checkCudaErrors(cudaFree(b_y_send_vec));
      checkCudaErrors(cudaFree(f_y_send_vec));
      checkCudaErrors(cudaFree(b_z_send_vec));
      checkCudaErrors(cudaFree(f_z_send_vec));
      checkCudaErrors(cudaFree(b_t_send_vec));
      checkCudaErrors(cudaFree(f_t_send_vec));
      checkCudaErrors(cudaFree(b_x_recv_vec));
      checkCudaErrors(cudaFree(f_x_recv_vec));
      checkCudaErrors(cudaFree(b_y_recv_vec));
      checkCudaErrors(cudaFree(f_y_recv_vec));
      checkCudaErrors(cudaFree(b_z_recv_vec));
      checkCudaErrors(cudaFree(f_z_recv_vec));
      checkCudaErrors(cudaFree(b_t_recv_vec));
      checkCudaErrors(cudaFree(f_t_recv_vec));
    }
  }
}
#endif