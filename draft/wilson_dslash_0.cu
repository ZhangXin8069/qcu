#include "../include/qcu.h"
#ifdef WILSON_DSLASH
__global__ void wilson_dslash(void *device_U, void *device_src,
                              void *device_dest, void *device_xyztsc,
                              const int device_parity) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int parity = idx;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  const int lat_x = xyztsc[_X_];
  const int lat_y = xyztsc[_Y_];
  const int lat_z = xyztsc[_Z_];
  const int lat_t = xyztsc[_T_];
  const int lat_tzyxcc = xyztsc[_TZYXCC_];
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
      ((static_cast<LatticeComplex *>(device_U)) + idx * _LAT_CC_);
  LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + idx * _LAT_SC_);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + idx * _LAT_SC_);
  LatticeComplex *tmp_U;
  LatticeComplex *tmp_src;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex src[_LAT_SC_];
  LatticeComplex dest[_LAT_SC_];
  // just wilson(Sum part)
  {
    // x-1
    move_backward_x(move, x, lat_x, eo, parity);
    tmp_U = (origin_U + move * _LAT_CC_ + (1 - parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * _LAT_SC_);
    give_ptr(src, tmp_src, _LAT_SC_);
  }
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        tmp0 +=
            (src[c1] + src[c1 + _LAT_CC_] * I) * U[c1 * _LAT_C_ + c0].conj();
        tmp1 += (src[c1 + _LAT_C_] + src[c1 + _LAT_HALF_SC_] * I) *
                U[c1 * _LAT_C_ + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + _LAT_C_] += tmp1;
      dest[c0 + _LAT_HALF_SC_] -= tmp1 * I;
      dest[c0 + _LAT_CC_] -= tmp0 * I;
    }
  }
  {
    // x+1
    move_forward_x(move, x, lat_x, eo, parity);
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * _LAT_SC_);
    give_ptr(src, tmp_src, _LAT_SC_);
  }
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        tmp0 += (src[c1] - src[c1 + _LAT_CC_] * I) * U[c0 * _LAT_C_ + c1];
        tmp1 += (src[c1 + _LAT_C_] - src[c1 + _LAT_HALF_SC_] * I) *
                U[c0 * _LAT_C_ + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + _LAT_C_] += tmp1;
      dest[c0 + _LAT_HALF_SC_] += tmp1 * I;
      dest[c0 + _LAT_CC_] += tmp0 * I;
    }
  }
  {
    // y-1
    move_backward(move, y, lat_y);
    tmp_U = (origin_U + move * lat_x * _LAT_CC_ + (3 - parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_x * _LAT_SC_);
    give_ptr(src, tmp_src, _LAT_SC_);
  }
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        tmp0 += (src[c1] - src[c1 + _LAT_CC_]) * U[c1 * _LAT_C_ + c0].conj();
        tmp1 += (src[c1 + _LAT_C_] + src[c1 + _LAT_HALF_SC_]) *
                U[c1 * _LAT_C_ + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + _LAT_C_] += tmp1;
      dest[c0 + _LAT_HALF_SC_] += tmp1;
      dest[c0 + _LAT_CC_] -= tmp0;
    }
  }
  {
    // y+1
    move_forward(move, y, lat_y);
    tmp_U = (origin_U + (2 + parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_x * _LAT_SC_);
    give_ptr(src, tmp_src, _LAT_SC_);
  }
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        tmp0 += (src[c1] + src[c1 + _LAT_CC_]) * U[c0 * _LAT_C_ + c1];
        tmp1 += (src[c1 + _LAT_C_] - src[c1 + _LAT_HALF_SC_]) *
                U[c0 * _LAT_C_ + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + _LAT_C_] += tmp1;
      dest[c0 + _LAT_HALF_SC_] -= tmp1;
      dest[c0 + _LAT_CC_] += tmp0;
    }
  }
  {
    // z-1
    move_backward(move, z, lat_z);
    tmp_U = (origin_U + move * lat_y * lat_x * _LAT_CC_ +
             (5 - parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_y * lat_x * _LAT_SC_);
    give_ptr(src, tmp_src, _LAT_SC_);
  }
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        tmp0 += (src[c1] + src[c1 + _LAT_HALF_SC_] * I) *
                U[c1 * _LAT_C_ + c0].conj();
        tmp1 += (src[c1 + _LAT_C_] - src[c1 + _LAT_CC_] * I) *
                U[c1 * _LAT_C_ + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + _LAT_C_] += tmp1;
      dest[c0 + _LAT_HALF_SC_] -= tmp0 * I;
      dest[c0 + _LAT_CC_] += tmp1 * I;
    }
  }
  {
    // z+1
    move_forward(move, z, lat_z);
    tmp_U = (origin_U + (4 + parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_y * lat_x * _LAT_SC_);
    give_ptr(src, tmp_src, _LAT_SC_);
  }
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        tmp0 += (src[c1] - src[c1 + _LAT_HALF_SC_] * I) * U[c0 * _LAT_C_ + c1];
        tmp1 +=
            (src[c1 + _LAT_C_] + src[c1 + _LAT_CC_] * I) * U[c0 * _LAT_C_ + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + _LAT_C_] += tmp1;
      dest[c0 + _LAT_HALF_SC_] += tmp0 * I;
      dest[c0 + _LAT_CC_] -= tmp1 * I;
    }
  }
  {
    // t-1
    move_backward(move, t, lat_t);
    tmp_U = (origin_U + move * lat_z * lat_y * lat_x * _LAT_CC_ +
             (7 - parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_z * lat_y * lat_x * _LAT_SC_);
    give_ptr(src, tmp_src, _LAT_SC_);
  }
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        tmp0 +=
            (src[c1] + src[c1 + _LAT_HALF_SC_]) * U[c1 * _LAT_C_ + c0].conj();
        tmp1 += (src[c1 + _LAT_C_] + src[c1 + _LAT_CC_]) *
                U[c1 * _LAT_C_ + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + _LAT_C_] += tmp1;
      dest[c0 + _LAT_HALF_SC_] += tmp0;
      dest[c0 + _LAT_CC_] += tmp1;
    }
  }
  {
    // t+1
    move_forward(move, t, lat_t);
    tmp_U = (origin_U + (6 + parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_z * lat_y * lat_x * _LAT_SC_);
    give_ptr(src, tmp_src, _LAT_SC_);
  }
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        tmp0 += (src[c1] - src[c1 + _LAT_HALF_SC_]) * U[c0 * _LAT_C_ + c1];
        tmp1 += (src[c1 + _LAT_C_] - src[c1 + _LAT_CC_]) * U[c0 * _LAT_C_ + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + _LAT_C_] += tmp1;
      dest[c0 + _LAT_HALF_SC_] -= tmp0;
      dest[c0 + _LAT_CC_] -= tmp1;
    }
  }
  give_ptr(origin_dest, dest, _LAT_SC_);
}
__global__ void wilson_dslash_x_send(void *device_U, void *device_src,
                                     void *device_dest, void *device_xyztsc,
                                     const int device_parity,
                                     void *device_b_x_send_vec,
                                     void *device_f_x_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int parity = idx;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  const int lat_x = xyztsc[_X_];
  const int lat_y = xyztsc[_Y_];
  const int lat_z = xyztsc[_Z_];
  const int lat_tzyxcc = xyztsc[_TZYXCC_];
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
      ((static_cast<LatticeComplex *>(device_U)) + idx * _LAT_CC_);
  LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + idx * _LAT_SC_);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + idx * _LAT_SC_);
  LatticeComplex *origin_b_x_send_vec =
      ((static_cast<LatticeComplex *>(device_b_x_send_vec)) +
       (t * lat_z * lat_y + z * lat_y + y) * _LAT_HALF_SC_);
  LatticeComplex *origin_f_x_send_vec =
      ((static_cast<LatticeComplex *>(device_f_x_send_vec)) +
       (t * lat_z * lat_y + z * lat_y + y) * _LAT_HALF_SC_);
  LatticeComplex *tmp_U;
  LatticeComplex *tmp_src;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex src[_LAT_SC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_x_send_vec[_LAT_HALF_SC_];
  LatticeComplex f_x_send_vec[_LAT_HALF_SC_];
  { // x-1
    move_backward_x(move, x, lat_x, eo, parity);
    if (move != lat_x - 1) {
      tmp_U = (origin_U + move * _LAT_CC_ + (1 - parity) * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * _LAT_SC_);
      give_ptr(src, tmp_src, _LAT_SC_);
      {
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 += (src[c1] + src[c1 + _LAT_CC_] * I) *
                    U[c1 * _LAT_C_ + c0].conj();
            tmp1 += (src[c1 + _LAT_C_] + src[c1 + _LAT_HALF_SC_] * I) *
                    U[c1 * _LAT_C_ + c0].conj();
          }
          dest[c0] += tmp0;
          dest[c0 + _LAT_C_] += tmp1;
          dest[c0 + _LAT_HALF_SC_] -= tmp1 * I;
          dest[c0 + _LAT_CC_] -= tmp0 * I;
        }
      }
    }
    if (x == 0 && move == 0) { // even-odd
      // send in x+1 way
      give_ptr(src, origin_src, _LAT_SC_);
      { // sigma src
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          b_x_send_vec[c1] = src[c1] - src[c1 + _LAT_CC_] * I;
          b_x_send_vec[c1 + _LAT_C_] =
              src[c1 + _LAT_C_] - src[c1 + _LAT_HALF_SC_] * I;
        }
        give_ptr(origin_b_x_send_vec, b_x_send_vec, _LAT_HALF_SC_);
      }
    }
  }
  { // x+1
    move_forward_x(move, x, lat_x, eo, parity);
    if (move != 1 - lat_x) {
      tmp_U = (origin_U + parity * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * _LAT_SC_);
      give_ptr(src, tmp_src, _LAT_SC_);
      {
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 += (src[c1] - src[c1 + _LAT_CC_] * I) * U[c0 * _LAT_C_ + c1];
            tmp1 += (src[c1 + _LAT_C_] - src[c1 + _LAT_HALF_SC_] * I) *
                    U[c0 * _LAT_C_ + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + _LAT_C_] += tmp1;
          dest[c0 + _LAT_HALF_SC_] += tmp1 * I;
          dest[c0 + _LAT_CC_] += tmp0 * I;
        }
      }
    }
    if (x == lat_x - 1 && move == 0) { // even-odd
      // send in x-1 way
      tmp_U = (origin_U + (1 - parity) * lat_tzyxcc); // even-odd
      give_u(U, tmp_U);
      give_ptr(src, origin_src, _LAT_SC_);
      { // just tmp
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 += (src[c1] + src[c1 + _LAT_CC_] * I) *
                    U[c1 * _LAT_C_ + c0].conj();
            tmp1 += (src[c1 + _LAT_C_] + src[c1 + _LAT_HALF_SC_] * I) *
                    U[c1 * _LAT_C_ + c0].conj();
          }
          f_x_send_vec[c0] = tmp0;
          f_x_send_vec[c0 + _LAT_C_] = tmp1;
        }
        give_ptr(origin_f_x_send_vec, f_x_send_vec, _LAT_HALF_SC_);
      }
    }
  } // just add
  add_ptr(origin_dest, dest, _LAT_SC_);
}
__global__ void wilson_dslash_x_recv(void *device_U, void *device_dest,
                                     void *device_xyztsc,
                                     const int device_parity,
                                     void *device_b_x_recv_vec,
                                     void *device_f_x_recv_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int parity = idx;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  const int lat_x = xyztsc[_X_];
  const int lat_y = xyztsc[_Y_];
  const int lat_z = xyztsc[_Z_];
  const int lat_tzyxcc = xyztsc[_TZYXCC_];
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
      ((static_cast<LatticeComplex *>(device_U)) + idx * _LAT_CC_);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + idx * _LAT_SC_);
  LatticeComplex *origin_b_x_recv_vec =
      ((static_cast<LatticeComplex *>(device_b_x_recv_vec)) +
       (t * lat_z * lat_y + z * lat_y + y) * _LAT_HALF_SC_);
  LatticeComplex *origin_f_x_recv_vec =
      ((static_cast<LatticeComplex *>(device_f_x_recv_vec)) +
       (t * lat_z * lat_y + z * lat_y + y) * _LAT_HALF_SC_);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_x_recv_vec[_LAT_HALF_SC_];
  LatticeComplex f_x_recv_vec[_LAT_HALF_SC_]; // needed
  {                                           // x-1
    move_backward_x(move, x, lat_x, eo, parity);
    if (move == lat_x - 1) { // recv in x-1 way
      give_ptr(b_x_recv_vec, origin_b_x_recv_vec, _LAT_HALF_SC_);
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        dest[c0] += b_x_recv_vec[c0];
        dest[c0 + _LAT_C_] += b_x_recv_vec[c0 + _LAT_C_];
        dest[c0 + _LAT_HALF_SC_] -= b_x_recv_vec[c0 + _LAT_C_] * I;
        dest[c0 + _LAT_CC_] -= b_x_recv_vec[c0] * I;
      }
    }
  }
  { // x+1
    move_forward_x(move, x, lat_x, eo, parity);
    if (move == 1 - lat_x) { // recv in x+1 way
      give_ptr(f_x_recv_vec, origin_f_x_recv_vec, _LAT_HALF_SC_);
      tmp_U = (origin_U + parity * lat_tzyxcc);
      give_u(U, tmp_U);
      {
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 += f_x_recv_vec[c1] * U[c0 * _LAT_C_ + c1];
            tmp1 += f_x_recv_vec[c1 + _LAT_C_] * U[c0 * _LAT_C_ + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + _LAT_C_] += tmp1;
          dest[c0 + _LAT_HALF_SC_] += tmp1 * I;
          dest[c0 + _LAT_CC_] += tmp0 * I;
        }
      }
    }
  } // just add
  add_ptr(origin_dest, dest, _LAT_SC_);
}
__global__ void wilson_dslash_y_send(void *device_U, void *device_src,
                                     void *device_dest, void *device_xyztsc,
                                     const int device_parity,
                                     void *device_b_y_send_vec,
                                     void *device_f_y_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int parity = idx;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  const int lat_x = xyztsc[_X_];
  const int lat_y = xyztsc[_Y_];
  const int lat_z = xyztsc[_Z_];
  const int lat_tzyxcc = xyztsc[_TZYXCC_];
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
      ((static_cast<LatticeComplex *>(device_U)) + idx * _LAT_CC_);
  LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + idx * _LAT_SC_);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + idx * _LAT_SC_);
  LatticeComplex *origin_b_y_send_vec =
      ((static_cast<LatticeComplex *>(device_b_y_send_vec)) +
       (t * lat_z * lat_x + z * lat_x + x) * _LAT_HALF_SC_);
  LatticeComplex *origin_f_y_send_vec =
      ((static_cast<LatticeComplex *>(device_f_y_send_vec)) +
       (t * lat_z * lat_x + z * lat_x + x) * _LAT_HALF_SC_);
  LatticeComplex *tmp_U;
  LatticeComplex *tmp_src;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex src[_LAT_SC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_y_send_vec[_LAT_HALF_SC_];
  LatticeComplex f_y_send_vec[_LAT_HALF_SC_];
  { // y-1
    move_backward(move, y, lat_y);
    if (move == -1) {
      tmp_U = (origin_U + move * lat_x * _LAT_CC_ + (3 - parity) * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * lat_x * _LAT_SC_);
      give_ptr(src, tmp_src, _LAT_SC_);
      {
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 +=
                (src[c1] - src[c1 + _LAT_CC_]) * U[c1 * _LAT_C_ + c0].conj();
            tmp1 += (src[c1 + _LAT_C_] + src[c1 + _LAT_HALF_SC_]) *
                    U[c1 * _LAT_C_ + c0].conj();
          }
          dest[c0] += tmp0;
          dest[c0 + _LAT_C_] += tmp1;
          dest[c0 + _LAT_HALF_SC_] += tmp1;
          dest[c0 + _LAT_CC_] -= tmp0;
        }
      }
    } else { // send in y+1 way
      give_ptr(src, origin_src, _LAT_SC_);
      { // sigma src
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          b_y_send_vec[c1] = src[c1] + src[c1 + _LAT_CC_];
          b_y_send_vec[c1 + _LAT_C_] =
              src[c1 + _LAT_C_] - src[c1 + _LAT_HALF_SC_];
        }
        give_ptr(origin_b_y_send_vec, b_y_send_vec, _LAT_HALF_SC_);
      }
    }
  }
  { // y+1
    move_forward(move, y, lat_y);
    if (move == 1) {
      tmp_U = (origin_U + (2 + parity) * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * lat_x * _LAT_SC_);
      give_ptr(src, tmp_src, _LAT_SC_);
      {
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 += (src[c1] + src[c1 + _LAT_CC_]) * U[c0 * _LAT_C_ + c1];
            tmp1 += (src[c1 + _LAT_C_] - src[c1 + _LAT_HALF_SC_]) *
                    U[c0 * _LAT_C_ + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + _LAT_C_] += tmp1;
          dest[c0 + _LAT_HALF_SC_] -= tmp1;
          dest[c0 + _LAT_CC_] += tmp0;
        }
      }
    } else {                                          // send in y-1 way
      tmp_U = (origin_U + (3 - parity) * lat_tzyxcc); // even-odd
      give_u(U, tmp_U);
      give_ptr(src, origin_src, _LAT_SC_);
      { // just tmp
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 +=
                (src[c1] - src[c1 + _LAT_CC_]) * U[c1 * _LAT_C_ + c0].conj();
            tmp1 += (src[c1 + _LAT_C_] + src[c1 + _LAT_HALF_SC_]) *
                    U[c1 * _LAT_C_ + c0].conj();
          }
          f_y_send_vec[c0] = tmp0;
          f_y_send_vec[c0 + _LAT_C_] = tmp1;
        }
        give_ptr(origin_f_y_send_vec, f_y_send_vec, _LAT_HALF_SC_);
      }
    }
  } // just add
  add_ptr(origin_dest, dest, _LAT_SC_);
}
__global__ void wilson_dslash_y_recv(void *device_U, void *device_dest,
                                     void *device_xyztsc,
                                     const int device_parity,
                                     void *device_b_y_recv_vec,
                                     void *device_f_y_recv_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int parity = idx;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  const int lat_x = xyztsc[_X_];
  const int lat_y = xyztsc[_Y_];
  const int lat_z = xyztsc[_Z_];
  const int lat_tzyxcc = xyztsc[_TZYXCC_];
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
      ((static_cast<LatticeComplex *>(device_U)) + idx * _LAT_CC_);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + idx * _LAT_SC_);
  LatticeComplex *origin_b_y_recv_vec =
      ((static_cast<LatticeComplex *>(device_b_y_recv_vec)) +
       (t * lat_z * lat_x + z * lat_x + x) * _LAT_HALF_SC_);
  LatticeComplex *origin_f_y_recv_vec =
      ((static_cast<LatticeComplex *>(device_f_y_recv_vec)) +
       (t * lat_z * lat_x + z * lat_x + x) * _LAT_HALF_SC_);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_y_recv_vec[_LAT_HALF_SC_];
  LatticeComplex f_y_recv_vec[_LAT_HALF_SC_]; // needed
  {                                           // y-1
    move_backward(move, y, lat_y);
    if (move != -1) { // recv in y-1 way
      give_ptr(b_y_recv_vec, origin_b_y_recv_vec, _LAT_HALF_SC_);
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        dest[c0] += b_y_recv_vec[c0];
        dest[c0 + _LAT_C_] += b_y_recv_vec[c0 + _LAT_C_];
        dest[c0 + _LAT_HALF_SC_] += b_y_recv_vec[c0 + _LAT_C_];
        dest[c0 + _LAT_CC_] -= b_y_recv_vec[c0];
      }
    }
  }
  { // y+1
    move_forward(move, y, lat_y);
    if (move != 1) { // recv in y+1 way
      give_ptr(f_y_recv_vec, origin_f_y_recv_vec, _LAT_HALF_SC_);
      tmp_U = (origin_U + (2 + parity) * lat_tzyxcc);
      give_u(U, tmp_U);
      {
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 += f_y_recv_vec[c1] * U[c0 * _LAT_C_ + c1];
            tmp1 += f_y_recv_vec[c1 + _LAT_C_] * U[c0 * _LAT_C_ + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + _LAT_C_] += tmp1;
          dest[c0 + _LAT_HALF_SC_] -= tmp1;
          dest[c0 + _LAT_CC_] += tmp0;
        }
      }
    }
  } // just add
  add_ptr(origin_dest, dest, _LAT_SC_);
}
__global__ void wilson_dslash_z_send(void *device_U, void *device_src,
                                     void *device_dest, void *device_xyztsc,
                                     const int device_parity,
                                     void *device_b_z_send_vec,
                                     void *device_f_z_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int parity = idx;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  const int lat_x = xyztsc[_X_];
  const int lat_y = xyztsc[_Y_];
  const int lat_z = xyztsc[_Z_];
  const int lat_tzyxcc = xyztsc[_TZYXCC_];
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
      ((static_cast<LatticeComplex *>(device_U)) + idx * _LAT_CC_);
  LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + idx * _LAT_SC_);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + idx * _LAT_SC_);
  LatticeComplex *origin_b_z_send_vec =
      ((static_cast<LatticeComplex *>(device_b_z_send_vec)) +
       (t * lat_y * lat_x + y * lat_x + x) * _LAT_HALF_SC_);
  LatticeComplex *origin_f_z_send_vec =
      ((static_cast<LatticeComplex *>(device_f_z_send_vec)) +
       (t * lat_y * lat_x + y * lat_x + x) * _LAT_HALF_SC_);
  LatticeComplex *tmp_U;
  LatticeComplex *tmp_src;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex src[_LAT_SC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_z_send_vec[_LAT_HALF_SC_];
  LatticeComplex f_z_send_vec[_LAT_HALF_SC_];
  { // z-1
    move_backward(move, z, lat_z);
    if (move == -1) {
      tmp_U = (origin_U + move * lat_y * lat_x * _LAT_CC_ +
               (5 - parity) * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * lat_y * lat_x * _LAT_SC_);
      give_ptr(src, tmp_src, _LAT_SC_);
      {
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 += (src[c1] + src[c1 + _LAT_HALF_SC_] * I) *
                    U[c1 * _LAT_C_ + c0].conj();
            tmp1 += (src[c1 + _LAT_C_] - src[c1 + _LAT_CC_] * I) *
                    U[c1 * _LAT_C_ + c0].conj();
          }
          dest[c0] += tmp0;
          dest[c0 + _LAT_C_] += tmp1;
          dest[c0 + _LAT_HALF_SC_] -= tmp0 * I;
          dest[c0 + _LAT_CC_] += tmp1 * I;
        }
      }
    } else { // send in z+1 way
      give_ptr(src, origin_src, _LAT_SC_);
      { // sigma src
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          b_z_send_vec[c1] = src[c1] - src[c1 + _LAT_HALF_SC_] * I;
          b_z_send_vec[c1 + _LAT_C_] =
              src[c1 + _LAT_C_] + src[c1 + _LAT_CC_] * I;
        }
        give_ptr(origin_b_z_send_vec, b_z_send_vec, _LAT_HALF_SC_);
      }
    }
  }
  { // z+1
    move_forward(move, z, lat_z);
    if (move == 1) {
      tmp_U = (origin_U + (4 + parity) * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * lat_y * lat_x * _LAT_SC_);
      give_ptr(src, tmp_src, _LAT_SC_);
      {
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 +=
                (src[c1] - src[c1 + _LAT_HALF_SC_] * I) * U[c0 * _LAT_C_ + c1];
            tmp1 += (src[c1 + _LAT_C_] + src[c1 + _LAT_CC_] * I) *
                    U[c0 * _LAT_C_ + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + _LAT_C_] += tmp1;
          dest[c0 + _LAT_HALF_SC_] += tmp0 * I;
          dest[c0 + _LAT_CC_] -= tmp1 * I;
        }
      }
    } else {                                          // send in z-1 way
      tmp_U = (origin_U + (5 - parity) * lat_tzyxcc); // even-odd
      give_u(U, tmp_U);
      give_ptr(src, origin_src, _LAT_SC_);
      { // just tmp
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 += (src[c1] + src[c1 + _LAT_HALF_SC_] * I) *
                    U[c1 * _LAT_C_ + c0].conj();
            tmp1 += (src[c1 + _LAT_C_] - src[c1 + _LAT_CC_] * I) *
                    U[c1 * _LAT_C_ + c0].conj();
          }
          f_z_send_vec[c0] = tmp0;
          f_z_send_vec[c0 + _LAT_C_] = tmp1;
        }
        give_ptr(origin_f_z_send_vec, f_z_send_vec, _LAT_HALF_SC_);
      }
    }
  } // just add
  add_ptr(origin_dest, dest, _LAT_SC_);
}
__global__ void wilson_dslash_z_recv(void *device_U, void *device_dest,
                                     void *device_xyztsc,
                                     const int device_parity,
                                     void *device_b_z_recv_vec,
                                     void *device_f_z_recv_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int parity = idx;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  const int lat_x = xyztsc[_X_];
  const int lat_y = xyztsc[_Y_];
  const int lat_z = xyztsc[_Z_];
  const int lat_tzyxcc = xyztsc[_TZYXCC_];
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
      ((static_cast<LatticeComplex *>(device_U)) + idx * _LAT_CC_);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + idx * _LAT_SC_);
  LatticeComplex *origin_b_z_recv_vec =
      ((static_cast<LatticeComplex *>(device_b_z_recv_vec)) +
       (t * lat_y * lat_x + y * lat_x + x) * _LAT_HALF_SC_);
  LatticeComplex *origin_f_z_recv_vec =
      ((static_cast<LatticeComplex *>(device_f_z_recv_vec)) +
       (t * lat_y * lat_x + y * lat_x + x) * _LAT_HALF_SC_);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_z_recv_vec[_LAT_HALF_SC_];
  LatticeComplex f_z_recv_vec[_LAT_HALF_SC_]; // needed
  {                                           // z-1
    move_backward(move, z, lat_z);
    if (move != -1) { // recv in z-1 way
      give_ptr(b_z_recv_vec, origin_b_z_recv_vec, _LAT_HALF_SC_);
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        dest[c0] += b_z_recv_vec[c0];
        dest[c0 + _LAT_C_] += b_z_recv_vec[c0 + _LAT_C_];
        dest[c0 + _LAT_HALF_SC_] -= b_z_recv_vec[c0] * I;
        dest[c0 + _LAT_CC_] += b_z_recv_vec[c0 + _LAT_C_] * I;
      }
    }
  }
  { // z+1
    move_forward(move, z, lat_z);
    if (move != 1) { // recv in z+1 way
      give_ptr(f_z_recv_vec, origin_f_z_recv_vec, _LAT_HALF_SC_);
      tmp_U = (origin_U + (4 + parity) * lat_tzyxcc);
      give_u(U, tmp_U);
      {
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 += f_z_recv_vec[c1] * U[c0 * _LAT_C_ + c1];
            tmp1 += f_z_recv_vec[c1 + _LAT_C_] * U[c0 * _LAT_C_ + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + _LAT_C_] += tmp1;
          dest[c0 + _LAT_HALF_SC_] += tmp0 * I;
          dest[c0 + _LAT_CC_] -= tmp1 * I;
        }
      }
    }
  } // just add
  add_ptr(origin_dest, dest, _LAT_SC_);
}
__global__ void wilson_dslash_t_send(void *device_U, void *device_src,
                                     void *device_dest, void *device_xyztsc,
                                     const int device_parity,
                                     void *device_b_t_send_vec,
                                     void *device_f_t_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int parity = idx;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  const int lat_x = xyztsc[_X_];
  const int lat_y = xyztsc[_Y_];
  const int lat_z = xyztsc[_Z_];
  const int lat_t = xyztsc[_T_];
  const int lat_tzyxcc = xyztsc[_TZYXCC_];
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
      ((static_cast<LatticeComplex *>(device_U)) + idx * _LAT_CC_);
  LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + idx * _LAT_SC_);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + idx * _LAT_SC_);
  LatticeComplex *origin_b_t_send_vec =
      ((static_cast<LatticeComplex *>(device_b_t_send_vec)) +
       (z * lat_y * lat_x + y * lat_x + x) * _LAT_HALF_SC_);
  LatticeComplex *origin_f_t_send_vec =
      ((static_cast<LatticeComplex *>(device_f_t_send_vec)) +
       (z * lat_y * lat_x + y * lat_x + x) * _LAT_HALF_SC_);
  LatticeComplex *tmp_U;
  LatticeComplex *tmp_src;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex src[_LAT_SC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_t_send_vec[_LAT_HALF_SC_];
  LatticeComplex f_t_send_vec[_LAT_HALF_SC_];
  { // t-1
    move_backward(move, t, lat_t);
    if (move == -1) {
      tmp_U = (origin_U + move * lat_z * lat_y * lat_x * _LAT_CC_ +
               (7 - parity) * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * lat_z * lat_y * lat_x * _LAT_SC_);
      give_ptr(src, tmp_src, _LAT_SC_);
      {
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 += (src[c1] + src[c1 + _LAT_HALF_SC_]) *
                    U[c1 * _LAT_C_ + c0].conj();
            tmp1 += (src[c1 + _LAT_C_] + src[c1 + _LAT_CC_]) *
                    U[c1 * _LAT_C_ + c0].conj();
          }
          dest[c0] += tmp0;
          dest[c0 + _LAT_C_] += tmp1;
          dest[c0 + _LAT_HALF_SC_] += tmp0;
          dest[c0 + _LAT_CC_] += tmp1;
        }
      }
    } else { // send in t+1 way
      give_ptr(src, origin_src, _LAT_SC_);
      { // sigma src
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          b_t_send_vec[c1] = src[c1] - src[c1 + _LAT_HALF_SC_];
          b_t_send_vec[c1 + _LAT_C_] = src[c1 + _LAT_C_] - src[c1 + _LAT_CC_];
        }
        give_ptr(origin_b_t_send_vec, b_t_send_vec, _LAT_HALF_SC_);
      }
    }
  }
  { // t+1
    move_forward(move, t, lat_t);
    if (move == 1) {
      tmp_U = (origin_U + (6 + parity) * lat_tzyxcc);
      give_u(U, tmp_U);
      tmp_src = (origin_src + move * lat_z * lat_y * lat_x * _LAT_SC_);
      give_ptr(src, tmp_src, _LAT_SC_);
      {
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 += (src[c1] - src[c1 + _LAT_HALF_SC_]) * U[c0 * _LAT_C_ + c1];
            tmp1 +=
                (src[c1 + _LAT_C_] - src[c1 + _LAT_CC_]) * U[c0 * _LAT_C_ + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + _LAT_C_] += tmp1;
          dest[c0 + _LAT_HALF_SC_] -= tmp0;
          dest[c0 + _LAT_CC_] -= tmp1;
        }
      }
    } else {                                          // send in t-1 way
      tmp_U = (origin_U + (7 - parity) * lat_tzyxcc); // even-odd
      give_u(U, tmp_U);
      give_ptr(src, origin_src, _LAT_SC_);
      { // just tmp
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 += (src[c1] + src[c1 + _LAT_HALF_SC_]) *
                    U[c1 * _LAT_C_ + c0].conj();
            tmp1 += (src[c1 + _LAT_C_] + src[c1 + _LAT_CC_]) *
                    U[c1 * _LAT_C_ + c0].conj();
          }
          f_t_send_vec[c0] = tmp0;
          f_t_send_vec[c0 + _LAT_C_] = tmp1;
        }
        give_ptr(origin_f_t_send_vec, f_t_send_vec, _LAT_HALF_SC_);
      }
    }
  } // just add
  add_ptr(origin_dest, dest, _LAT_SC_);
}
__global__ void wilson_dslash_t_recv(void *device_U, void *device_dest,
                                     void *device_xyztsc,
                                     const int device_parity,
                                     void *device_b_t_recv_vec,
                                     void *device_f_t_recv_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int parity = idx;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  const int lat_x = xyztsc[_X_];
  const int lat_y = xyztsc[_Y_];
  const int lat_z = xyztsc[_Z_];
  const int lat_t = xyztsc[_T_];
  const int lat_tzyxcc = xyztsc[_TZYXCC_];
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
      ((static_cast<LatticeComplex *>(device_U)) + idx * _LAT_CC_);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + idx * _LAT_SC_);
  LatticeComplex *origin_b_t_recv_vec =
      ((static_cast<LatticeComplex *>(device_b_t_recv_vec)) +
       (z * lat_y * lat_x + y * lat_x + x) * _LAT_HALF_SC_);
  LatticeComplex *origin_f_t_recv_vec =
      ((static_cast<LatticeComplex *>(device_f_t_recv_vec)) +
       (z * lat_y * lat_x + y * lat_x + x) * _LAT_HALF_SC_);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_t_recv_vec[_LAT_HALF_SC_];
  LatticeComplex f_t_recv_vec[_LAT_HALF_SC_]; // needed
  {                                           // t-1
    move_backward(move, t, lat_t);
    if (move != -1) { // recv in t-1 way
      give_ptr(b_t_recv_vec, origin_b_t_recv_vec, _LAT_HALF_SC_);
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        dest[c0] += b_t_recv_vec[c0];
        dest[c0 + _LAT_C_] += b_t_recv_vec[c0 + _LAT_C_];
        dest[c0 + _LAT_HALF_SC_] += b_t_recv_vec[c0];
        dest[c0 + _LAT_CC_] += b_t_recv_vec[c0 + _LAT_C_];
      }
    }
  }
  { // t+1
    move_forward(move, t, lat_t);
    if (move != 1) { // recv in t+1 way
      give_ptr(f_t_recv_vec, origin_f_t_recv_vec, _LAT_HALF_SC_);
      tmp_U = (origin_U + (6 + parity) * lat_tzyxcc);
      give_u(U, tmp_U);
      {
        for (int c0 = 0; c0 < _LAT_C_; c0++) {
          tmp0 = zero;
          tmp1 = zero;
          for (int c1 = 0; c1 < _LAT_C_; c1++) {
            tmp0 += f_t_recv_vec[c1] * U[c0 * _LAT_C_ + c1];
            tmp1 += f_t_recv_vec[c1 + _LAT_C_] * U[c0 * _LAT_C_ + c1];
          }
          dest[c0] += tmp0;
          dest[c0 + _LAT_C_] += tmp1;
          dest[c0 + _LAT_HALF_SC_] -= tmp0;
          dest[c0 + _LAT_CC_] -= tmp1;
        }
      }
    }
  } // just add
  add_ptr(origin_dest, dest, _LAT_SC_);
}
#endif