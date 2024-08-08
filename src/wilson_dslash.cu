#include "../include/qcu.h"
#include "wilson_dslash.h"
#ifdef WILSON_DSLASH
#define __X__
#define __Y__
#define __Z__
#define __T__
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
#ifdef __X__
  { // x part
   {// x-1
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
      tmp0 += (src[c1] + src[c1 + _LAT_3C_] * I) * U[c1 * _LAT_C_ + c0].conj();
      tmp1 += (src[c1 + _LAT_1C_] + src[c1 + _LAT_2C_] * I) *
              U[c1 * _LAT_C_ + c0].conj();
    }
    dest[c0] += tmp0;
    dest[c0 + _LAT_1C_] += tmp1;
    dest[c0 + _LAT_2C_] -= tmp1 * I;
    dest[c0 + _LAT_3C_] -= tmp0 * I;
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
      tmp0 += (src[c1] - src[c1 + _LAT_3C_] * I) * U[c0 * _LAT_C_ + c1];
      tmp1 +=
          (src[c1 + _LAT_1C_] - src[c1 + _LAT_2C_] * I) * U[c0 * _LAT_C_ + c1];
    }
    dest[c0] += tmp0;
    dest[c0 + _LAT_1C_] += tmp1;
    dest[c0 + _LAT_2C_] += tmp1 * I;
    dest[c0 + _LAT_3C_] += tmp0 * I;
  }
}
}
#endif
#ifdef __Y__
{ // y part
 {// y-1
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
      tmp0 += (src[c1] - src[c1 + _LAT_3C_]) * U[c1 * _LAT_C_ + c0].conj();
      tmp1 += (src[c1 + _LAT_1C_] + src[c1 + _LAT_2C_]) *
              U[c1 * _LAT_C_ + c0].conj();
    }
    dest[c0] += tmp0;
    dest[c0 + _LAT_1C_] += tmp1;
    dest[c0 + _LAT_2C_] += tmp1;
    dest[c0 + _LAT_3C_] -= tmp0;
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
      tmp0 += (src[c1] + src[c1 + _LAT_3C_]) * U[c0 * _LAT_C_ + c1];
      tmp1 += (src[c1 + _LAT_1C_] - src[c1 + _LAT_2C_]) * U[c0 * _LAT_C_ + c1];
    }
    dest[c0] += tmp0;
    dest[c0 + _LAT_1C_] += tmp1;
    dest[c0 + _LAT_2C_] -= tmp1;
    dest[c0 + _LAT_3C_] += tmp0;
  }
}
}
#endif
#ifdef __Z__
{ // z part
 {// z-1
  move_backward(move, z, lat_z);
tmp_U =
    (origin_U + move * lat_y * lat_x * _LAT_CC_ + (5 - parity) * lat_tzyxcc);
give_u(U, tmp_U);
tmp_src = (origin_src + move * lat_y * lat_x * _LAT_SC_);
give_ptr(src, tmp_src, _LAT_SC_);
}
{
  for (int c0 = 0; c0 < _LAT_C_; c0++) {
    tmp0 = zero;
    tmp1 = zero;
    for (int c1 = 0; c1 < _LAT_C_; c1++) {
      tmp0 += (src[c1] + src[c1 + _LAT_2C_] * I) * U[c1 * _LAT_C_ + c0].conj();
      tmp1 += (src[c1 + _LAT_1C_] - src[c1 + _LAT_3C_] * I) *
              U[c1 * _LAT_C_ + c0].conj();
    }
    dest[c0] += tmp0;
    dest[c0 + _LAT_1C_] += tmp1;
    dest[c0 + _LAT_2C_] -= tmp0 * I;
    dest[c0 + _LAT_3C_] += tmp1 * I;
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
      tmp0 += (src[c1] - src[c1 + _LAT_2C_] * I) * U[c0 * _LAT_C_ + c1];
      tmp1 +=
          (src[c1 + _LAT_1C_] + src[c1 + _LAT_3C_] * I) * U[c0 * _LAT_C_ + c1];
    }
    dest[c0] += tmp0;
    dest[c0 + _LAT_1C_] += tmp1;
    dest[c0 + _LAT_2C_] += tmp0 * I;
    dest[c0 + _LAT_3C_] -= tmp1 * I;
  }
}
}
#endif
#ifdef __T__
{ // t part
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
        tmp0 += (src[c1] + src[c1 + _LAT_2C_]) * U[c1 * _LAT_C_ + c0].conj();
        tmp1 += (src[c1 + _LAT_1C_] + src[c1 + _LAT_3C_]) *
                U[c1 * _LAT_C_ + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + _LAT_1C_] += tmp1;
      dest[c0 + _LAT_2C_] += tmp0;
      dest[c0 + _LAT_3C_] += tmp1;
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
        tmp0 += (src[c1] - src[c1 + _LAT_2C_]) * U[c0 * _LAT_C_ + c1];
        tmp1 +=
            (src[c1 + _LAT_1C_] - src[c1 + _LAT_3C_]) * U[c0 * _LAT_C_ + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + _LAT_1C_] += tmp1;
      dest[c0 + _LAT_2C_] -= tmp0;
      dest[c0 + _LAT_3C_] -= tmp1;
    }
  }
}
#endif
give_ptr(origin_dest, dest, _LAT_SC_);
}
__global__ void wilson_dslash_inside(void *device_U, void *device_src,
                                     void *device_dest, void *device_xyztsc,
                                     int device_parity) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int parity = idx;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  int lat_x = xyztsc[_X_];
  int lat_y = xyztsc[_Y_];
  int lat_z = xyztsc[_Z_];
  int lat_t = xyztsc[_T_];
  int lat_tzyxcc = xyztsc[_TZYXCC_];
  int move;
  move = lat_x * lat_y * lat_z;
  int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  int z = parity / move;
  parity -= z * move;
  int y = parity / lat_x;
  int x = parity - y * lat_x;
  parity = device_parity;
  int eo = (y + z + t) & 0x01; // (y+z+t)%2
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
#ifdef __X__
  { // x part
   {// x-1
    move_backward_x(move, x, lat_x, eo, parity);
  tmp_U = (origin_U + move * _LAT_CC_ + (1 - parity) * lat_tzyxcc);
  give_u(U, tmp_U);
  tmp_src = (origin_src + move * _LAT_SC_);
  give_ptr(src, tmp_src, _LAT_SC_);
}
{
  LatticeComplex _(double((move != lat_x - 1)), 0.0); // just inside
  for (int c0 = 0; c0 < _LAT_C_; c0++) {
    tmp0 = zero;
    tmp1 = zero;
    for (int c1 = 0; c1 < _LAT_C_; c1++) {
      tmp0 += (src[c1] + src[c1 + _LAT_3C_] * I) * U[c1 * _LAT_C_ + c0].conj();
      tmp1 += (src[c1 + _LAT_1C_] + src[c1 + _LAT_2C_] * I) *
              U[c1 * _LAT_C_ + c0].conj();
    }
    dest[c0] += tmp0 * _;
    dest[c0 + _LAT_1C_] += tmp1 * _;
    dest[c0 + _LAT_2C_] -= tmp1 * I * _;
    dest[c0 + _LAT_3C_] -= tmp0 * I * _;
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
  LatticeComplex _(double((move != 1 - lat_x)), 0.0); // just inside
  for (int c0 = 0; c0 < _LAT_C_; c0++) {
    tmp0 = zero;
    tmp1 = zero;
    for (int c1 = 0; c1 < _LAT_C_; c1++) {
      tmp0 += (src[c1] - src[c1 + _LAT_3C_] * I) * U[c0 * _LAT_C_ + c1];
      tmp1 +=
          (src[c1 + _LAT_1C_] - src[c1 + _LAT_2C_] * I) * U[c0 * _LAT_C_ + c1];
    }
    dest[c0] += tmp0 * _;
    dest[c0 + _LAT_1C_] += tmp1 * _;
    dest[c0 + _LAT_2C_] += tmp1 * I * _;
    dest[c0 + _LAT_3C_] += tmp0 * I * _;
  }
}
}
#endif
#ifdef __Y__
{ // y part
 {// y-1
  move_backward(move, y, lat_y);
tmp_U = (origin_U + move * lat_x * _LAT_CC_ + (3 - parity) * lat_tzyxcc);
give_u(U, tmp_U);
tmp_src = (origin_src + move * lat_x * _LAT_SC_);
give_ptr(src, tmp_src, _LAT_SC_);
}
{
  LatticeComplex _(double((move == -1)), 0.0); // just inside
  for (int c0 = 0; c0 < _LAT_C_; c0++) {
    tmp0 = zero;
    tmp1 = zero;
    for (int c1 = 0; c1 < _LAT_C_; c1++) {
      tmp0 += (src[c1] - src[c1 + _LAT_3C_]) * U[c1 * _LAT_C_ + c0].conj();
      tmp1 += (src[c1 + _LAT_1C_] + src[c1 + _LAT_2C_]) *
              U[c1 * _LAT_C_ + c0].conj();
    }
    dest[c0] += tmp0 * _;
    dest[c0 + _LAT_1C_] += tmp1 * _;
    dest[c0 + _LAT_2C_] += tmp1 * _;
    dest[c0 + _LAT_3C_] -= tmp0 * _;
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
  LatticeComplex _(double((move == 1)), 0.0); // just inside
  for (int c0 = 0; c0 < _LAT_C_; c0++) {
    tmp0 = zero;
    tmp1 = zero;
    for (int c1 = 0; c1 < _LAT_C_; c1++) {
      tmp0 += (src[c1] + src[c1 + _LAT_3C_]) * U[c0 * _LAT_C_ + c1];
      tmp1 += (src[c1 + _LAT_1C_] - src[c1 + _LAT_2C_]) * U[c0 * _LAT_C_ + c1];
    }
    dest[c0] += tmp0 * _;
    dest[c0 + _LAT_1C_] += tmp1 * _;
    dest[c0 + _LAT_2C_] -= tmp1 * _;
    dest[c0 + _LAT_3C_] += tmp0 * _;
  }
}
}
#endif
#ifdef __Z__
{ // z part
 {// z-1
  move_backward(move, z, lat_z);
tmp_U =
    (origin_U + move * lat_y * lat_x * _LAT_CC_ + (5 - parity) * lat_tzyxcc);
give_u(U, tmp_U);
tmp_src = (origin_src + move * lat_y * lat_x * _LAT_SC_);
give_ptr(src, tmp_src, _LAT_SC_);
}
{
  LatticeComplex _(double((move == -1)), 0.0); // just inside
  for (int c0 = 0; c0 < _LAT_C_; c0++) {
    tmp0 = zero;
    tmp1 = zero;
    for (int c1 = 0; c1 < _LAT_C_; c1++) {
      tmp0 += (src[c1] + src[c1 + _LAT_2C_] * I) * U[c1 * _LAT_C_ + c0].conj();
      tmp1 += (src[c1 + _LAT_1C_] - src[c1 + _LAT_3C_] * I) *
              U[c1 * _LAT_C_ + c0].conj();
    }
    dest[c0] += tmp0 * _;
    dest[c0 + _LAT_1C_] += tmp1 * _;
    dest[c0 + _LAT_2C_] -= tmp0 * I * _;
    dest[c0 + _LAT_3C_] += tmp1 * I * _;
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
  LatticeComplex _(double((move == 1)), 0.0); // just inside
  for (int c0 = 0; c0 < _LAT_C_; c0++) {
    tmp0 = zero;
    tmp1 = zero;
    for (int c1 = 0; c1 < _LAT_C_; c1++) {
      tmp0 += (src[c1] - src[c1 + _LAT_2C_] * I) * U[c0 * _LAT_C_ + c1];
      tmp1 +=
          (src[c1 + _LAT_1C_] + src[c1 + _LAT_3C_] * I) * U[c0 * _LAT_C_ + c1];
    }
    dest[c0] += tmp0 * _;
    dest[c0 + _LAT_1C_] += tmp1 * _;
    dest[c0 + _LAT_2C_] += tmp0 * I * _;
    dest[c0 + _LAT_3C_] -= tmp1 * I * _;
  }
}
}
#endif
#ifdef __T__
{
  // t part
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
    LatticeComplex _(double((move == -1)), 0.0); // just inside
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        tmp0 += (src[c1] + src[c1 + _LAT_2C_]) * U[c1 * _LAT_C_ + c0].conj();
        tmp1 += (src[c1 + _LAT_1C_] + src[c1 + _LAT_3C_]) *
                U[c1 * _LAT_C_ + c0].conj();
      }
      dest[c0] += tmp0 * _;
      dest[c0 + _LAT_1C_] += tmp1 * _;
      dest[c0 + _LAT_2C_] += tmp0 * _;
      dest[c0 + _LAT_3C_] += tmp1 * _;
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
    LatticeComplex _(double((move == 1)), 0.0); // just inside
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        tmp0 += (src[c1] - src[c1 + _LAT_2C_]) * U[c0 * _LAT_C_ + c1];
        tmp1 +=
            (src[c1 + _LAT_1C_] - src[c1 + _LAT_3C_]) * U[c0 * _LAT_C_ + c1];
      }
      dest[c0] += tmp0 * _;
      dest[c0 + _LAT_1C_] += tmp1 * _;
      dest[c0 + _LAT_2C_] -= tmp0 * _;
      dest[c0 + _LAT_3C_] -= tmp1 * _;
    }
  }
}
#endif
give_ptr(origin_dest, dest, _LAT_SC_);
}
__global__ void wilson_dslash_x_send(void *device_U, void *device_src,
                                     void *device_xyztsc, int device_parity,
                                     void *device_b_x_send_vec,
                                     void *device_f_x_send_vec) {
#ifdef __X__
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  // int lat_x = xyztsc[_X_];
  int lat_x = 1; // so let x=0 first, then x = lat_x -1
  int lat_y = xyztsc[_Y_];
  int lat_z = xyztsc[_Z_];
  int lat_tzyxcc = xyztsc[_TZYXCC_];
  int move;
  move = lat_x * lat_y * lat_z;
  int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  int z = parity / move;
  parity -= z * move;
  int y = parity / lat_x;
  int x = parity - y * lat_x;
  parity = device_parity;
  int eo = (y + z + t) & 0x01; // (y+z+t)%2
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex src[_LAT_SC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_x_send_vec[_LAT_HALF_SC_];
  LatticeComplex f_x_send_vec[_LAT_HALF_SC_];
  LatticeComplex *origin_U;
  LatticeComplex *origin_src;
  LatticeComplex *origin_b_x_send_vec;
  LatticeComplex *origin_f_x_send_vec;
  {
    lat_x = xyztsc[_X_]; // give lat_size back
    x = 0;               // b_x
    origin_src = ((static_cast<LatticeComplex *>(device_src)) +
                  (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_b_x_send_vec =
        ((static_cast<LatticeComplex *>(device_b_x_send_vec)) +
         (((t * lat_z + z) * lat_y + y)) * _LAT_HALF_SC_);
  }
  { // x-1
    move_backward_x(move, x, lat_x, eo, parity);
    // even-odd
    // send in x+1 way
    give_ptr(src, origin_src, _LAT_SC_);
    {                                             // sigma src
      LatticeComplex _(double((move == 0)), 0.0); // just inside
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        b_x_send_vec[c1] = (src[c1] - src[c1 + _LAT_3C_] * I) * _;
        b_x_send_vec[c1 + _LAT_1C_] =
            (src[c1 + _LAT_1C_] - src[c1 + _LAT_2C_] * I) * _;
      }
      give_ptr(origin_b_x_send_vec, b_x_send_vec, _LAT_HALF_SC_);
    }
  }
  {
    x = lat_x - 1; // f_x
    origin_U = ((static_cast<LatticeComplex *>(device_U)) +
                (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_CC_);
    origin_src = ((static_cast<LatticeComplex *>(device_src)) +
                  (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_f_x_send_vec =
        ((static_cast<LatticeComplex *>(device_f_x_send_vec)) +
         (((t * lat_z + z) * lat_y + y)) * _LAT_HALF_SC_);
  }
  { // x+1
    move_forward_x(move, x, lat_x, eo, parity);
    // even-odd
    // send in x-1 way
    tmp_U = (origin_U + (1 - parity) * lat_tzyxcc); // even-odd
    give_u(U, tmp_U);
    give_ptr(src, origin_src, _LAT_SC_);
    {                                             // just tmp
      LatticeComplex _(double((move == 0)), 0.0); // just inside
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 +=
              (src[c1] + src[c1 + _LAT_3C_] * I) * U[c1 * _LAT_C_ + c0].conj();
          tmp1 += (src[c1 + _LAT_1C_] + src[c1 + _LAT_2C_] * I) *
                  U[c1 * _LAT_C_ + c0].conj();
        }
        f_x_send_vec[c0] = tmp0 * _;
        f_x_send_vec[c0 + _LAT_1C_] = tmp1 * _;
      }
      give_ptr(origin_f_x_send_vec, f_x_send_vec, _LAT_HALF_SC_);
    }
  }
#endif
}
__global__ void wilson_dslash_x_recv(void *device_U, void *device_dest,
                                     void *device_xyztsc, int device_parity,
                                     void *device_b_x_recv_vec,
                                     void *device_f_x_recv_vec) {
#ifdef __X__
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  // int lat_x = xyztsc[_X_];
  int lat_x = 1; // so let x=0 first, then x = lat_x -1
  int lat_y = xyztsc[_Y_];
  int lat_z = xyztsc[_Z_];
  int lat_tzyxcc = xyztsc[_TZYXCC_];
  int move;
  move = lat_x * lat_y * lat_z;
  int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  int z = parity / move;
  parity -= z * move;
  int y = parity / lat_x;
  int x = parity - y * lat_x;
  parity = device_parity;
  int eo = (y + z + t) & 0x01; // (y+z+t)%2
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_x_recv_vec[_LAT_HALF_SC_];
  LatticeComplex f_x_recv_vec[_LAT_HALF_SC_]; // needed
  LatticeComplex *origin_U;
  LatticeComplex *origin_dest;
  LatticeComplex *origin_b_x_recv_vec;
  LatticeComplex *origin_f_x_recv_vec;
  {
    lat_x = xyztsc[_X_]; // give lat_size back
    x = 0;               // b_x
    origin_dest = ((static_cast<LatticeComplex *>(device_dest)) +
                   (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_b_x_recv_vec =
        ((static_cast<LatticeComplex *>(device_b_x_recv_vec)) +
         (((t * lat_z + z) * lat_y + y)) * _LAT_HALF_SC_);
  }
  { // x-1
    move_backward_x(move, x, lat_x, eo, parity);
    // recv in x-1 way
    give_ptr(b_x_recv_vec, origin_b_x_recv_vec, _LAT_HALF_SC_);
    LatticeComplex _(double((move == lat_x - 1)), 0.0); // just inside
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      dest[c0] += b_x_recv_vec[c0];
      dest[c0 + _LAT_1C_] += b_x_recv_vec[c0 + _LAT_1C_];
      dest[c0 + _LAT_2C_] -= b_x_recv_vec[c0 + _LAT_1C_] * I;
      dest[c0 + _LAT_3C_] -= b_x_recv_vec[c0] * I;
    }
  }
  add_ptr(origin_dest, dest, _LAT_SC_); // even-odd
  for (int i = 0; i < _LAT_SC_; i++) {
    dest[i] = zero;
  }
  {
    x = lat_x - 1; // f_x
    origin_U = ((static_cast<LatticeComplex *>(device_U)) +
                (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_CC_);
    origin_dest = ((static_cast<LatticeComplex *>(device_dest)) +
                   (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_f_x_recv_vec =
        ((static_cast<LatticeComplex *>(device_f_x_recv_vec)) +
         (((t * lat_z + z) * lat_y + y)) * _LAT_HALF_SC_);
  }
  { // x+1
    move_forward_x(move, x, lat_x, eo, parity);
    // recv in x+1 way
    give_ptr(f_x_recv_vec, origin_f_x_recv_vec, _LAT_HALF_SC_);
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(U, tmp_U);
    {
      LatticeComplex _(double((move == 1 - lat_x)), 0.0); // just inside
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += f_x_recv_vec[c1] * U[c0 * _LAT_C_ + c1];
          tmp1 += f_x_recv_vec[c1 + _LAT_1C_] * U[c0 * _LAT_C_ + c1];
        }
        dest[c0] += tmp0;
        dest[c0 + _LAT_1C_] += tmp1;
        dest[c0 + _LAT_2C_] += tmp1 * I;
        dest[c0 + _LAT_3C_] += tmp0 * I;
      }
    }
  } // just add
  add_ptr(origin_dest, dest, _LAT_SC_);
#endif
}
__global__ void wilson_dslash_y_send(void *device_U, void *device_src,
                                     void *device_xyztsc, int device_parity,
                                     void *device_b_y_send_vec,
                                     void *device_f_y_send_vec) {
#ifdef __Y__
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  int lat_x = xyztsc[_X_];
  // int lat_y = yyztsc[_y_];
  int lat_y = 1; // so let y=0 first, then y = lat_y -1
  int lat_z = xyztsc[_Z_];
  int lat_tzyxcc = xyztsc[_TZYXCC_];
  int move;
  move = lat_x * lat_y * lat_z;
  int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  int z = parity / move;
  parity -= z * move;
  int y = parity / lat_x;
  int x = parity - y * lat_x;
  parity = device_parity;
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex src[_LAT_SC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_y_send_vec[_LAT_HALF_SC_];
  LatticeComplex f_y_send_vec[_LAT_HALF_SC_];
  LatticeComplex *origin_U;
  LatticeComplex *origin_src;
  LatticeComplex *origin_b_y_send_vec;
  LatticeComplex *origin_f_y_send_vec;
  {
    lat_y = xyztsc[_Y_]; // give lat_size back
    y = 0;               // b_y
    origin_src = ((static_cast<LatticeComplex *>(device_src)) +
                  (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_b_y_send_vec =
        ((static_cast<LatticeComplex *>(device_b_y_send_vec)) +
         (((t * lat_z + z)) * lat_x + x) * _LAT_HALF_SC_);
  }
  { // y-1
    // move_backward(move, y, lat_y);
    // send in y+1 way
    give_ptr(src, origin_src, _LAT_SC_);
    { // sigma src
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        b_y_send_vec[c1] = src[c1] + src[c1 + _LAT_3C_];
        b_y_send_vec[c1 + _LAT_1C_] = src[c1 + _LAT_1C_] - src[c1 + _LAT_2C_];
      }
      give_ptr(origin_b_y_send_vec, b_y_send_vec, _LAT_HALF_SC_);
    }
  }
  {
    y = lat_y - 1; // f_y
    origin_U = ((static_cast<LatticeComplex *>(device_U)) +
                (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_CC_);
    origin_src = ((static_cast<LatticeComplex *>(device_src)) +
                  (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_f_y_send_vec =
        ((static_cast<LatticeComplex *>(device_f_y_send_vec)) +
         (((t * lat_z + z)) * lat_x + x) * _LAT_HALF_SC_);
  }
  { // y+1
    // move_forward(move, y, lat_y);
    // send in y-1 way
    tmp_U = (origin_U + (3 - parity) * lat_tzyxcc); // even-odd
    give_u(U, tmp_U);
    give_ptr(src, origin_src, _LAT_SC_);
    { // just tmp
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += (src[c1] - src[c1 + _LAT_3C_]) * U[c1 * _LAT_C_ + c0].conj();
          tmp1 += (src[c1 + _LAT_1C_] + src[c1 + _LAT_2C_]) *
                  U[c1 * _LAT_C_ + c0].conj();
        }
        f_y_send_vec[c0] = tmp0;
        f_y_send_vec[c0 + _LAT_1C_] = tmp1;
      }
      give_ptr(origin_f_y_send_vec, f_y_send_vec, _LAT_HALF_SC_);
    }
  }
#endif
}
__global__ void wilson_dslash_y_recv(void *device_U, void *device_dest,
                                     void *device_xyztsc, int device_parity,
                                     void *device_b_y_recv_vec,
                                     void *device_f_y_recv_vec) {
#ifdef __Y__
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  int lat_x = xyztsc[_X_];
  // int lat_y = yyztsc[_y_];
  int lat_y = 1; // so let y=0 first, then y = lat_y -1
  int lat_z = xyztsc[_Z_];
  int lat_tzyxcc = xyztsc[_TZYXCC_];
  int move;
  move = lat_x * lat_y * lat_z;
  int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  int z = parity / move;
  parity -= z * move;
  int y = parity / lat_x;
  int x = parity - y * lat_x;
  parity = device_parity;
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_y_recv_vec[_LAT_HALF_SC_];
  LatticeComplex f_y_recv_vec[_LAT_HALF_SC_]; // needed
  LatticeComplex *origin_U;
  LatticeComplex *origin_dest;
  LatticeComplex *origin_b_y_recv_vec;
  LatticeComplex *origin_f_y_recv_vec;
  {
    lat_y = xyztsc[_Y_]; // give lat_size back
    y = 0;               // b_y
    origin_dest = ((static_cast<LatticeComplex *>(device_dest)) +
                   (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_b_y_recv_vec =
        ((static_cast<LatticeComplex *>(device_b_y_recv_vec)) +
         (((t * lat_z + z)) * lat_x + x) * _LAT_HALF_SC_);
  }
  { // y-1
    move_backward(move, y, lat_y);
    // recv in y-1 way
    give_ptr(b_y_recv_vec, origin_b_y_recv_vec, _LAT_HALF_SC_);
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      dest[c0] += b_y_recv_vec[c0];
      dest[c0 + _LAT_1C_] += b_y_recv_vec[c0 + _LAT_1C_];
      dest[c0 + _LAT_2C_] += b_y_recv_vec[c0 + _LAT_1C_];
      dest[c0 + _LAT_3C_] -= b_y_recv_vec[c0];
    }
  }
  // just add
  add_ptr(origin_dest, dest, _LAT_SC_);
  for (int i = 0; i < _LAT_SC_; i++) {
    dest[i] = zero;
  }
  {
    y = lat_y - 1; // f_y
    origin_U = ((static_cast<LatticeComplex *>(device_U)) +
                (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_CC_);
    origin_dest = ((static_cast<LatticeComplex *>(device_dest)) +
                   (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_f_y_recv_vec =
        ((static_cast<LatticeComplex *>(device_f_y_recv_vec)) +
         (((t * lat_z + z)) * lat_x + x) * _LAT_HALF_SC_);
  }
  { // y+1
    // move_forward(move, y, lat_y);
    // recv in y+1 way
    give_ptr(f_y_recv_vec, origin_f_y_recv_vec, _LAT_HALF_SC_);
    tmp_U = (origin_U + (2 + parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    {
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += f_y_recv_vec[c1] * U[c0 * _LAT_C_ + c1];
          tmp1 += f_y_recv_vec[c1 + _LAT_1C_] * U[c0 * _LAT_C_ + c1];
        }
        dest[c0] += tmp0;
        dest[c0 + _LAT_1C_] += tmp1;
        dest[c0 + _LAT_2C_] -= tmp1;
        dest[c0 + _LAT_3C_] += tmp0;
      }
    }
  } // just add
  add_ptr(origin_dest, dest, _LAT_SC_);
#endif
}
__global__ void wilson_dslash_z_send(void *device_U, void *device_src,
                                     void *device_xyztsc, int device_parity,
                                     void *device_b_z_send_vec,
                                     void *device_f_z_send_vec) {
#ifdef __Z__
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  int lat_x = xyztsc[_X_];
  int lat_y = xyztsc[_Y_];
  // int lat_z = zzztsc[_z_];
  int lat_z = 1; // so let z=0 first, then z = lat_z -1
  int lat_tzyxcc = xyztsc[_TZYXCC_];
  int move;
  move = lat_x * lat_y * lat_z;
  int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  int z = parity / move;
  parity -= z * move;
  int y = parity / lat_x;
  int x = parity - y * lat_x;
  parity = device_parity;
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex src[_LAT_SC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_z_send_vec[_LAT_HALF_SC_];
  LatticeComplex f_z_send_vec[_LAT_HALF_SC_];
  LatticeComplex *origin_U;
  LatticeComplex *origin_src;
  LatticeComplex *origin_b_z_send_vec;
  LatticeComplex *origin_f_z_send_vec;
  {
    lat_z = xyztsc[_Z_]; // give lat_size back
    z = 0;               // b_z
    origin_src = ((static_cast<LatticeComplex *>(device_src)) +
                  (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_b_z_send_vec =
        ((static_cast<LatticeComplex *>(device_b_z_send_vec)) +
         (((t)*lat_y + y) * lat_x + x) * _LAT_HALF_SC_);
  }
  { // z-1
    // move_backward(move, z, lat_z);
    // send in z+1 way
    give_ptr(src, origin_src, _LAT_SC_);
    { // sigma src
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        b_z_send_vec[c1] = src[c1] - src[c1 + _LAT_2C_] * I;
        b_z_send_vec[c1 + _LAT_1C_] =
            src[c1 + _LAT_1C_] + src[c1 + _LAT_3C_] * I;
      }
      give_ptr(origin_b_z_send_vec, b_z_send_vec, _LAT_HALF_SC_);
    }
  }
  {
    z = lat_z - 1; // f_z
    origin_U = ((static_cast<LatticeComplex *>(device_U)) +
                (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_CC_);
    origin_src = ((static_cast<LatticeComplex *>(device_src)) +
                  (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_f_z_send_vec =
        ((static_cast<LatticeComplex *>(device_f_z_send_vec)) +
         (((t)*lat_y + y) * lat_x + x) * _LAT_HALF_SC_);
  }
  { // z+1
    // move_forward(move, z, lat_z);
    // send in z-1 way
    tmp_U = (origin_U + (5 - parity) * lat_tzyxcc); // even-odd
    give_u(U, tmp_U);
    give_ptr(src, origin_src, _LAT_SC_);
    { // just tmp
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 +=
              (src[c1] + src[c1 + _LAT_2C_] * I) * U[c1 * _LAT_C_ + c0].conj();
          tmp1 += (src[c1 + _LAT_1C_] - src[c1 + _LAT_3C_] * I) *
                  U[c1 * _LAT_C_ + c0].conj();
        }
        f_z_send_vec[c0] = tmp0;
        f_z_send_vec[c0 + _LAT_1C_] = tmp1;
      }
      give_ptr(origin_f_z_send_vec, f_z_send_vec, _LAT_HALF_SC_);
    }
  }
#endif
}
__global__ void wilson_dslash_z_recv(void *device_U, void *device_dest,
                                     void *device_xyztsc, int device_parity,
                                     void *device_b_z_recv_vec,
                                     void *device_f_z_recv_vec) {
#ifdef __Z__
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  int lat_x = xyztsc[_X_];
  int lat_y = xyztsc[_Y_];
  // int lat_z = zzztsc[_z_];
  int lat_z = 1; // so let z=0 first, then z = lat_z -1
  int lat_tzyxcc = xyztsc[_TZYXCC_];
  int move;
  move = lat_x * lat_y * lat_z;
  int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  int z = parity / move;
  parity -= z * move;
  int y = parity / lat_x;
  int x = parity - y * lat_x;
  parity = device_parity;
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_z_recv_vec[_LAT_HALF_SC_];
  LatticeComplex f_z_recv_vec[_LAT_HALF_SC_]; // needed
  LatticeComplex *origin_U;
  LatticeComplex *origin_dest;
  LatticeComplex *origin_b_z_recv_vec;
  LatticeComplex *origin_f_z_recv_vec;
  {
    lat_z = xyztsc[_Z_]; // give lat_size back
    z = 0;               // b_z
    origin_dest = ((static_cast<LatticeComplex *>(device_dest)) +
                   (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_b_z_recv_vec =
        ((static_cast<LatticeComplex *>(device_b_z_recv_vec)) +
         (((t)*lat_y + y) * lat_x + x) * _LAT_HALF_SC_);
  }
  { // z-1
    // move_backward(move, z, lat_z);
    // recv in z-1 way
    give_ptr(b_z_recv_vec, origin_b_z_recv_vec, _LAT_HALF_SC_);
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      dest[c0] += b_z_recv_vec[c0];
      dest[c0 + _LAT_1C_] += b_z_recv_vec[c0 + _LAT_1C_];
      dest[c0 + _LAT_2C_] -= b_z_recv_vec[c0] * I;
      dest[c0 + _LAT_3C_] += b_z_recv_vec[c0 + _LAT_1C_] * I;
    }
  }
  // just add
  add_ptr(origin_dest, dest, _LAT_SC_);
  for (int i = 0; i < _LAT_SC_; i++) {
    dest[i] = zero;
  }
  {
    z = lat_z - 1; // f_z
    origin_U = ((static_cast<LatticeComplex *>(device_U)) +
                (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_CC_);
    origin_dest = ((static_cast<LatticeComplex *>(device_dest)) +
                   (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_f_z_recv_vec =
        ((static_cast<LatticeComplex *>(device_f_z_recv_vec)) +
         (((t)*lat_y + y) * lat_x + x) * _LAT_HALF_SC_);
  }
  { // z+1
    // move_forward(move, z, lat_z);
    // recv in z+1 way
    give_ptr(f_z_recv_vec, origin_f_z_recv_vec, _LAT_HALF_SC_);
    tmp_U = (origin_U + (4 + parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    {
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += f_z_recv_vec[c1] * U[c0 * _LAT_C_ + c1];
          tmp1 += f_z_recv_vec[c1 + _LAT_1C_] * U[c0 * _LAT_C_ + c1];
        }
        dest[c0] += tmp0;
        dest[c0 + _LAT_1C_] += tmp1;
        dest[c0 + _LAT_2C_] += tmp0 * I;
        dest[c0 + _LAT_3C_] -= tmp1 * I;
      }
    }
  } // just add
  add_ptr(origin_dest, dest, _LAT_SC_);
#endif
}
__global__ void wilson_dslash_t_send(void *device_U, void *device_src,
                                     void *device_xyztsc, int device_parity,
                                     void *device_b_t_send_vec,
                                     void *device_f_t_send_vec) {
#ifdef __T__
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  int lat_x = xyztsc[_X_];
  int lat_y = xyztsc[_Y_];
  int lat_z = xyztsc[_Z_];
  // int lat_t = ttttsc[_t_];
  int lat_t = 1; // so let t=0 first, then t = lat_t -1
  int lat_tzyxcc = xyztsc[_TZYXCC_];
  int move;
  move = lat_x * lat_y * lat_z;
  int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  int z = parity / move;
  parity -= z * move;
  int y = parity / lat_x;
  int x = parity - y * lat_x;
  parity = device_parity;
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex src[_LAT_SC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_t_send_vec[_LAT_HALF_SC_];
  LatticeComplex f_t_send_vec[_LAT_HALF_SC_];
  LatticeComplex *origin_U;
  LatticeComplex *origin_src;
  LatticeComplex *origin_b_t_send_vec;
  LatticeComplex *origin_f_t_send_vec;
  {
    lat_t = xyztsc[_T_]; // give lat_size back
    t = 0;               // b_t
    origin_src = ((static_cast<LatticeComplex *>(device_src)) +
                  (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_b_t_send_vec =
        ((static_cast<LatticeComplex *>(device_b_t_send_vec)) +
         (((z)*lat_y + y) * lat_x + x) * _LAT_HALF_SC_);
  }
  { // t-1
    // move_backward(move, t, lat_t);
    // send in t+1 way
    give_ptr(src, origin_src, _LAT_SC_);
    { // sigma src
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        b_t_send_vec[c1] = src[c1] - src[c1 + _LAT_2C_];
        b_t_send_vec[c1 + _LAT_1C_] = src[c1 + _LAT_1C_] - src[c1 + _LAT_3C_];
      }
      give_ptr(origin_b_t_send_vec, b_t_send_vec, _LAT_HALF_SC_);
    }
  }
  {
    t = lat_t - 1; // f_t
    origin_U = ((static_cast<LatticeComplex *>(device_U)) +
                (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_CC_);
    origin_src = ((static_cast<LatticeComplex *>(device_src)) +
                  (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_f_t_send_vec =
        ((static_cast<LatticeComplex *>(device_f_t_send_vec)) +
         (((z)*lat_y + y) * lat_x + x) * _LAT_HALF_SC_);
  }
  { // t+1
    // move_forward(move, t, lat_t);
    // send in t-1 way
    tmp_U = (origin_U + (7 - parity) * lat_tzyxcc); // even-odd
    give_u(U, tmp_U);
    give_ptr(src, origin_src, _LAT_SC_);
    { // just tmp
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += (src[c1] + src[c1 + _LAT_2C_]) * U[c1 * _LAT_C_ + c0].conj();
          tmp1 += (src[c1 + _LAT_1C_] + src[c1 + _LAT_3C_]) *
                  U[c1 * _LAT_C_ + c0].conj();
        }
        f_t_send_vec[c0] = tmp0;
        f_t_send_vec[c0 + _LAT_1C_] = tmp1;
      }
      give_ptr(origin_f_t_send_vec, f_t_send_vec, _LAT_HALF_SC_);
    }
  }
#endif
}
__global__ void wilson_dslash_t_recv(void *device_U, void *device_dest,
                                     void *device_xyztsc, int device_parity,
                                     void *device_b_t_recv_vec,
                                     void *device_f_t_recv_vec) {
#ifdef __T__
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  int *xyztsc = static_cast<int *>(device_xyztsc);
  int lat_x = xyztsc[_X_];
  int lat_y = xyztsc[_Y_];
  int lat_z = xyztsc[_Z_];
  // int lat_t = ttttsc[_t_];
  int lat_t = 1; // so let t=0 first, then t = lat_t -1
  int lat_tzyxcc = xyztsc[_TZYXCC_];
  int move;
  move = lat_x * lat_y * lat_z;
  int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  int z = parity / move;
  parity -= z * move;
  int y = parity / lat_x;
  int x = parity - y * lat_x;
  parity = device_parity;
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *tmp_U;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[_LAT_CC_];
  LatticeComplex dest[_LAT_SC_];
  LatticeComplex b_t_recv_vec[_LAT_HALF_SC_];
  LatticeComplex f_t_recv_vec[_LAT_HALF_SC_]; // needed
  LatticeComplex *origin_U;
  LatticeComplex *origin_dest;
  LatticeComplex *origin_b_t_recv_vec;
  LatticeComplex *origin_f_t_recv_vec;
  {
    lat_t = xyztsc[_T_]; // give lat_size back
    t = 0;               // b_t
    origin_dest = ((static_cast<LatticeComplex *>(device_dest)) +
                   (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_b_t_recv_vec =
        ((static_cast<LatticeComplex *>(device_b_t_recv_vec)) +
         (((z)*lat_y + y) * lat_x + x) * _LAT_HALF_SC_);
  }
  { // t-1
    // move_backward(move, t, lat_t);
    // recv in t-1 way
    give_ptr(b_t_recv_vec, origin_b_t_recv_vec, _LAT_HALF_SC_);
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      dest[c0] += b_t_recv_vec[c0];
      dest[c0 + _LAT_1C_] += b_t_recv_vec[c0 + _LAT_1C_];
      dest[c0 + _LAT_2C_] += b_t_recv_vec[c0];
      dest[c0 + _LAT_3C_] += b_t_recv_vec[c0 + _LAT_1C_];
    }
  }
  // just add
  add_ptr(origin_dest, dest, _LAT_SC_);
  for (int i = 0; i < _LAT_SC_; i++) {
    dest[i] = zero;
  }
  {
    t = lat_t - 1; // f_t
    origin_U = ((static_cast<LatticeComplex *>(device_U)) +
                (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_CC_);
    origin_dest = ((static_cast<LatticeComplex *>(device_dest)) +
                   (((t * lat_z + z) * lat_y + y) * lat_x + x) * _LAT_SC_);
    origin_f_t_recv_vec =
        ((static_cast<LatticeComplex *>(device_f_t_recv_vec)) +
         (((z)*lat_y + y) * lat_x + x) * _LAT_HALF_SC_);
  }
  { // t+1
    // move_forward(move, t, lat_t);
    // recv in t+1 way
    give_ptr(f_t_recv_vec, origin_f_t_recv_vec, _LAT_HALF_SC_);
    tmp_U = (origin_U + (6 + parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    {
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += f_t_recv_vec[c1] * U[c0 * _LAT_C_ + c1];
          tmp1 += f_t_recv_vec[c1 + _LAT_1C_] * U[c0 * _LAT_C_ + c1];
        }
        dest[c0] += tmp0;
        dest[c0 + _LAT_1C_] += tmp1;
        dest[c0 + _LAT_2C_] -= tmp0;
        dest[c0 + _LAT_3C_] -= tmp1;
      }
    }
  } // just add
  add_ptr(origin_dest, dest, _LAT_SC_);
#endif
}
#endif