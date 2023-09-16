__global__ void
wilson_dslash_z_send(void *device_U, void *device_src, void *device_dest,
                     int device_lat_x, const int device_lat_y,
                     const int device_lat_z, const int device_lat_t,
                     const int device_parity, const int device_grid_z,
                     void *device_b_z_send_vec, void *device_f_z_send_vec) {
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
  const int grid_z = device_grid_z;
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
  register LatticeComplex I(0.0, 1.0);
  register LatticeComplex zero(0.0, 0.0);
  register LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * 9);
  register LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  register LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  register LatticeComplex *origin_b_z_send_vec =
      ((static_cast<LatticeComplex *>(device_b_z_send_vec)) +
       (t * lat_y * lat_x + y * lat_x + x) * 6);
  register LatticeComplex *origin_f_z_send_vec =
      ((static_cast<LatticeComplex *>(device_f_z_send_vec)) +
       (t * lat_y * lat_x + y * lat_x + x) * 6);
  register LatticeComplex *tmp_U;
  register LatticeComplex *tmp_src;
  register LatticeComplex tmp0(0.0, 0.0);
  register LatticeComplex tmp1(0.0, 0.0);
  register LatticeComplex U[9];
  register LatticeComplex src[12];
  register LatticeComplex dest[12];
  register LatticeComplex *b_z_send_vec[6];
  register LatticeComplex *f_z_send_vec[6];
  give_value(dest, zero, 12);
  if (grid_z == 1) {
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
  } else {
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
        tmp_U = (origin_U + parity * lat_tzyxcc);
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
  }
}

__global__ void
wilson_dslash_z_recv(void *device_U, void *device_dest, int device_lat_x,
                     const int device_lat_y, const int device_lat_z,
                     const int device_lat_t, const int device_parity,
                     const int device_grid_z, void *device_b_z_recv_vec,
                     void *device_f_z_recv_vec) {
  register int parity = blockIdx.x * blockDim.x + threadIdx.x;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  const int grid_z = device_grid_z;
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
  register LatticeComplex I(0.0, 1.0);
  register LatticeComplex zero(0.0, 0.0);
  register LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * 9);
  register LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  register LatticeComplex *origin_b_z_recv_vec =
      ((static_cast<LatticeComplex *>(device_b_z_recv_vec)) +
       (t * lat_y * lat_x + y * lat_x + x) * 6);
  register LatticeComplex *origin_f_z_recv_vec =
      ((static_cast<LatticeComplex *>(device_f_z_recv_vec)) +
       (t * lat_y * lat_x + y * lat_x + x) * 6);
  register LatticeComplex *tmp_U;
  register LatticeComplex tmp0(0.0, 0.0);
  register LatticeComplex tmp1(0.0, 0.0);
  register LatticeComplex U[9];
  register LatticeComplex dest[12];
  register LatticeComplex *b_z_recv_vec[6];
  register LatticeComplex *f_z_recv_vec[6];
  if (grid_z != 1) {
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
        tmp_U = (origin_U + parity * lat_t * lat_z * lat_y * lat_x * 9);
        give_u(U, tmp_U);
        {
          for (int c0 = 0; c0 < 3; c0++) {
            tmp0 = zero;
            tmp1 = zero;
            for (int c1 = 0; c1 < 3; c1++) {
              tmp0 += f_z_recv_vec[c0] * U[c0 * 3 + c1];
              tmp1 += f_z_recv_vec[c0 + 3] * U[c0 * 3 + c1];
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
}