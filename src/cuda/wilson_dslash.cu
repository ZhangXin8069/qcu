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
  give_value(dest, zero, 12);
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

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge,
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
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    // printf(
    //     "wilson dslash total time: (without malloc free memcpy) : %.9lf
    //     sec\n", double(duration) / 1e9);
  }
  {
    // free
    checkCudaErrors(cudaFree(clover));
  }
}
#endif
