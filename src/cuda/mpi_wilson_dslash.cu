#pragma optimize(5)
#include "../../include/qcu_cuda.h"

__global__ void mpi_wilson_dslash(
    void *device_U, void *device_src, void *device_dest, int device_lat_x,
    const int device_lat_y, const int device_lat_z, const int device_lat_t,
    const int device_parity, const int device_node_rank, int device_grid_x, const int device_grid_y,
    const int device_grid_z, const int device_grid_t) {
  register int parity = blockIdx.x * blockDim.x + threadIdx.x;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  const int node_rank = device_node_rank;
  const int grid_x = device_grid_x;
  const int grid_y = device_grid_y;
  const int grid_z = device_grid_z;
  const int grid_t = device_grid_t;
  const int lat_xcc = lat_x * 9;
  const int lat_yxcc = lat_y * lat_xcc;
  const int lat_zyxcc = lat_z * lat_yxcc;
  const int lat_tzyxcc = lat_t * lat_zyxcc;
  const int lat_xsc = lat_x * 12;
  const int lat_yxsc = lat_y * lat_xsc;
  const int lat_zyxsc = lat_z * lat_yxsc;
  // [rank // Gt // Gz // Gy, rank // Gt // Gz % Gy, rank // Gt % Gz, rank % Gt]
  const int grid_index_x = node_rank / lat_t / lat_z / lat_y;
  const int grid_index_y = node_rank / lat_t / lat_z % lat_y;
  const int grid_index_z = node_rank / lat_t % lat_z;
  const int grid_index_t = node_rank % lat_t;
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
  register LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  register LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  register LatticeComplex *tmp_U;
  register LatticeComplex *tmp_src;
  register LatticeComplex tmp0(0.0, 0.0);
  register LatticeComplex tmp1(0.0, 0.0);
  register LatticeComplex U[9];
  register LatticeComplex src[12];
  register LatticeComplex dest[12];
  register LatticeComplex send_vec[6];
  register LatticeComplex recv_vec[6];
  MPI_Request send_request[8];
  MPI_Request recv_request[8];
  // just wilson(Sum part)
  give_value(dest, zero, 12);
  if (grid_x == 1) {
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
  } else {
    {
      // x-1
      move_backward_x(move, x, lat_x, eo, parity);
      if (move == 0 || move == -1) {
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
      } else {
        // {
        //   tmp_U = (origin_U + parity * lat_tzyxcc);
        //   give_u(U, tmp_U);
        //   tmp_src = (origin_src + move * 12);
        //   give_ptr(src, tmp_src, 12);
        //   {
        //     for (int c0 = 0; c0 < 3; c0++) {
        //       tmp0 = zero;
        //       tmp1 = zero;
        //       for (int c1 = 0; c1 < 3; c1++) {
        //         tmp0 += (src[c1] - src[c1 + 9] * I) * U[c0 * 3 + c1];
        //         tmp1 += (src[c1 + 3] - src[c1 + 6] * I) * U[c0 * 3 + c1];
        //       }
        //       dest[c0] += tmp0;
        //       dest[c0 + 3] += tmp1;
        //       dest[c0 + 6] += tmp1 * I;
        //       dest[c0 + 9] += tmp0 * I;
        //     }
        //   }
        // }
        // MPI_Isend(send_vec, 12, MPI_DOUBLE, forward_rank, forward_rank,
        //           MPI_COMM_WORLD, &send_request[0]);
        // MPI_Wait(&send_request[0], MPI_STATUS_IGNORE);
        // MPI_Barrier(MPI_COMM_WORLD);
        // MPI_Irecv(recv_vec, 12, MPI_DOUBLE, forward_rank, node_rank,
        //           MPI_COMM_WORLD, &recv_request[0]);
        // MPI_Wait(&recv_request[0], MPI_STATUS_IGNORE);
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
    }
    {
      // x+1
      move_forward_x(move, x, lat_x, eo, parity);
      if (move == 0 || move == 1) {
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
      } else {
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
    }
  }
  if (grid_y == 1) {
    { // y-1
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
  } else {
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
      }
    }
  }
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
      }
    }
  }
  if (grid_t == 1) {
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
  } else {
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
      }
    }
  }
  give_ptr(origin_dest, dest, 12);
}