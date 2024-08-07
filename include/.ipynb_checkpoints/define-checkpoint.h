#ifndef _DEFINE_H
#define _DEFINE_H

#include "./qcu.h"
#define BLOCK_SIZE 256
#define X 0
#define Y 1
#define Z 2
#define T 3
#define _DIM_ 4
#define B_X 0
#define F_X 1
#define B_Y 2
#define F_Y 3
#define B_Z 4
#define F_Z 5
#define B_T 6
#define F_T 7
#define _WARDS_ 8
#define YZT 0
#define XZT 1
#define XYT 2
#define XYZ 3
#define EVEN 0
#define ODD 1
#define EVENODD 2
#define LAT_C 3
#define LAT_S 4
#define _LAT_D_ 4
#define _B_ 0
#define _F_ 1
#define _BF_ 2
#define _OUTPUT_SIZE_ 10
#define _BACKWARD_ -1
#define _NOWARD_ 0
#define _FORWARD_ 1
#define _SR_ 2
#define _LAT_EXAMPLE_ 16
#define _GRID_EXAMPLE_ 1

#define WILSON_DSLASH
#define CLOVER_DSLASH
// #define OVERLAP_DSLASH
#define MPI_WILSON_DSLASH
// #define MPI_CLOVER_DSLASH
// #define MPI_OVERLAP_DSLASH
// #define TEST_WILSON_DSLASH
// #define TEST_CLOVER_DSLASH
// #define TEST_OVERLAP_DSLASH
// #define WILSON_BISTABCG
// #define CLOVER_BISTABCG
// #define OVERLAP_BISTABCG
#define MPI_WILSON_BISTABCG
// #define MPI_CLOVER_BISTABCG
// #define MPI_OVERLAP_BISTABCG
// #define TEST_WILSON_BISTABCG
// #define TEST_CLOVER_BISTABCG
// #define TEST_OVERLAP_BISTABCG
// #define WILSON_MULTGRID
// #define CLOVER_MULTGRID
// #define OVERLAP_MULTGRID
// #define MPI_WILSON_MULTGRID
// #define MPI_CLOVER_MULTGRID
// #define MPI_OVERLAP_MULTGRID
// #define TEST_WILSON_MULTGRID
// #define TEST_CLOVER_MULTGRID
// #define TEST_OVERLAP_MULTGRID

#define checkCudaErrors(err)                                                   \
  {                                                                            \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr,                                                          \
              "checkCudaErrors() API error = %04d \"%s\" from file <%s>, "     \
              "line %i.\n",                                                    \
              err, cudaGetErrorString(err), __FILE__, __LINE__);               \
      exit(-1);                                                                \
    }                                                                          \
  }

// little strange, but don't want change
#define give_value(U, zero, n)                                                 \
  {                                                                            \
    LatticeComplex *tmp_U = static_cast<LatticeComplex *>(U);                  \
    for (int i = 0; i < n; i++) {                                              \
      tmp_U[i] = zero;                                                         \
    }                                                                          \
  }

#define give_ptr(U, origin_U, n)                                               \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] = origin_U[i];                                                      \
    }                                                                          \
  }

#define give_rand(input_matrix, size)                                          \
  {                                                                            \
    for (int i = 0; i < size; i++) {                                           \
      input_matrix[i].real = static_cast<double>(rand()) / RAND_MAX;           \
      input_matrix[i].imag = static_cast<double>(rand()) / RAND_MAX;           \
    }                                                                          \
  }

#define give_u(tmp, tmp_U)                                                     \
  {                                                                            \
    for (int i = 0; i < 6; i++) {                                              \
      tmp[i] = tmp_U[i];                                                       \
    }                                                                          \
    tmp[6] = (tmp[1] * tmp[5] - tmp[2] * tmp[4]).conj();                       \
    tmp[7] = (tmp[2] * tmp[3] - tmp[0] * tmp[5]).conj();                       \
    tmp[8] = (tmp[0] * tmp[4] - tmp[1] * tmp[3]).conj();                       \
  }

#define add_value(U, tmp, n)                                                   \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] += tmp;                                                             \
    }                                                                          \
  }

#define subt_value(U, tmp, n)                                                  \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] -= tmp;                                                             \
    }                                                                          \
  }

#define mult_value(U, tmp, n)                                                  \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] *= tmp;                                                             \
    }                                                                          \
  }

#define divi_value(U, tmp, n)                                                  \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] /= tmp;                                                             \
    }                                                                          \
  }

#define add_ptr(U, tmp, n)                                                     \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] += tmp[i];                                                          \
    }                                                                          \
  }

#define subt_ptr(U, tmp, n)                                                    \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] -= tmp[i];                                                          \
    }                                                                          \
  }

#define mult_ptr(U, tmp, n)                                                    \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] *= tmp[i];                                                          \
    }                                                                          \
  }

#define divi_ptr(U, tmp, n)                                                    \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] /= tmp[i];                                                          \
    }                                                                          \
  }

#define mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero)                         \
  {                                                                            \
    for (int c0 = 0; c0 < 3; c0++) {                                           \
      for (int c1 = 0; c1 < 3; c1++) {                                         \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < 3; cc++) {                                       \
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[cc * 3 + c1];                       \
        }                                                                      \
        tmp3[c0 * 3 + c1] = tmp0;                                              \
      }                                                                        \
    }                                                                          \
  }

#define mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero)                          \
  {                                                                            \
    for (int c0 = 0; c0 < 3; c0++) {                                           \
      for (int c1 = 0; c1 < 3; c1++) {                                         \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < 3; cc++) {                                       \
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[c1 * 3 + cc].conj();                \
        }                                                                      \
        tmp3[c0 * 3 + c1] = tmp0;                                              \
      }                                                                        \
    }                                                                          \
  }

#define mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero)                          \
  {                                                                            \
    for (int c0 = 0; c0 < 3; c0++) {                                           \
      for (int c1 = 0; c1 < 3; c1++) {                                         \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < 3; cc++) {                                       \
          tmp0 += tmp1[cc * 3 + c0].conj() * tmp2[cc * 3 + c1];                \
        }                                                                      \
        tmp3[c0 * 3 + c1] = tmp0;                                              \
      }                                                                        \
    }                                                                          \
  }

#define mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero)                           \
  {                                                                            \
    for (int c0 = 0; c0 < 3; c0++) {                                           \
      for (int c1 = 0; c1 < 3; c1++) {                                         \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < 3; cc++) {                                       \
          tmp0 += tmp1[cc * 3 + c0].conj() * tmp2[c1 * 3 + cc].conj();         \
        }                                                                      \
        tmp3[c0 * 3 + c1] = tmp0;                                              \
      }                                                                        \
    }                                                                          \
  }

#define inverse(input_matrix, inverse_matrix, augmented_matrix, pivot, factor, \
                size)                                                          \
  {                                                                            \
    for (int i = 0; i < size; i++) {                                           \
      for (int j = 0; j < size; j++) {                                         \
        inverse_matrix[i * size + j] = input_matrix[i * size + j];             \
        augmented_matrix[i * 2 * size + j] = inverse_matrix[i * size + j];     \
      }                                                                        \
      augmented_matrix[i * 2 * size + size + i] = 1.0;                         \
    }                                                                          \
    for (int i = 0; i < size; i++) {                                           \
      pivot = augmented_matrix[i * 2 * size + i];                              \
      for (int j = 0; j < 2 * size; j++) {                                     \
        augmented_matrix[i * 2 * size + j] /= pivot;                           \
      }                                                                        \
      for (int j = 0; j < size; j++) {                                         \
        if (j != i) {                                                          \
          factor = augmented_matrix[j * 2 * size + i];                         \
          for (int k = 0; k < 2 * size; ++k) {                                 \
            augmented_matrix[j * 2 * size + k] -=                              \
                factor * augmented_matrix[i * 2 * size + k];                   \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    for (int i = 0; i < size; i++) {                                           \
      for (int j = 0; j < size; j++) {                                         \
        inverse_matrix[i * size + j] =                                         \
            augmented_matrix[i * 2 * size + size + j];                         \
      }                                                                        \
    }                                                                          \
  }

#define print_gauge(input_gauge)                                               \
  {                                                                            \
    printf("############\n");                                                  \
    printf("[");                                                               \
    printf("[%.9lf,%.9lf]", input_gauge[0].real, input_gauge[0].imag);         \
    printf("[%.9lf,%.9lf]", input_gauge[1].real, input_gauge[1].imag);         \
    printf("[%.9lf,%.9lf]", input_gauge[2].real, input_gauge[2].imag);         \
    printf("]\n");                                                             \
    printf("[");                                                               \
    printf("[%.9lf,%.9lf]", input_gauge[3].real, input_gauge[3].imag);         \
    printf("[%.9lf,%.9lf]", input_gauge[4].real, input_gauge[4].imag);         \
    printf("[%.9lf,%.9lf]", input_gauge[5].real, input_gauge[5].imag);         \
    printf("]\n");                                                             \
    printf("[");                                                               \
    printf("[%.9lf,%.9lf]", input_gauge[6].real, input_gauge[6].imag);         \
    printf("[%.9lf,%.9lf]", input_gauge[7].real, input_gauge[7].imag);         \
    printf("[%.9lf,%.9lf]", input_gauge[8].real, input_gauge[8].imag);         \
    printf("]\n");                                                             \
  }

#define print_tmp(input_tmp, n)                                                \
  {                                                                            \
    register LatticeComplex *tmp_vec =                                         \
        static_cast<LatticeComplex *>(input_tmp);                              \
    printf("############\n");                                                  \
    for (int i = 0; i < n; i++) {                                              \
      printf("[%.9lf,%.9lf]\n", tmp_vec[i].real, tmp_vec[i].imag);             \
    }                                                                          \
  }

#define print_fermi(input_fermi)                                               \
  {                                                                            \
    int tmp;                                                                   \
    for (int s = 0; s < 4; s++) {                                              \
      printf("######S%.1d######\n", s);                                        \
      tmp = s * 9;                                                             \
      printf("[");                                                             \
      printf("[%.9lf,%.9lf]", input_fermi[tmp].real, input_fermi[tmp].imag);   \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 1].real,                       \
             input_fermi[tmp + 1].imag);                                       \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 2].real,                       \
             input_fermi[tmp + 2].imag);                                       \
      printf("]\n");                                                           \
      printf("[");                                                             \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 3].real,                       \
             input_fermi[tmp + 3].imag);                                       \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 4].real,                       \
             input_fermi[tmp + 4].imag);                                       \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 5].real,                       \
             input_fermi[tmp + 5].imag);                                       \
      printf("]\n");                                                           \
      printf("[");                                                             \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 6].real,                       \
             input_fermi[tmp + 6].imag);                                       \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 7].real,                       \
             input_fermi[tmp + 7].imag);                                       \
      printf("[%.9lf,%.9lf]", input_fermi[tmp + 8].real,                       \
             input_fermi[tmp + 8].imag);                                       \
      printf("]\n");                                                           \
    }                                                                          \
  }

#define move_backward(move, y, lat_y)                                          \
  { move = -1 + (y == 0) * lat_y; }

#define move_forward(move, y, lat_y)                                           \
  { move = 1 - (y == lat_y - 1) * lat_y; }

#define move_backward_x(move, x, lat_x, eo, parity)                            \
  { move = (-1 + (x == 0) * lat_x) * (eo == parity); }

#define move_forward_x(move, x, lat_x, eo, parity)                             \
  { move = (1 - (x == lat_x - 1) * lat_x) * (eo != parity); }

#define give_dims(param, lat_1dim, lat_3dim, lat_4dim)                         \
  {                                                                            \
    lat_1dim[X] = param->lattice_size[X] >> 1;                                 \
    lat_1dim[Y] = param->lattice_size[Y];                                      \
    lat_1dim[Z] = param->lattice_size[Z];                                      \
    lat_1dim[T] = param->lattice_size[T];                                      \
    lat_3dim[YZT] = lat_1dim[Y] * lat_1dim[Z] * lat_1dim[T];                   \
    lat_3dim[XZT] = lat_1dim[X] * lat_1dim[Z] * lat_1dim[T];                   \
    lat_3dim[XYT] = lat_1dim[X] * lat_1dim[Y] * lat_1dim[T];                   \
    lat_3dim[XYZ] = lat_1dim[X] * lat_1dim[Y] * lat_1dim[Z];                   \
    lat_4dim = lat_3dim[XYZ] * lat_1dim[T];                                    \
  }

#define give_grid(grid, node_rank, grid_1dim, grid_index_1dim)                 \
  {                                                                            \
    MPI_Comm_rank(MPI_COMM_WORLD, &node_rank);                                 \
    grid_1dim[X] = grid->lattice_size[X];                                      \
    grid_1dim[Y] = grid->lattice_size[Y];                                      \
    grid_1dim[Z] = grid->lattice_size[Z];                                      \
    grid_1dim[T] = grid->lattice_size[T];                                      \
    grid_index_1dim[X] =                                                       \
        node_rank / grid_1dim[T] / grid_1dim[Z] / grid_1dim[Y];                \
    grid_index_1dim[Y] =                                                       \
        node_rank / grid_1dim[T] / grid_1dim[Z] % grid_1dim[Y];                \
    grid_index_1dim[Z] = node_rank / grid_1dim[T] % grid_1dim[Z];              \
    grid_index_1dim[T] = node_rank % grid_1dim[T];                             \
  }

#define zero_vec(lat_3dim6, send_vec, recv_vec, zero)                          \
  {                                                                            \
    for (int i = 0; i < _DIM_; i++) {                                            \
      give_value(send_vec[i * _SR_], zero, lat_3dim6[i]);                        \
      give_value(send_vec[i * _SR_ + 1], zero, lat_3dim6[i]);                    \
      give_value(recv_vec[i * _SR_], zero, lat_3dim6[i]);                        \
      give_value(recv_vec[i * _SR_ + 1], zero, lat_3dim6[i]);                    \
    }                                                                          \
  }

#define _mpiDslashQcu(gridDim, blockDim, gauge, fermion_in, fermion_out,       \
                      parity, lat_1dim, lat_3dim12, node_rank, grid_1dim,      \
                      grid_index_1dim, move, send_request, recv_request,       \
                      send_vec, recv_vec)                                      \
  {                                                                            \
    wilson_dslash_clear_dest<<<gridDim, blockDim>>>(fermion_out, lat_1dim[X],  \
                                                    lat_1dim[Y], lat_1dim[Z]); \
    wilson_dslash_x_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], \
        lat_1dim[T], parity, send_vec[B_X], send_vec[F_X]);                    \
    if (grid_1dim[X] != 1) {                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      move_backward(move[_B_], grid_index_1dim[X], grid_1dim[X]);                \
      move_forward(move[_F_], grid_index_1dim[X], grid_1dim[X]);                 \
      move[_B_] =                                                                \
          node_rank + move[_B_] * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];    \
      move[_F_] =                                                                \
          node_rank + move[_F_] * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];    \
      MPI_Irecv(recv_vec[B_X], lat_3dim12[YZT], MPI_DOUBLE, move[_B_], F_X,      \
                MPI_COMM_WORLD, &recv_request[B_X]);                           \
      MPI_Irecv(recv_vec[F_X], lat_3dim12[YZT], MPI_DOUBLE, move[_F_], B_X,      \
                MPI_COMM_WORLD, &recv_request[F_X]);                           \
      MPI_Isend(send_vec[B_X], lat_3dim12[YZT], MPI_DOUBLE, move[_B_], B_X,      \
                MPI_COMM_WORLD, &send_request[B_X]);                           \
      MPI_Isend(send_vec[F_X], lat_3dim12[YZT], MPI_DOUBLE, move[_F_], F_X,      \
                MPI_COMM_WORLD, &send_request[F_T]);                           \
    }                                                                          \
    wilson_dslash_y_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], \
        lat_1dim[T], parity, send_vec[B_Y], send_vec[F_Y]);                    \
    if (grid_1dim[Y] != 1) {                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      move_backward(move[_B_], grid_index_1dim[Y], grid_1dim[Y]);                \
      move_forward(move[_F_], grid_index_1dim[Y], grid_1dim[Y]);                 \
      move[_B_] = node_rank + move[_B_] * grid_1dim[Z] * grid_1dim[T];             \
      move[_F_] = node_rank + move[_F_] * grid_1dim[Z] * grid_1dim[T];             \
      MPI_Irecv(recv_vec[B_Y], lat_3dim12[XZT], MPI_DOUBLE, move[_B_], F_Y,      \
                MPI_COMM_WORLD, &recv_request[B_Y]);                           \
      MPI_Irecv(recv_vec[F_Y], lat_3dim12[XZT], MPI_DOUBLE, move[_F_], B_Y,      \
                MPI_COMM_WORLD, &recv_request[F_Y]);                           \
      MPI_Isend(send_vec[B_Y], lat_3dim12[XZT], MPI_DOUBLE, move[_B_], B_Y,      \
                MPI_COMM_WORLD, &send_request[B_Y]);                           \
      MPI_Isend(send_vec[F_Y], lat_3dim12[XZT], MPI_DOUBLE, move[_F_], F_Y,      \
                MPI_COMM_WORLD, &send_request[F_Y]);                           \
    }                                                                          \
    wilson_dslash_z_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], \
        lat_1dim[T], parity, send_vec[B_Z], send_vec[F_Z]);                    \
    if (grid_1dim[Z] != 1) {                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      move_backward(move[_B_], grid_index_1dim[Z], grid_1dim[Z]);                \
      move_forward(move[_F_], grid_index_1dim[Z], grid_1dim[Z]);                 \
      move[_B_] = node_rank + move[_B_] * grid_1dim[T];                            \
      move[_F_] = node_rank + move[_F_] * grid_1dim[T];                            \
      MPI_Irecv(recv_vec[B_Z], lat_3dim12[XYT], MPI_DOUBLE, move[_B_], F_Z,      \
                MPI_COMM_WORLD, &recv_request[B_Z]);                           \
      MPI_Irecv(recv_vec[F_Z], lat_3dim12[XYT], MPI_DOUBLE, move[_F_], B_Z,      \
                MPI_COMM_WORLD, &recv_request[F_Z]);                           \
      MPI_Isend(send_vec[B_Z], lat_3dim12[XYT], MPI_DOUBLE, move[_B_], B_Z,      \
                MPI_COMM_WORLD, &send_request[B_Z]);                           \
      MPI_Isend(send_vec[F_Z], lat_3dim12[XYT], MPI_DOUBLE, move[_F_], F_Z,      \
                MPI_COMM_WORLD, &send_request[F_Z]);                           \
    }                                                                          \
    wilson_dslash_t_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], \
        lat_1dim[T], parity, send_vec[B_T], send_vec[F_T]);                    \
    if (grid_1dim[T] != 1) {                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      move_backward(move[_B_], grid_index_1dim[T], grid_1dim[T]);                \
      move_forward(move[_F_], grid_index_1dim[T], grid_1dim[T]);                 \
      move[_B_] = node_rank + move[_B_];                                           \
      move[_F_] = node_rank + move[_F_];                                           \
      MPI_Irecv(recv_vec[B_T], lat_3dim12[XYZ], MPI_DOUBLE, move[_B_], F_T,      \
                MPI_COMM_WORLD, &recv_request[B_T]);                           \
      MPI_Irecv(recv_vec[F_T], lat_3dim12[XYZ], MPI_DOUBLE, move[_F_], B_T,      \
                MPI_COMM_WORLD, &recv_request[F_T]);                           \
      MPI_Isend(send_vec[B_T], lat_3dim12[XYZ], MPI_DOUBLE, move[_B_], B_T,      \
                MPI_COMM_WORLD, &send_request[B_T]);                           \
      MPI_Isend(send_vec[F_T], lat_3dim12[XYZ], MPI_DOUBLE, move[_F_], F_T,      \
                MPI_COMM_WORLD, &send_request[F_T]);                           \
    }                                                                          \
    if (grid_1dim[X] != 1) {                                                   \
      MPI_Wait(&recv_request[B_X], MPI_STATUS_IGNORE);                         \
      MPI_Wait(&recv_request[F_X], MPI_STATUS_IGNORE);                         \
      wilson_dslash_x_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, recv_vec[B_X], recv_vec[F_X]);                  \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_x_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, send_vec[F_X], send_vec[B_X]);                  \
    }                                                                          \
    if (grid_1dim[Y] != 1) {                                                   \
      MPI_Wait(&recv_request[B_Y], MPI_STATUS_IGNORE);                         \
      MPI_Wait(&recv_request[F_Y], MPI_STATUS_IGNORE);                         \
      wilson_dslash_y_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, recv_vec[B_Y], recv_vec[F_Y]);                  \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_y_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, send_vec[F_Y], send_vec[B_Y]);                  \
    }                                                                          \
    if (grid_1dim[Z] != 1) {                                                   \
      MPI_Wait(&recv_request[B_Z], MPI_STATUS_IGNORE);                         \
      MPI_Wait(&recv_request[F_Z], MPI_STATUS_IGNORE);                         \
      wilson_dslash_z_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, recv_vec[B_Z], recv_vec[F_Z]);                  \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_z_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, send_vec[F_Z], send_vec[B_Z]);                  \
    }                                                                          \
    if (grid_1dim[T] != 1) {                                                   \
      MPI_Wait(&recv_request[B_T], MPI_STATUS_IGNORE);                         \
      MPI_Wait(&recv_request[F_T], MPI_STATUS_IGNORE);                         \
      wilson_dslash_t_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, recv_vec[B_T], recv_vec[F_T]);                  \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_t_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, send_vec[F_T], send_vec[B_T]);                  \
    }                                                                          \
    MPI_Barrier(MPI_COMM_WORLD);                                               \
    checkCudaErrors(cudaDeviceSynchronize());                                  \
  }

#define malloc_vec(lat_3dim6, send_vec, recv_vec)                              \
  {                                                                            \
    for (int i = 0; i < _DIM_; i++) {                                            \
      cudaMallocManaged(&send_vec[i * _SR_],                                     \
                        lat_3dim6[i] * sizeof(LatticeComplex));                \
      cudaMallocManaged(&send_vec[i * _SR_ + 1],                                 \
                        lat_3dim6[i] * sizeof(LatticeComplex));                \
      cudaMallocManaged(&recv_vec[i * _SR_],                                     \
                        lat_3dim6[i] * sizeof(LatticeComplex));                \
      cudaMallocManaged(&recv_vec[i * _SR_ + 1],                                 \
                        lat_3dim6[i] * sizeof(LatticeComplex));                \
    }                                                                          \
  }

#define free_vec(send_vec, recv_vec)                                           \
  {                                                                            \
    for (int i = 0; i < _WARDS_; i++) {                                          \
      cudaFree(send_vec[i]);                                                   \
      cudaFree(recv_vec[i]);                                                   \
    }                                                                          \
  }

#define mpi_dot(local_result, lat_4dim12, val0, val1, tmp, zero)               \
  {                                                                            \
    local_result = zero;                                                       \
    for (int i = 0; i < lat_4dim12; i++) {                                     \
      local_result += val0[i].conj() * val1[i];                                \
    }                                                                          \
    MPI_Allreduce(&local_result, &tmp, 2, MPI_DOUBLE, MPI_SUM,                 \
                  MPI_COMM_WORLD);                                             \
    MPI_Barrier(MPI_COMM_WORLD);                                               \
  }

#define mpi_diff(local_result, lat_4dim12, val0, val1, tmp, latt_tmp0, tmp0,   \
                 tmp1, zero)                                                   \
  {                                                                            \
    give_value(latt_tmp0, zero, lat_4dim12);                                   \
    for (int i = 0; i < lat_4dim12; i++) {                                     \
      latt_tmp0[i] = val0[i] - val1[i];                                        \
    }                                                                          \
    mpi_dot(local_result, lat_4dim12, latt_tmp0, latt_tmp0, tmp0, zero);       \
    mpi_dot(local_result, lat_4dim12, val1, val1, tmp1, zero);                 \
    tmp = tmp0 / tmp1;                                                         \
  }

#define _dslash_eo(dest_e, src_o, node_rank, gridDim, blockDim, gauge,         \
                   lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move,     \
                   send_request, recv_request, send_vec, recv_vec, zero)       \
  {                                                                            \
    zero_vec(lat_3dim6, send_vec, recv_vec, zero);                             \
    _mpiDslashQcu(gridDim, blockDim, gauge, src_o, dest_e, EVEN, lat_1dim,     \
                  lat_3dim12, node_rank, grid_1dim, grid_index_1dim, move,     \
                  send_request, recv_request, send_vec, recv_vec);             \
  }

#define _dslash_oe(dest_o, src_e, node_rank, gridDim, blockDim, gauge,         \
                   lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move,     \
                   send_request, recv_request, send_vec, recv_vec, zero)       \
  {                                                                            \
    zero_vec(lat_3dim6, send_vec, recv_vec, zero);                             \
    _mpiDslashQcu(gridDim, blockDim, gauge, src_e, dest_o, ODD, lat_1dim,      \
                  lat_3dim12, node_rank, grid_1dim, grid_index_1dim, move,     \
                  send_request, recv_request, send_vec, recv_vec);             \
  }

// src_o-kappa**2*dslash_oe(dslash_eo(src_o))
#define _dslash(dest_o, src_o, kappa, latt_tmp0, latt_tmp1, node_rank,         \
                gridDim, blockDim, gauge, lat_1dim, lat_3dim12, lat_4dim12,    \
                grid_1dim, grid_index_1dim, move, send_request, recv_request,  \
                send_vec, recv_vec, zero)                                      \
  {                                                                            \
    give_value(latt_tmp0, zero, lat_4dim12);                                   \
    _dslash_eo(latt_tmp0, src_o, node_rank, gridDim, blockDim, gauge,          \
               lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move,         \
               send_request, recv_request, send_vec, recv_vec, zero);          \
    give_value(latt_tmp1, zero, lat_4dim12);                                   \
    _dslash_oe(latt_tmp1, latt_tmp0, node_rank, gridDim, blockDim, gauge,      \
               lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move,         \
               send_request, recv_request, send_vec, recv_vec, zero);          \
    for (int i = 0; i < lat_4dim12; i++) {                                     \
      dest_o[i] = src_o[i] - latt_tmp1[i] * kappa * kappa;                     \
    }                                                                          \
  }

#endif
