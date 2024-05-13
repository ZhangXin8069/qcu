#ifndef _DEFINE_H
#define _DEFINE_H
#pragma optimize(5)
#include "./qcu.h"
#define BLOCK_SIZE 256
#define X 0
#define Y 1
#define Z 2
#define T 3
#define DIM 4
#define B_X 0
#define F_X 1
#define B_Y 2
#define F_Y 3
#define B_Z 4
#define F_Z 5
#define B_T 6
#define F_T 7
#define WARDS 8
#define YZT 0
#define XZT 1
#define XYT 2
#define XYZ 3
#define EVEN 0
#define ODD 1
#define EVENODD 2
#define LAT_C 3
#define LAT_S 4
#define LAT_D 4
#define B 0
#define F 1
#define BF 2
#define OUTPUT_SIZE 10
#define BACKWARD -1
#define NOWARD 0
#define FORWARD 1
#define SR 2
#define LAT_EXAMPLE 32
#define GRID_EXAMPLE 1

#define WILSON_DSLASH
#define CLOVER_DSLASH
// #define OVERLAP_DSLASH
#define MPI_WILSON_DSLASH
// #define MPI_CLOVER_DSLASH
// #define MPI_OVERLAP_DSLASH
#define NCCL_WILSON_DSLASH
// #define NCCL_CLOVER_DSLASH
// #define NCCL_OVERLAP_DSLASH
#define TEST_WILSON_DSLASH
#define TEST_CLOVER_DSLASH
// #define TEST_OVERLAP_DSLASH
#define WILSON_BISTABCG
// #define CLOVER_BISTABCG
// #define OVERLAP_BISTABCG
#define MPI_WILSON_BISTABCG
// #define MPI_CLOVER_BISTABCG
// #define MPI_OVERLAP_BISTABCG
#define NCCL_WILSON_BISTABCG
// #define NCCL_CLOVER_BISTABCG
// #define NCCL_OVERLAP_BISTABCG
#define TEST_WILSON_BISTABCG
// #define TEST_CLOVER_BISTABCG
// #define TEST_OVERLAP_BISTABCG
// #define WILSON_MULTGRID
// #define CLOVER_MULTGRID
// #define OVERLAP_MULTGRID
// #define MPI_WILSON_MULTGRID
// #define MPI_CLOVER_MULTGRID
// #define MPI_OVERLAP_MULTGRID
// #define NCCL_WILSON_MULTGRID
// #define NCCL_CLOVER_MULTGRID
// #define NCCL_OVERLAP_MULTGRID
// #define TEST_WILSON_MULTGRID
// #define TEST_CLOVER_MULTGRID
// #define TEST_OVERLAP_MULTGRID
#define print_ptr(ptr, index)                                                  \
  {                                                                            \
    checkCudaErrors(cudaDeviceSynchronize());                                  \
    printf("ptr(%p)[%d]:%.9lf + %.9lfi\n", ptr, index,                         \
           static_cast<LatticeComplex *>(ptr)[index].real,                     \
           static_cast<LatticeComplex *>(ptr)[index].imag);                    \
  }

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

#define MPICHECK(cmd)                                                          \
  do {                                                                         \
    int e = cmd;                                                               \
    if (e != MPI_SUCCESS) {                                                    \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,            \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
             ncclGetErrorString(r));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// little strange, but don't want change
#define host_give_value(U, zero, n)                                            \
  {                                                                            \
    LatticeComplex *tmp_U = static_cast<LatticeComplex *>(U);                  \
    for (int i = 0; i < n; i++) {                                              \
      tmp_U[i] = zero;                                                         \
    }                                                                          \
  }

#define device_give_value(host_Udevice_U, host_zero, n)                        \
  {                                                                            \
    host_give_value(host_U, host_zero, n);                                     \
    cudaMemcpy(device_U, host_U, sizeof(LatticeComplex) * n,                   \
               cudaMemcpyHostToDevice);                                        \
    checkCudaErrors(cudaDeviceSynchronize());                                  \
  }

#define host_give_rand(input_matrix, size)                                     \
  {                                                                            \
    for (int i = 0; i < size; i++) {                                           \
      input_matrix[i].real = static_cast<double>(rand()) / RAND_MAX;           \
      input_matrix[i].imag = static_cast<double>(rand()) / RAND_MAX;           \
    }                                                                          \
  }

#define device_give_rand(host_input_matrix, device_input_matrix, size)         \
  {                                                                            \
    host_give_rand(host_input_matrix, size);                                   \
    cudaMemcpy(device_input_matrix, host_input_matrix,                         \
               sizeof(LatticeComplex) * size, cudaMemcpyHostToDevice);         \
    checkCudaErrors(cudaDeviceSynchronize());                                  \
  }

#define host_zero_vec(lat_3dim6, host_send_vec, host_recv_vec, zero)           \
  {                                                                            \
    for (int i = 0; i < DIM; i++) {                                            \
      host_give_value(host_send_vec[i * SR], zero, lat_3dim6[i]);              \
      host_give_value(host_send_vec[i * SR + 1], zero, lat_3dim6[i]);          \
      host_give_value(host_recv_vec[i * SR], zero, lat_3dim6[i]);              \
      host_give_value(host_recv_vec[i * SR + 1], zero, lat_3dim6[i]);          \
    }                                                                          \
  }

#define device_zero_vec(lat_3dim6, device_send_vec, device_recv_vec,           \
                        host_send_vec, host_recv_vec, zero)                    \
  {                                                                            \
    host_zero_vec(lat_3dim6, host_send_vec, host_recv_vec, zero);              \
    for (int i = 0; i < DIM; i++) {                                            \
      cudaMemcpy(device_send_vec[i * SR], device_send_vec[i * SR],             \
                 sizeof(LatticeComplex) * lat_3dim6[i],                        \
                 cudaMemcpyHostToDevice);                                      \
      cudaMemcpy(device_send_vec[i * SR + 1], device_send_vec[i * SR + 1],     \
                 sizeof(LatticeComplex) * lat_3dim6[i],                        \
                 cudaMemcpyHostToDevice);                                      \
      cudaMemcpy(device_recv_vec[i * SR], device_recv_vec[i * SR],             \
                 sizeof(LatticeComplex) * lat_3dim6[i],                        \
                 cudaMemcpyHostToDevice);                                      \
      cudaMemcpy(device_recv_vec[i * SR + 1], device_recv_vec[i * SR + 1],     \
                 sizeof(LatticeComplex) * lat_3dim6[i],                        \
                 cudaMemcpyHostToDevice);                                      \
    }                                                                          \
    checkCudaErrors(cudaDeviceSynchronize());                                  \
  }

#define give_ptr(U, origin_U, n)                                               \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] = origin_U[i];                                                      \
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

#define _mpiDslashQcu(gridDim, blockDim, gauge, fermion_in, fermion_out,       \
                      parity, lat_1dim, lat_3dim12, node_rank, grid_1dim,      \
                      grid_index_1dim, move, send_request, recv_request,       \
                      device_send_vec, device_recv_vec, host_send_vec,         \
                      host_recv_vec)                                           \
  {                                                                            \
    wilson_dslash_clear_dest<<<gridDim, blockDim>>>(fermion_out, lat_1dim[X],  \
                                                    lat_1dim[Y], lat_1dim[Z]); \
    checkCudaErrors(cudaDeviceSynchronize());                                  \
    wilson_dslash_x_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], \
        lat_1dim[T], parity, device_send_vec[B_X], device_send_vec[F_X]);      \
    cudaMemcpy(host_send_vec[B_X], device_send_vec[B_X],                       \
               sizeof(double) * lat_3dim12[YZT], cudaMemcpyDeviceToHost);      \
    cudaMemcpy(host_send_vec[F_X], device_send_vec[F_X],                       \
               sizeof(double) * lat_3dim12[YZT], cudaMemcpyDeviceToHost);      \
    if (grid_1dim[X] != 1) {                                                   \
      move_backward(move[B], grid_index_1dim[X], grid_1dim[X]);                \
      move_forward(move[F], grid_index_1dim[X], grid_1dim[X]);                 \
      move[B] =                                                                \
          node_rank + move[B] * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];    \
      move[F] =                                                                \
          node_rank + move[F] * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];    \
      MPI_Irecv(host_recv_vec[B_X], lat_3dim12[YZT], MPI_DOUBLE, move[B], F_X, \
                MPI_COMM_WORLD, &recv_request[B_X]);                           \
      MPI_Irecv(host_recv_vec[F_X], lat_3dim12[YZT], MPI_DOUBLE, move[F], B_X, \
                MPI_COMM_WORLD, &recv_request[F_X]);                           \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      MPI_Isend(host_send_vec[B_X], lat_3dim12[YZT], MPI_DOUBLE, move[B], B_X, \
                MPI_COMM_WORLD, &send_request[B_X]);                           \
      MPI_Isend(host_send_vec[F_X], lat_3dim12[YZT], MPI_DOUBLE, move[F], F_X, \
                MPI_COMM_WORLD, &send_request[F_T]);                           \
    }                                                                          \
    wilson_dslash_y_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], \
        lat_1dim[T], parity, device_send_vec[B_Y], device_send_vec[F_Y]);      \
    cudaMemcpy(host_send_vec[B_Y], device_send_vec[B_Y],                       \
               sizeof(double) * lat_3dim12[XZT], cudaMemcpyDeviceToHost);      \
    cudaMemcpy(host_send_vec[F_Y], device_send_vec[F_Y],                       \
               sizeof(double) * lat_3dim12[XZT], cudaMemcpyDeviceToHost);      \
    if (grid_1dim[Y] != 1) {                                                   \
      move_backward(move[B], grid_index_1dim[Y], grid_1dim[Y]);                \
      move_forward(move[F], grid_index_1dim[Y], grid_1dim[Y]);                 \
      move[B] = node_rank + move[B] * grid_1dim[Z] * grid_1dim[T];             \
      move[F] = node_rank + move[F] * grid_1dim[Z] * grid_1dim[T];             \
      MPI_Irecv(host_recv_vec[B_Y], lat_3dim12[XZT], MPI_DOUBLE, move[B], F_Y, \
                MPI_COMM_WORLD, &recv_request[B_Y]);                           \
      MPI_Irecv(host_recv_vec[F_Y], lat_3dim12[XZT], MPI_DOUBLE, move[F], B_Y, \
                MPI_COMM_WORLD, &recv_request[F_Y]);                           \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      MPI_Isend(host_send_vec[B_Y], lat_3dim12[XZT], MPI_DOUBLE, move[B], B_Y, \
                MPI_COMM_WORLD, &send_request[B_Y]);                           \
      MPI_Isend(host_send_vec[F_Y], lat_3dim12[XZT], MPI_DOUBLE, move[F], F_Y, \
                MPI_COMM_WORLD, &send_request[F_Y]);                           \
    }                                                                          \
    wilson_dslash_z_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], \
        lat_1dim[T], parity, device_send_vec[B_Z], device_send_vec[F_Z]);      \
    cudaMemcpy(host_send_vec[B_Z], device_send_vec[B_Z],                       \
               sizeof(double) * lat_3dim12[XYT], cudaMemcpyDeviceToHost);      \
    cudaMemcpy(host_send_vec[F_Z], device_send_vec[F_Z],                       \
               sizeof(double) * lat_3dim12[XYT], cudaMemcpyDeviceToHost);      \
    if (grid_1dim[Z] != 1) {                                                   \
      move_backward(move[B], grid_index_1dim[Z], grid_1dim[Z]);                \
      move_forward(move[F], grid_index_1dim[Z], grid_1dim[Z]);                 \
      move[B] = node_rank + move[B] * grid_1dim[T];                            \
      move[F] = node_rank + move[F] * grid_1dim[T];                            \
      MPI_Irecv(host_recv_vec[B_Z], lat_3dim12[XYT], MPI_DOUBLE, move[B], F_Z, \
                MPI_COMM_WORLD, &recv_request[B_Z]);                           \
      MPI_Irecv(host_recv_vec[F_Z], lat_3dim12[XYT], MPI_DOUBLE, move[F], B_Z, \
                MPI_COMM_WORLD, &recv_request[F_Z]);                           \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      MPI_Isend(host_send_vec[B_Z], lat_3dim12[XYT], MPI_DOUBLE, move[B], B_Z, \
                MPI_COMM_WORLD, &send_request[B_Z]);                           \
      MPI_Isend(host_send_vec[F_Z], lat_3dim12[XYT], MPI_DOUBLE, move[F], F_Z, \
                MPI_COMM_WORLD, &send_request[F_Z]);                           \
    }                                                                          \
    wilson_dslash_t_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], \
        lat_1dim[T], parity, device_send_vec[B_T], device_send_vec[F_T]);      \
    cudaMemcpy(host_send_vec[B_T], device_send_vec[B_T],                       \
               sizeof(double) * lat_3dim12[XYZ], cudaMemcpyDeviceToHost);      \
    cudaMemcpy(host_send_vec[F_T], device_send_vec[F_T],                       \
               sizeof(double) * lat_3dim12[XYZ], cudaMemcpyDeviceToHost);      \
    if (grid_1dim[T] != 1) {                                                   \
      move_backward(move[B], grid_index_1dim[T], grid_1dim[T]);                \
      move_forward(move[F], grid_index_1dim[T], grid_1dim[T]);                 \
      move[B] = node_rank + move[B];                                           \
      move[F] = node_rank + move[F];                                           \
      MPI_Irecv(host_recv_vec[B_T], lat_3dim12[XYZ], MPI_DOUBLE, move[B], F_T, \
                MPI_COMM_WORLD, &recv_request[B_T]);                           \
      MPI_Irecv(host_recv_vec[F_T], lat_3dim12[XYZ], MPI_DOUBLE, move[F], B_T, \
                MPI_COMM_WORLD, &recv_request[F_T]);                           \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      MPI_Isend(host_send_vec[B_T], lat_3dim12[XYZ], MPI_DOUBLE, move[B], B_T, \
                MPI_COMM_WORLD, &send_request[B_T]);                           \
      MPI_Isend(host_send_vec[F_T], lat_3dim12[XYZ], MPI_DOUBLE, move[F], F_T, \
                MPI_COMM_WORLD, &send_request[F_T]);                           \
    }                                                                          \
    if (grid_1dim[X] != 1) {                                                   \
      MPI_Wait(&recv_request[B_X], MPI_STATUS_IGNORE);                         \
      MPI_Wait(&recv_request[F_X], MPI_STATUS_IGNORE);                         \
      cudaMemcpy(device_recv_vec[B_X], host_recv_vec[B_X],                     \
                 sizeof(double) * lat_3dim12[YZT], cudaMemcpyHostToDevice);    \
      cudaMemcpy(device_recv_vec[F_X], host_recv_vec[F_X],                     \
                 sizeof(double) * lat_3dim12[YZT], cudaMemcpyHostToDevice);    \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_x_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_recv_vec[B_X], device_recv_vec[F_X]);    \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_x_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_send_vec[F_X], device_send_vec[B_X]);    \
    }                                                                          \
    if (grid_1dim[Y] != 1) {                                                   \
      MPI_Wait(&recv_request[B_Y], MPI_STATUS_IGNORE);                         \
      MPI_Wait(&recv_request[F_Y], MPI_STATUS_IGNORE);                         \
      cudaMemcpy(device_recv_vec[B_Y], host_recv_vec[B_Y],                     \
                 sizeof(double) * lat_3dim12[XZT], cudaMemcpyHostToDevice);    \
      cudaMemcpy(device_recv_vec[F_Y], host_recv_vec[F_Y],                     \
                 sizeof(double) * lat_3dim12[XZT], cudaMemcpyHostToDevice);    \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_y_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_recv_vec[B_Y], device_recv_vec[F_Y]);    \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_y_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_send_vec[F_Y], device_send_vec[B_Y]);    \
    }                                                                          \
    if (grid_1dim[Z] != 1) {                                                   \
      MPI_Wait(&recv_request[B_Z], MPI_STATUS_IGNORE);                         \
      MPI_Wait(&recv_request[F_Z], MPI_STATUS_IGNORE);                         \
      cudaMemcpy(device_recv_vec[B_Z], host_recv_vec[B_Z],                     \
                 sizeof(double) * lat_3dim12[XYT], cudaMemcpyHostToDevice);    \
      cudaMemcpy(device_recv_vec[F_Z], host_recv_vec[F_Z],                     \
                 sizeof(double) * lat_3dim12[XYT], cudaMemcpyHostToDevice);    \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_z_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_recv_vec[B_Z], device_recv_vec[F_Z]);    \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_z_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_send_vec[F_Z], device_send_vec[B_Z]);    \
    }                                                                          \
    if (grid_1dim[T] != 1) {                                                   \
      MPI_Wait(&recv_request[B_T], MPI_STATUS_IGNORE);                         \
      MPI_Wait(&recv_request[F_T], MPI_STATUS_IGNORE);                         \
      cudaMemcpy(device_recv_vec[B_T], host_recv_vec[B_T],                     \
                 sizeof(double) * lat_3dim12[XYZ], cudaMemcpyHostToDevice);    \
      cudaMemcpy(device_recv_vec[F_T], host_recv_vec[F_T],                     \
                 sizeof(double) * lat_3dim12[XYZ], cudaMemcpyHostToDevice);    \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_t_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_recv_vec[B_T], device_recv_vec[F_T]);    \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_t_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_send_vec[F_T], device_send_vec[B_T]);    \
    }                                                                          \
    MPI_Barrier(MPI_COMM_WORLD);                                               \
    checkCudaErrors(cudaDeviceSynchronize());                                  \
  }

#define malloc_vec(lat_3dim6, device_send_vec, device_recv_vec, host_send_vec, \
                   host_recv_vec)                                              \
  {                                                                            \
    for (int i = 0; i < DIM; i++) {                                            \
      cudaMalloc(&device_send_vec[i * SR],                                     \
                 lat_3dim6[i] * sizeof(LatticeComplex));                       \
      cudaMalloc(&device_send_vec[i * SR + 1],                                 \
                 lat_3dim6[i] * sizeof(LatticeComplex));                       \
      cudaMalloc(&device_recv_vec[i * SR],                                     \
                 lat_3dim6[i] * sizeof(LatticeComplex));                       \
      cudaMalloc(&device_recv_vec[i * SR + 1],                                 \
                 lat_3dim6[i] * sizeof(LatticeComplex));                       \
      host_send_vec[i * SR] =                                                  \
          (void *)malloc(lat_3dim6[i] * sizeof(LatticeComplex));               \
      host_send_vec[i * SR + 1] =                                              \
          (void *)malloc(lat_3dim6[i] * sizeof(LatticeComplex));               \
      host_recv_vec[i * SR] =                                                  \
          (void *)malloc(lat_3dim6[i] * sizeof(LatticeComplex));               \
      host_recv_vec[i * SR + 1] =                                              \
          (void *)malloc(lat_3dim6[i] * sizeof(LatticeComplex));               \
    }                                                                          \
  }

#define free_vec(device_send_vec, device_recv_vec, host_send_vec,              \
                 host_recv_vec)                                                \
  {                                                                            \
    for (int i = 0; i < WARDS; i++) {                                          \
      cudaFree(device_send_vec[i]);                                            \
      cudaFree(device_recv_vec[i]);                                            \
      free(host_send_vec[i]);                                                  \
      free(host_recv_vec[i]);                                                  \
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

#define mpi_diff(local_result, lat_4dim12, val0, val1, tmp, device_latt_tmp0,  \
                 tmp0, tmp1, zero)                                             \
  {                                                                            \
    for (int i = 0; i < lat_4dim12; i++) {                                     \
      device_latt_tmp0[i] = val0[i] - val1[i];                                 \
    }                                                                          \
    mpi_dot(local_result, lat_4dim12, device_latt_tmp0, device_latt_tmp0,      \
            tmp0, zero);                                                       \
    mpi_dot(local_result, lat_4dim12, val1, val1, tmp1, zero);                 \
    tmp = tmp0 / tmp1;                                                         \
  }

#define mpi_dslash_eo(dest_e, src_o, node_rank, gridDim, blockDim, gauge,      \
                      lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move,  \
                      send_request, recv_request, device_send_vec,             \
                      device_recv_vec, host_send_vec, host_recv_vec, zero)     \
  {                                                                            \
    _mpiDslashQcu(gridDim, blockDim, gauge, src_o, dest_e, EVEN, lat_1dim,     \
                  lat_3dim12, node_rank, grid_1dim, grid_index_1dim, move,     \
                  send_request, recv_request, device_send_vec,                 \
                  device_recv_vec, host_send_vec, host_recv_vec);              \
  }

#define mpi_dslash_oe(dest_o, src_e, node_rank, gridDim, blockDim, gauge,      \
                      lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move,  \
                      send_request, recv_request, device_send_vec,             \
                      device_recv_vec, host_send_vec, host_recv_vec, zero)     \
  {                                                                            \
    _mpiDslashQcu(gridDim, blockDim, gauge, src_e, dest_o, ODD, lat_1dim,      \
                  lat_3dim12, node_rank, grid_1dim, grid_index_1dim, move,     \
                  send_request, recv_request, device_send_vec,                 \
                  device_recv_vec, host_send_vec, host_recv_vec);              \
  }

// src_o-kappa**2*dslash_oe(dslash_eo(src_o))
#define mpi_dslash(dest_o, src_o, kappa, device_latt_tmp0, device_latt_tmp1,   \
                   node_rank, gridDim, blockDim, gauge, lat_1dim, lat_3dim12,  \
                   lat_4dim12, grid_1dim, grid_index_1dim, move, send_request, \
                   recv_request, device_send_vec, device_recv_vec,             \
                   host_send_vec, host_recv_vec, zero)                         \
  {                                                                            \
    mpi_dslash_eo(device_latt_tmp0, src_o, node_rank, gridDim, blockDim,       \
                  gauge, lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim,     \
                  move, send_request, recv_request, device_send_vec,           \
                  device_recv_vec, host_send_vec, host_recv_vec, zero);        \
    mpi_dslash_oe(device_latt_tmp1, device_latt_tmp0, node_rank, gridDim,      \
                  blockDim, gauge, lat_1dim, lat_3dim12, grid_1dim,            \
                  grid_index_1dim, move, send_request, recv_request,           \
                  device_send_vec, device_recv_vec, host_send_vec,             \
                  host_recv_vec, zero);                                        \
    for (int i = 0; i < lat_4dim12; i++) {                                     \
      dest_o[i] = src_o[i] - device_latt_tmp1[i] * kappa * kappa;              \
    }                                                                          \
  }

static uint64_t getHostHash(const char *string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

static void getHostName(char *hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

#define _ncclDslashQcu(gridDim, blockDim, gauge, fermion_in, fermion_out,      \
                       parity, lat_1dim, lat_3dim12, node_rank, grid_1dim,     \
                       grid_index_1dim, move, device_send_vec,                 \
                       device_recv_vec, nccl_comm, stream)                     \
  {                                                                            \
    ncclGroupStart();                                                          \
    wilson_dslash_clear_dest<<<gridDim, blockDim>>>(fermion_out, lat_1dim[X],  \
                                                    lat_1dim[Y], lat_1dim[Z]); \
    checkCudaErrors(cudaDeviceSynchronize());                                  \
    wilson_dslash_x_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], \
        lat_1dim[T], parity, device_send_vec[B_X], device_send_vec[F_X]);      \
    if (grid_1dim[X] != 1) {                                                   \
      move_backward(move[B], grid_index_1dim[X], grid_1dim[X]);                \
      move_forward(move[F], grid_index_1dim[X], grid_1dim[X]);                 \
      move[B] =                                                                \
          node_rank + move[B] * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];    \
      move[F] =                                                                \
          node_rank + move[F] * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];    \
      ncclRecv(device_recv_vec[B_X], lat_3dim12[YZT], ncclDouble, move[B],     \
               nccl_comm, stream);                                             \
      ncclRecv(device_recv_vec[F_X], lat_3dim12[YZT], ncclDouble, move[F],     \
               nccl_comm, stream);                                             \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      ncclSend(device_send_vec[B_X], lat_3dim12[YZT], ncclDouble, move[B],     \
               nccl_comm, stream);                                             \
      ncclSend(device_send_vec[F_X], lat_3dim12[YZT], ncclDouble, move[F],     \
               nccl_comm, stream);                                             \
    }                                                                          \
    wilson_dslash_y_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], \
        lat_1dim[T], parity, device_send_vec[B_Y], device_send_vec[F_Y]);      \
    if (grid_1dim[Y] != 1) {                                                   \
      move_backward(move[B], grid_index_1dim[Y], grid_1dim[Y]);                \
      move_forward(move[F], grid_index_1dim[Y], grid_1dim[Y]);                 \
      move[B] = node_rank + move[B] * grid_1dim[Z] * grid_1dim[T];             \
      move[F] = node_rank + move[F] * grid_1dim[Z] * grid_1dim[T];             \
      ncclRecv(device_recv_vec[B_Y], lat_3dim12[XZT], ncclDouble, move[B],     \
               nccl_comm, stream);                                             \
      ncclRecv(device_recv_vec[F_Y], lat_3dim12[XZT], ncclDouble, move[F],     \
               nccl_comm, stream);                                             \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      ncclSend(device_send_vec[B_Y], lat_3dim12[XZT], ncclDouble, move[B],     \
               nccl_comm, stream);                                             \
      ncclSend(device_send_vec[F_Y], lat_3dim12[XZT], ncclDouble, move[F],     \
               nccl_comm, stream);                                             \
    }                                                                          \
    wilson_dslash_z_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], \
        lat_1dim[T], parity, device_send_vec[B_Z], device_send_vec[F_Z]);      \
    if (grid_1dim[Z] != 1) {                                                   \
      move_backward(move[B], grid_index_1dim[Z], grid_1dim[Z]);                \
      move_forward(move[F], grid_index_1dim[Z], grid_1dim[Z]);                 \
      move[B] = node_rank + move[B] * grid_1dim[T];                            \
      move[F] = node_rank + move[F] * grid_1dim[T];                            \
      ncclRecv(device_recv_vec[B_Z], lat_3dim12[XYT], ncclDouble, move[B],     \
               nccl_comm, stream);                                             \
      ncclRecv(device_recv_vec[F_Z], lat_3dim12[XYT], ncclDouble, move[F],     \
               nccl_comm, stream);                                             \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      ncclSend(device_send_vec[B_Z], lat_3dim12[XYT], ncclDouble, move[B],     \
               nccl_comm, stream);                                             \
      ncclSend(device_send_vec[F_Z], lat_3dim12[XYT], ncclDouble, move[F],     \
               nccl_comm, stream);                                             \
    }                                                                          \
    wilson_dslash_t_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], \
        lat_1dim[T], parity, device_send_vec[B_T], device_send_vec[F_T]);      \
    if (grid_1dim[T] != 1) {                                                   \
      move_backward(move[B], grid_index_1dim[T], grid_1dim[T]);                \
      move_forward(move[F], grid_index_1dim[T], grid_1dim[T]);                 \
      move[B] = node_rank + move[B];                                           \
      move[F] = node_rank + move[F];                                           \
      ncclRecv(device_recv_vec[B_T], lat_3dim12[XYZ], ncclDouble, move[B],     \
               nccl_comm, stream);                                             \
      ncclRecv(device_recv_vec[F_T], lat_3dim12[XYZ], ncclDouble, move[F],     \
               nccl_comm, stream);                                             \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      ncclSend(device_send_vec[B_T], lat_3dim12[XYZ], ncclDouble, move[B],     \
               nccl_comm, stream);                                             \
      ncclSend(device_send_vec[F_T], lat_3dim12[XYZ], ncclDouble, move[F],     \
               nccl_comm, stream);                                             \
    }                                                                          \
    if (grid_1dim[X] != 1) {                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_x_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_recv_vec[B_X], device_recv_vec[F_X]);    \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_x_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_send_vec[F_X], device_send_vec[B_X]);    \
    }                                                                          \
    if (grid_1dim[Y] != 1) {                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_y_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_recv_vec[B_Y], device_recv_vec[F_Y]);    \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_y_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_send_vec[F_Y], device_send_vec[B_Y]);    \
    }                                                                          \
    if (grid_1dim[Z] != 1) {                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_z_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_recv_vec[B_Z], device_recv_vec[F_Z]);    \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_z_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_send_vec[F_Z], device_send_vec[B_Z]);    \
    }                                                                          \
    if (grid_1dim[T] != 1) {                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_t_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_recv_vec[B_T], device_recv_vec[F_T]);    \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_t_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],           \
          lat_1dim[T], parity, device_send_vec[F_T], device_send_vec[B_T]);    \
    }                                                                          \
    checkCudaErrors(cudaDeviceSynchronize());                                  \
    MPI_Barrier(MPI_COMM_WORLD);                                               \
    ncclGroupEnd();                                                            \
  }

#define nccl_dot(local_result, lat_4dim12, val0, val1, tmp, zero, nccl_comm,   \
                 stream)                                                       \
  {                                                                            \
    (*local_result) = zero;                                                    \
    for (int i = 0; i < lat_4dim12; i++) {                                     \
      (*local_result) += val0[i].conj() * val1[i];                             \
    }                                                                          \
    NCCLCHECK(ncclAllReduce((const void *)local_result, (void *)tmp, 2,        \
                            ncclDouble, ncclSum, nccl_comm, stream));          \
    CUDACHECK(cudaStreamSynchronize(stream));                                  \
  }

#define nccl_diff(local_result, lat_4dim12, val0, val1, tmp, device_latt_tmp0, \
                  tmp0, tmp1, zero, nccl_comm, stream)                         \
  {                                                                            \
    for (int i = 0; i < lat_4dim12; i++) {                                     \
      device_latt_tmp0[i] = val0[i] - val1[i];                                 \
    }                                                                          \
    nccl_dot(local_result, lat_4dim12, device_latt_tmp0, device_latt_tmp0,     \
             tmp0, zero, nccl_comm, stream);                                   \
    nccl_dot(local_result, lat_4dim12, val1, val1, tmp1, zero, nccl_comm,      \
             stream);                                                          \
    (*tmp) = (*tmp0) / (*tmp1);                                                \
  }

#define nccl_dslash_eo(dest_e, src_o, node_rank, gridDim, blockDim, gauge,     \
                       lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move, \
                       device_send_vec, device_recv_vec, zero, nccl_comm,      \
                       stream)                                                 \
  {                                                                            \
    device_zero_vec(lat_3dim6, device_send_vec, device_recv_vec,               \
                    host_send_vec, host_recv_vec, zero);                       \
    _ncclDslashQcu(gridDim, blockDim, gauge, src_o, dest_e, EVEN, lat_1dim,    \
                   lat_3dim12, node_rank, grid_1dim, grid_index_1dim, move,    \
                   device_send_vec, device_recv_vec, nccl_comm, stream);       \
  }

#define nccl_dslash_oe(dest_o, src_e, node_rank, gridDim, blockDim, gauge,     \
                       lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move, \
                       device_send_vec, device_recv_vec, zero, nccl_comm,      \
                       stream)                                                 \
  {                                                                            \
    device_zero_vec(lat_3dim6, device_send_vec, device_recv_vec,               \
                    host_send_vec, host_recv_vec, zero);                       \
    _ncclDslashQcu(gridDim, blockDim, gauge, src_e, dest_o, ODD, lat_1dim,     \
                   lat_3dim12, node_rank, grid_1dim, grid_index_1dim, move,    \
                   device_send_vec, device_recv_vec, nccl_comm, stream);       \
  }

// src_o-kappa**2*dslash_oe(dslash_eo(src_o))
#define nccl_dslash(dest_o, src_o, kappa, device_latt_tmp0, device_latt_tmp1,  \
                    node_rank, gridDim, blockDim, gauge, lat_1dim, lat_3dim12, \
                    lat_4dim12, grid_1dim, grid_index_1dim, move,              \
                    device_send_vec, device_recv_vec, zero, nccl_comm, stream) \
  {                                                                            \
    nccl_dslash_eo(device_latt_tmp0, src_o, node_rank, gridDim, blockDim,      \
                   gauge, lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim,    \
                   move, device_send_vec, device_recv_vec, zero, nccl_comm,    \
                   stream);                                                    \
    nccl_dslash_oe(device_latt_tmp1, device_latt_tmp0, node_rank, gridDim,     \
                   blockDim, gauge, lat_1dim, lat_3dim12, grid_1dim,           \
                   grid_index_1dim, move, device_send_vec, device_recv_vec,    \
                   zero, nccl_comm, stream);                                   \
    for (int i = 0; i < lat_4dim12; i++) {                                     \
      dest_o[i] = src_o[i] - device_latt_tmp1[i] * kappa * kappa;              \
    }                                                                          \
  }

#endif