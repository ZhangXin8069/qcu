#ifndef _DEFINE_H
#define _DEFINE_H
#include "lattice_complex.h"
#include <strings.h>

#include "./qcu.h"
#define _BLOCK_SIZE_ 256
#define _X_ 0
#define _Y_ 1
#define _Z_ 2
#define _T_ 3
#define _DIM_ 4
#define _B_X_ 0
#define _F_X_ 1
#define _B_Y_ 2
#define _F_Y_ 3
#define _B_Z_ 4
#define _F_Z_ 5
#define _B_T_ 6
#define _F_T_ 7
#define _WARDS_ 8
#define _YZT_ 0
#define _XZT_ 1
#define _XYT_ 2
#define _XYZ_ 3
#define _EVEN_ 0
#define _ODD_ 1
#define _EVENODD_ 2
#define _LAT_C_ 3
#define _LAT_S_ 4
#define _LAT_SC_ 12
#define _LAT_D_ 4
#define _B_ 0
#define _F_ 1
#define _BF_ 2
#define _OUTPUT_SIZE_ 10
#define _BACKWARD_ -1
#define _NOWARD_ 0
#define _FORWARD_ 1
#define _SR_ 2
#define _LAT_EXAMPLE_ 32
#define _GRID_EXAMPLE_ 1
#define WILSON_DSLASH
#define CLOVER_DSLASH
// #define OVERLAP_DSLASH
#define MPI_WILSON_DSLASH
#define MPI_CLOVER_DSLASH
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
template <typename LATTICE_TEMPLATE>

#define device_print(device_vec, host_vec, index, size, node_rank, tag)        \
  {                                                                            \
    int index_;                                                                \
    if (index < 0) {                                                           \
      index_ = size + index;                                                   \
    } else {                                                                   \
      index_ = index;                                                          \
    }                                                                          \
    cudaMemcpy(host_vec, device_vec, size * sizeof(LatticeComplex),            \
               cudaMemcpyDeviceToHost);                                        \
    print_ptr(host_vec, index_, node_rank, tag);                               \
  }

#define print_ptr(ptr, index, node_rank, tag)                                  \
  {                                                                            \
    checkCudaErrors(cudaDeviceSynchronize());                                  \
    printf("#%d#<%d>ptr(%p)[%d]:%.9lf + %.9lfi\n", tag, node_rank, ptr, index, \
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
    for (int i = 0; i < _DIM_; i++) {                                          \
      host_give_value(host_send_vec[i * _SR_], zero, lat_3dim6[i]);            \
      host_give_value(host_send_vec[i * _SR_ + 1], zero, lat_3dim6[i]);        \
      host_give_value(host_recv_vec[i * _SR_], zero, lat_3dim6[i]);            \
      host_give_value(host_recv_vec[i * _SR_ + 1], zero, lat_3dim6[i]);        \
    }                                                                          \
  }

#define device_zero_vec(lat_3dim6, device_send_vec, device_recv_vec,           \
                        host_send_vec, host_recv_vec, zero)                    \
  {                                                                            \
    host_zero_vec(lat_3dim6, host_send_vec, host_recv_vec, zero);              \
    for (int i = 0; i < _DIM_; i++) {                                          \
      cudaMemcpy(device_send_vec[i * _SR_], device_send_vec[i * _SR_],         \
                 sizeof(LatticeComplex) * lat_3dim6[i],                        \
                 cudaMemcpyHostToDevice);                                      \
      cudaMemcpy(device_send_vec[i * _SR_ + 1], device_send_vec[i * _SR_ + 1], \
                 sizeof(LatticeComplex) * lat_3dim6[i],                        \
                 cudaMemcpyHostToDevice);                                      \
      cudaMemcpy(device_recv_vec[i * _SR_], device_recv_vec[i * _SR_],         \
                 sizeof(LatticeComplex) * lat_3dim6[i],                        \
                 cudaMemcpyHostToDevice);                                      \
      cudaMemcpy(device_recv_vec[i * _SR_ + 1], device_recv_vec[i * _SR_ + 1], \
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
    lat_1dim[_X_] = param->lattice_size[_X_] >> 1;                             \
    lat_1dim[_Y_] = param->lattice_size[_Y_];                                  \
    lat_1dim[_Z_] = param->lattice_size[_Z_];                                  \
    lat_1dim[_T_] = param->lattice_size[_T_];                                  \
    lat_3dim[_YZT_] = lat_1dim[_Y_] * lat_1dim[_Z_] * lat_1dim[_T_];           \
    lat_3dim[_XZT_] = lat_1dim[_X_] * lat_1dim[_Z_] * lat_1dim[_T_];           \
    lat_3dim[_XYT_] = lat_1dim[_X_] * lat_1dim[_Y_] * lat_1dim[_T_];           \
    lat_3dim[_XYZ_] = lat_1dim[_X_] * lat_1dim[_Y_] * lat_1dim[_Z_];           \
    lat_4dim = lat_3dim[_XYZ_] * lat_1dim[_T_];                                \
  }

#define give_grid(grid, node_rank, grid_1dim, grid_index_1dim)                 \
  {                                                                            \
    MPI_Comm_rank(MPI_COMM_WORLD, &node_rank);                                 \
    grid_1dim[_X_] = grid->lattice_size[_X_];                                  \
    grid_1dim[_Y_] = grid->lattice_size[_Y_];                                  \
    grid_1dim[_Z_] = grid->lattice_size[_Z_];                                  \
    grid_1dim[_T_] = grid->lattice_size[_T_];                                  \
    grid_index_1dim[_X_] =                                                     \
        node_rank / grid_1dim[_T_] / grid_1dim[_Z_] / grid_1dim[_Y_];          \
    grid_index_1dim[_Y_] =                                                     \
        node_rank / grid_1dim[_T_] / grid_1dim[_Z_] % grid_1dim[_Y_];          \
    grid_index_1dim[_Z_] = node_rank / grid_1dim[_T_] % grid_1dim[_Z_];        \
    grid_index_1dim[_T_] = node_rank % grid_1dim[_T_];                         \
  }

#define malloc_vec(lat_3dim6, device_send_vec, device_recv_vec, host_send_vec, \
                   host_recv_vec)                                              \
  {                                                                            \
    for (int i = 0; i < _DIM_; i++) {                                          \
      cudaMalloc(&device_send_vec[i * _SR_],                                   \
                 lat_3dim6[i] * sizeof(LatticeComplex));                       \
      cudaMalloc(&device_send_vec[i * _SR_ + 1],                               \
                 lat_3dim6[i] * sizeof(LatticeComplex));                       \
      cudaMalloc(&device_recv_vec[i * _SR_],                                   \
                 lat_3dim6[i] * sizeof(LatticeComplex));                       \
      cudaMalloc(&device_recv_vec[i * _SR_ + 1],                               \
                 lat_3dim6[i] * sizeof(LatticeComplex));                       \
      host_send_vec[i * _SR_] =                                                \
          (void *)malloc(lat_3dim6[i] * sizeof(LatticeComplex));               \
      host_send_vec[i * _SR_ + 1] =                                            \
          (void *)malloc(lat_3dim6[i] * sizeof(LatticeComplex));               \
      host_recv_vec[i * _SR_] =                                                \
          (void *)malloc(lat_3dim6[i] * sizeof(LatticeComplex));               \
      host_recv_vec[i * _SR_ + 1] =                                            \
          (void *)malloc(lat_3dim6[i] * sizeof(LatticeComplex));               \
    }                                                                          \
  }

#define free_vec(device_send_vec, device_recv_vec, host_send_vec,              \
                 host_recv_vec)                                                \
  {                                                                            \
    for (int i = 0; i < _WARDS_; i++) {                                        \
      cudaFree(device_send_vec[i]);                                            \
      cudaFree(device_recv_vec[i]);                                            \
      free(host_send_vec[i]);                                                  \
      free(host_recv_vec[i]);                                                  \
    }                                                                          \
  }

#define _mpiDslashQcu(gridDim, blockDim, gauge, fermion_in, fermion_out,       \
                      parity, lat_1dim, lat_3dim12, node_rank, grid_1dim,      \
                      grid_index_1dim, move, send_request, recv_request,       \
                      device_send_vec, device_recv_vec, host_send_vec,         \
                      host_recv_vec)                                           \
  {                                                                            \
    wilson_dslash_clear_dest<<<gridDim, blockDim>>>(                           \
        fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_]);             \
    checkCudaErrors(cudaDeviceSynchronize());                                  \
    wilson_dslash_x_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[_X_], lat_1dim[_Y_],          \
        lat_1dim[_Z_], lat_1dim[_T_], parity, device_send_vec[_B_X_],          \
        device_send_vec[_F_X_]);                                               \
    cudaMemcpy(host_send_vec[_B_X_], device_send_vec[_B_X_],                   \
               sizeof(double) * lat_3dim12[_YZT_], cudaMemcpyDeviceToHost);    \
    cudaMemcpy(host_send_vec[_F_X_], device_send_vec[_F_X_],                   \
               sizeof(double) * lat_3dim12[_YZT_], cudaMemcpyDeviceToHost);    \
    if (grid_1dim[_X_] != 1) {                                                 \
      move_backward(move[_B_], grid_index_1dim[_X_], grid_1dim[_X_]);          \
      move_forward(move[_F_], grid_index_1dim[_X_], grid_1dim[_X_]);           \
      move[_B_] = node_rank + move[_B_] * grid_1dim[_Y_] * grid_1dim[_Z_] *    \
                                  grid_1dim[_T_];                              \
      move[_F_] = node_rank + move[_F_] * grid_1dim[_Y_] * grid_1dim[_Z_] *    \
                                  grid_1dim[_T_];                              \
      MPI_Irecv(host_recv_vec[_B_X_], lat_3dim12[_YZT_], MPI_DOUBLE,           \
                move[_B_], _F_X_, MPI_COMM_WORLD, &recv_request[_B_X_]);       \
      MPI_Irecv(host_recv_vec[_F_X_], lat_3dim12[_YZT_], MPI_DOUBLE,           \
                move[_F_], _B_X_, MPI_COMM_WORLD, &recv_request[_F_X_]);       \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      MPI_Isend(host_send_vec[_B_X_], lat_3dim12[_YZT_], MPI_DOUBLE,           \
                move[_B_], _B_X_, MPI_COMM_WORLD, &send_request[_B_X_]);       \
      MPI_Isend(host_send_vec[_F_X_], lat_3dim12[_YZT_], MPI_DOUBLE,           \
                move[_F_], _F_X_, MPI_COMM_WORLD, &send_request[_F_T_]);       \
    }                                                                          \
    wilson_dslash_y_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[_X_], lat_1dim[_Y_],          \
        lat_1dim[_Z_], lat_1dim[_T_], parity, device_send_vec[_B_Y_],          \
        device_send_vec[_F_Y_]);                                               \
    cudaMemcpy(host_send_vec[_B_Y_], device_send_vec[_B_Y_],                   \
               sizeof(double) * lat_3dim12[_XZT_], cudaMemcpyDeviceToHost);    \
    cudaMemcpy(host_send_vec[_F_Y_], device_send_vec[_F_Y_],                   \
               sizeof(double) * lat_3dim12[_XZT_], cudaMemcpyDeviceToHost);    \
    if (grid_1dim[_Y_] != 1) {                                                 \
      move_backward(move[_B_], grid_index_1dim[_Y_], grid_1dim[_Y_]);          \
      move_forward(move[_F_], grid_index_1dim[_Y_], grid_1dim[_Y_]);           \
      move[_B_] = node_rank + move[_B_] * grid_1dim[_Z_] * grid_1dim[_T_];     \
      move[_F_] = node_rank + move[_F_] * grid_1dim[_Z_] * grid_1dim[_T_];     \
      MPI_Irecv(host_recv_vec[_B_Y_], lat_3dim12[_XZT_], MPI_DOUBLE,           \
                move[_B_], _F_Y_, MPI_COMM_WORLD, &recv_request[_B_Y_]);       \
      MPI_Irecv(host_recv_vec[_F_Y_], lat_3dim12[_XZT_], MPI_DOUBLE,           \
                move[_F_], _B_Y_, MPI_COMM_WORLD, &recv_request[_F_Y_]);       \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      MPI_Isend(host_send_vec[_B_Y_], lat_3dim12[_XZT_], MPI_DOUBLE,           \
                move[_B_], _B_Y_, MPI_COMM_WORLD, &send_request[_B_Y_]);       \
      MPI_Isend(host_send_vec[_F_Y_], lat_3dim12[_XZT_], MPI_DOUBLE,           \
                move[_F_], _F_Y_, MPI_COMM_WORLD, &send_request[_F_Y_]);       \
    }                                                                          \
    wilson_dslash_z_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[_X_], lat_1dim[_Y_],          \
        lat_1dim[_Z_], lat_1dim[_T_], parity, device_send_vec[_B_Z_],          \
        device_send_vec[_F_Z_]);                                               \
    cudaMemcpy(host_send_vec[_B_Z_], device_send_vec[_B_Z_],                   \
               sizeof(double) * lat_3dim12[_XYT_], cudaMemcpyDeviceToHost);    \
    cudaMemcpy(host_send_vec[_F_Z_], device_send_vec[_F_Z_],                   \
               sizeof(double) * lat_3dim12[_XYT_], cudaMemcpyDeviceToHost);    \
    if (grid_1dim[_Z_] != 1) {                                                 \
      move_backward(move[_B_], grid_index_1dim[_Z_], grid_1dim[_Z_]);          \
      move_forward(move[_F_], grid_index_1dim[_Z_], grid_1dim[_Z_]);           \
      move[_B_] = node_rank + move[_B_] * grid_1dim[_T_];                      \
      move[_F_] = node_rank + move[_F_] * grid_1dim[_T_];                      \
      MPI_Irecv(host_recv_vec[_B_Z_], lat_3dim12[_XYT_], MPI_DOUBLE,           \
                move[_B_], _F_Z_, MPI_COMM_WORLD, &recv_request[_B_Z_]);       \
      MPI_Irecv(host_recv_vec[_F_Z_], lat_3dim12[_XYT_], MPI_DOUBLE,           \
                move[_F_], _B_Z_, MPI_COMM_WORLD, &recv_request[_F_Z_]);       \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      MPI_Isend(host_send_vec[_B_Z_], lat_3dim12[_XYT_], MPI_DOUBLE,           \
                move[_B_], _B_Z_, MPI_COMM_WORLD, &send_request[_B_Z_]);       \
      MPI_Isend(host_send_vec[_F_Z_], lat_3dim12[_XYT_], MPI_DOUBLE,           \
                move[_F_], _F_Z_, MPI_COMM_WORLD, &send_request[_F_Z_]);       \
    }                                                                          \
    wilson_dslash_t_send<<<gridDim, blockDim>>>(                               \
        gauge, fermion_in, fermion_out, lat_1dim[_X_], lat_1dim[_Y_],          \
        lat_1dim[_Z_], lat_1dim[_T_], parity, device_send_vec[_B_T_],          \
        device_send_vec[_F_T_]);                                               \
    cudaMemcpy(host_send_vec[_B_T_], device_send_vec[_B_T_],                   \
               sizeof(double) * lat_3dim12[_XYZ_], cudaMemcpyDeviceToHost);    \
    cudaMemcpy(host_send_vec[_F_T_], device_send_vec[_F_T_],                   \
               sizeof(double) * lat_3dim12[_XYZ_], cudaMemcpyDeviceToHost);    \
    if (grid_1dim[_T_] != 1) {                                                 \
      move_backward(move[_B_], grid_index_1dim[_T_], grid_1dim[_T_]);          \
      move_forward(move[_F_], grid_index_1dim[_T_], grid_1dim[_T_]);           \
      move[_B_] = node_rank + move[_B_];                                       \
      move[_F_] = node_rank + move[_F_];                                       \
      MPI_Irecv(host_recv_vec[_B_T_], lat_3dim12[_XYZ_], MPI_DOUBLE,           \
                move[_B_], _F_T_, MPI_COMM_WORLD, &recv_request[_B_T_]);       \
      MPI_Irecv(host_recv_vec[_F_T_], lat_3dim12[_XYZ_], MPI_DOUBLE,           \
                move[_F_], _B_T_, MPI_COMM_WORLD, &recv_request[_F_T_]);       \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      MPI_Isend(host_send_vec[_B_T_], lat_3dim12[_XYZ_], MPI_DOUBLE,           \
                move[_B_], _B_T_, MPI_COMM_WORLD, &send_request[_B_T_]);       \
      MPI_Isend(host_send_vec[_F_T_], lat_3dim12[_XYZ_], MPI_DOUBLE,           \
                move[_F_], _F_T_, MPI_COMM_WORLD, &send_request[_F_T_]);       \
    }                                                                          \
    if (grid_1dim[_X_] != 1) {                                                 \
      MPI_Wait(&recv_request[_B_X_], MPI_STATUS_IGNORE);                       \
      MPI_Wait(&recv_request[_F_X_], MPI_STATUS_IGNORE);                       \
      cudaMemcpy(device_recv_vec[_B_X_], host_recv_vec[_B_X_],                 \
                 sizeof(double) * lat_3dim12[_YZT_], cudaMemcpyHostToDevice);  \
      cudaMemcpy(device_recv_vec[_F_X_], host_recv_vec[_F_X_],                 \
                 sizeof(double) * lat_3dim12[_YZT_], cudaMemcpyHostToDevice);  \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_x_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_recv_vec[_B_X_],                       \
          device_recv_vec[_F_X_]);                                             \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_x_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_send_vec[_F_X_],                       \
          device_send_vec[_B_X_]);                                             \
    }                                                                          \
    if (grid_1dim[_Y_] != 1) {                                                 \
      MPI_Wait(&recv_request[_B_Y_], MPI_STATUS_IGNORE);                       \
      MPI_Wait(&recv_request[_F_Y_], MPI_STATUS_IGNORE);                       \
      cudaMemcpy(device_recv_vec[_B_Y_], host_recv_vec[_B_Y_],                 \
                 sizeof(double) * lat_3dim12[_XZT_], cudaMemcpyHostToDevice);  \
      cudaMemcpy(device_recv_vec[_F_Y_], host_recv_vec[_F_Y_],                 \
                 sizeof(double) * lat_3dim12[_XZT_], cudaMemcpyHostToDevice);  \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_y_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_recv_vec[_B_Y_],                       \
          device_recv_vec[_F_Y_]);                                             \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_y_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_send_vec[_F_Y_],                       \
          device_send_vec[_B_Y_]);                                             \
    }                                                                          \
    if (grid_1dim[_Z_] != 1) {                                                 \
      MPI_Wait(&recv_request[_B_Z_], MPI_STATUS_IGNORE);                       \
      MPI_Wait(&recv_request[_F_Z_], MPI_STATUS_IGNORE);                       \
      cudaMemcpy(device_recv_vec[_B_Z_], host_recv_vec[_B_Z_],                 \
                 sizeof(double) * lat_3dim12[_XYT_], cudaMemcpyHostToDevice);  \
      cudaMemcpy(device_recv_vec[_F_Z_], host_recv_vec[_F_Z_],                 \
                 sizeof(double) * lat_3dim12[_XYT_], cudaMemcpyHostToDevice);  \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_z_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_recv_vec[_B_Z_],                       \
          device_recv_vec[_F_Z_]);                                             \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_z_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_send_vec[_F_Z_],                       \
          device_send_vec[_B_Z_]);                                             \
    }                                                                          \
    if (grid_1dim[_T_] != 1) {                                                 \
      MPI_Wait(&recv_request[_B_T_], MPI_STATUS_IGNORE);                       \
      MPI_Wait(&recv_request[_F_T_], MPI_STATUS_IGNORE);                       \
      cudaMemcpy(device_recv_vec[_B_T_], host_recv_vec[_B_T_],                 \
                 sizeof(double) * lat_3dim12[_XYZ_], cudaMemcpyHostToDevice);  \
      cudaMemcpy(device_recv_vec[_F_T_], host_recv_vec[_F_T_],                 \
                 sizeof(double) * lat_3dim12[_XYZ_], cudaMemcpyHostToDevice);  \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_t_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_recv_vec[_B_T_],                       \
          device_recv_vec[_F_T_]);                                             \
    } else {                                                                   \
      checkCudaErrors(cudaDeviceSynchronize());                                \
      wilson_dslash_t_recv<<<gridDim, blockDim>>>(                             \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_send_vec[_F_T_],                       \
          device_send_vec[_B_T_]);                                             \
    }                                                                          \
    MPI_Barrier(MPI_COMM_WORLD);                                               \
    checkCudaErrors(cudaDeviceSynchronize());                                  \
  }

#define mpi_dot(device_dot_tmp, host_dot_tmp, val0, val1, tmp, gridDim,        \
                blockDim)                                                      \
  {                                                                            \
    LatticeComplex local_result(0.0, 0.0);                                     \
    int lat_4dim = gridDim.x * blockDim.x;                                     \
    wilson_bistabcg_part_dot<<<gridDim, blockDim>>>(device_dot_tmp, val0,      \
                                                    val1);                     \
    cudaMemcpy(host_dot_tmp, device_dot_tmp,                                   \
               sizeof(LatticeComplex) * lat_4dim, cudaMemcpyDeviceToHost);     \
    checkCudaErrors(cudaDeviceSynchronize());                                  \
    for (int i = 0; i < lat_4dim; i++) {                                       \
      local_result += host_dot_tmp[i];                                         \
    }                                                                          \
    MPI_Allreduce(&local_result, &tmp, 2, MPI_DOUBLE, MPI_SUM,                 \
                  MPI_COMM_WORLD);                                             \
    MPI_Barrier(MPI_COMM_WORLD);                                               \
  }

#define mpi_diff(device_dot_tmp, host_dot_tmp, val0, val1, tmp,                \
                 device_latt_tmp0, tmp0, tmp1, gridDim, blockDim)              \
  {                                                                            \
    wilson_bistabcg_part_cut<<<gridDim, blockDim>>>(device_latt_tmp0, val0,    \
                                                    val1);                     \
    checkCudaErrors(cudaDeviceSynchronize());                                  \
    mpi_dot(device_dot_tmp, host_dot_tmp, device_latt_tmp0, device_latt_tmp0,  \
            tmp0, gridDim, blockDim);                                          \
    mpi_dot(device_dot_tmp, host_dot_tmp, val1, val1, tmp1, gridDim,           \
            blockDim);                                                         \
    tmp = tmp0 / tmp1;                                                         \
  }

#define mpi_dslash_eo(dest_e, src_o, node_rank, gridDim, blockDim, gauge,      \
                      lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move,  \
                      send_request, recv_request, device_send_vec,             \
                      device_recv_vec, host_send_vec, host_recv_vec)           \
  {                                                                            \
    _mpiDslashQcu(gridDim, blockDim, gauge, src_o, dest_e, _EVEN_, lat_1dim,   \
                  lat_3dim12, node_rank, grid_1dim, grid_index_1dim, move,     \
                  send_request, recv_request, device_send_vec,                 \
                  device_recv_vec, host_send_vec, host_recv_vec);              \
  }

#define mpi_dslash_oe(dest_o, src_e, node_rank, gridDim, blockDim, gauge,      \
                      lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move,  \
                      send_request, recv_request, device_send_vec,             \
                      device_recv_vec, host_send_vec, host_recv_vec)           \
  {                                                                            \
    _mpiDslashQcu(gridDim, blockDim, gauge, src_e, dest_o, _ODD_, lat_1dim,    \
                  lat_3dim12, node_rank, grid_1dim, grid_index_1dim, move,     \
                  send_request, recv_request, device_send_vec,                 \
                  device_recv_vec, host_send_vec, host_recv_vec);              \
  }

// src_o-kappa**2*dslash_oe(dslash_eo(src_o))
#define mpi_dslash(dest_o, src_o, kappa, device_latt_tmp0, device_latt_tmp1,   \
                   node_rank, gridDim, blockDim, gauge, lat_1dim, lat_3dim12,  \
                   lat_4dim12, grid_1dim, grid_index_1dim, move, send_request, \
                   recv_request, device_send_vec, device_recv_vec,             \
                   host_send_vec, host_recv_vec)                               \
  {                                                                            \
    mpi_dslash_eo(device_latt_tmp0, src_o, node_rank, gridDim, blockDim,       \
                  gauge, lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim,     \
                  move, send_request, recv_request, device_send_vec,           \
                  device_recv_vec, host_send_vec, host_recv_vec);              \
    mpi_dslash_oe(device_latt_tmp1, device_latt_tmp0, node_rank, gridDim,      \
                  blockDim, gauge, lat_1dim, lat_3dim12, grid_1dim,            \
                  grid_index_1dim, move, send_request, recv_request,           \
                  device_send_vec, device_recv_vec, host_send_vec,             \
                  host_recv_vec);                                              \
    wilson_bistabcg_give_dest_o<<<gridDim, blockDim>>>(                        \
        dest_o, src_o, device_latt_tmp1, kappa);                               \
  }

#define _ncclDslashQcu(gridDim, blockDim, gauge, fermion_in, fermion_out,      \
                       parity, lat_1dim, lat_3dim12, node_rank, grid_1dim,     \
                       grid_index_1dim, move, device_send_vec,                 \
                       device_recv_vec, qcu_nccl_comm, qcu_stream)             \
  {                                                                            \
    ncclGroupStart();                                                          \
    wilson_dslash_clear_dest<<<gridDim, blockDim, 0, qcu_stream>>>(            \
        fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_]);             \
    cudaStreamSynchronize(qcu_stream);                                         \
    wilson_dslash_x_send<<<gridDim, blockDim, 0, qcu_stream>>>(                \
        gauge, fermion_in, fermion_out, lat_1dim[_X_], lat_1dim[_Y_],          \
        lat_1dim[_Z_], lat_1dim[_T_], parity, device_send_vec[_B_X_],          \
        device_send_vec[_F_X_]);                                               \
    if (grid_1dim[_X_] != 1) {                                                 \
      move_backward(move[_B_], grid_index_1dim[_X_], grid_1dim[_X_]);          \
      move_forward(move[_F_], grid_index_1dim[_X_], grid_1dim[_X_]);           \
      move[_B_] = node_rank + move[_B_] * grid_1dim[_Y_] * grid_1dim[_Z_] *    \
                                  grid_1dim[_T_];                              \
      move[_F_] = node_rank + move[_F_] * grid_1dim[_Y_] * grid_1dim[_Z_] *    \
                                  grid_1dim[_T_];                              \
      cudaStreamSynchronize(qcu_stream);                                       \
      ncclSend(device_send_vec[_B_X_], lat_3dim12[_YZT_], ncclDouble,          \
               move[_B_], qcu_nccl_comm, qcu_stream);                          \
      ncclRecv(device_recv_vec[_F_X_], lat_3dim12[_YZT_], ncclDouble,          \
               move[_F_], qcu_nccl_comm, qcu_stream);                          \
      ncclSend(device_send_vec[_F_X_], lat_3dim12[_YZT_], ncclDouble,          \
               move[_F_], qcu_nccl_comm, qcu_stream);                          \
      ncclRecv(device_recv_vec[_B_X_], lat_3dim12[_YZT_], ncclDouble,          \
               move[_B_], qcu_nccl_comm, qcu_stream);                          \
    }                                                                          \
    wilson_dslash_y_send<<<gridDim, blockDim, 0, qcu_stream>>>(                \
        gauge, fermion_in, fermion_out, lat_1dim[_X_], lat_1dim[_Y_],          \
        lat_1dim[_Z_], lat_1dim[_T_], parity, device_send_vec[_B_Y_],          \
        device_send_vec[_F_Y_]);                                               \
    if (grid_1dim[_Y_] != 1) {                                                 \
      move_backward(move[_B_], grid_index_1dim[_Y_], grid_1dim[_Y_]);          \
      move_forward(move[_F_], grid_index_1dim[_Y_], grid_1dim[_Y_]);           \
      move[_B_] = node_rank + move[_B_] * grid_1dim[_Z_] * grid_1dim[_T_];     \
      move[_F_] = node_rank + move[_F_] * grid_1dim[_Z_] * grid_1dim[_T_];     \
      cudaStreamSynchronize(qcu_stream);                                       \
      ncclSend(device_send_vec[_B_Y_], lat_3dim12[_XZT_], ncclDouble,          \
               move[_B_], qcu_nccl_comm, qcu_stream);                          \
      ncclRecv(device_recv_vec[_F_Y_], lat_3dim12[_XZT_], ncclDouble,          \
               move[_F_], qcu_nccl_comm, qcu_stream);                          \
      ncclSend(device_send_vec[_F_Y_], lat_3dim12[_XZT_], ncclDouble,          \
               move[_F_], qcu_nccl_comm, qcu_stream);                          \
      ncclRecv(device_recv_vec[_B_Y_], lat_3dim12[_XZT_], ncclDouble,          \
               move[_B_], qcu_nccl_comm, qcu_stream);                          \
    }                                                                          \
    wilson_dslash_z_send<<<gridDim, blockDim, 0, qcu_stream>>>(                \
        gauge, fermion_in, fermion_out, lat_1dim[_X_], lat_1dim[_Y_],          \
        lat_1dim[_Z_], lat_1dim[_T_], parity, device_send_vec[_B_Z_],          \
        device_send_vec[_F_Z_]);                                               \
    if (grid_1dim[_Z_] != 1) {                                                 \
      move_backward(move[_B_], grid_index_1dim[_Z_], grid_1dim[_Z_]);          \
      move_forward(move[_F_], grid_index_1dim[_Z_], grid_1dim[_Z_]);           \
      move[_B_] = node_rank + move[_B_] * grid_1dim[_T_];                      \
      move[_F_] = node_rank + move[_F_] * grid_1dim[_T_];                      \
      cudaStreamSynchronize(qcu_stream);                                       \
      ncclSend(device_send_vec[_B_Z_], lat_3dim12[_XYT_], ncclDouble,          \
               move[_B_], qcu_nccl_comm, qcu_stream);                          \
      ncclRecv(device_recv_vec[_F_Z_], lat_3dim12[_XYT_], ncclDouble,          \
               move[_F_], qcu_nccl_comm, qcu_stream);                          \
      ncclSend(device_send_vec[_F_Z_], lat_3dim12[_XYT_], ncclDouble,          \
               move[_F_], qcu_nccl_comm, qcu_stream);                          \
      ncclRecv(device_recv_vec[_B_Z_], lat_3dim12[_XYT_], ncclDouble,          \
               move[_B_], qcu_nccl_comm, qcu_stream);                          \
    }                                                                          \
    wilson_dslash_t_send<<<gridDim, blockDim, 0, qcu_stream>>>(                \
        gauge, fermion_in, fermion_out, lat_1dim[_X_], lat_1dim[_Y_],          \
        lat_1dim[_Z_], lat_1dim[_T_], parity, device_send_vec[_B_T_],          \
        device_send_vec[_F_T_]);                                               \
    if (grid_1dim[_T_] != 1) {                                                 \
      move_backward(move[_B_], grid_index_1dim[_T_], grid_1dim[_T_]);          \
      move_forward(move[_F_], grid_index_1dim[_T_], grid_1dim[_T_]);           \
      move[_B_] = node_rank + move[_B_];                                       \
      move[_F_] = node_rank + move[_F_];                                       \
      cudaStreamSynchronize(qcu_stream);                                       \
      ncclSend(device_send_vec[_B_T_], lat_3dim12[_XYZ_], ncclDouble,          \
               move[_B_], qcu_nccl_comm, qcu_stream);                          \
      ncclRecv(device_recv_vec[_F_T_], lat_3dim12[_XYZ_], ncclDouble,          \
               move[_F_], qcu_nccl_comm, qcu_stream);                          \
      ncclSend(device_send_vec[_F_T_], lat_3dim12[_XYZ_], ncclDouble,          \
               move[_F_], qcu_nccl_comm, qcu_stream);                          \
      ncclRecv(device_recv_vec[_B_T_], lat_3dim12[_XYZ_], ncclDouble,          \
               move[_B_], qcu_nccl_comm, qcu_stream);                          \
    }                                                                          \
    checkCudaErrors(cudaStreamSynchronize(qcu_stream));                        \
    ncclGroupEnd();                                                            \
    if (grid_1dim[_X_] != 1) {                                                 \
      wilson_dslash_x_recv<<<gridDim, blockDim, 0, qcu_stream>>>(              \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_recv_vec[_B_X_],                       \
          device_recv_vec[_F_X_]);                                             \
    } else {                                                                   \
      wilson_dslash_x_recv<<<gridDim, blockDim, 0, qcu_stream>>>(              \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_send_vec[_F_X_],                       \
          device_send_vec[_B_X_]);                                             \
    }                                                                          \
    if (grid_1dim[_Y_] != 1) {                                                 \
      wilson_dslash_y_recv<<<gridDim, blockDim, 0, qcu_stream>>>(              \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_recv_vec[_B_Y_],                       \
          device_recv_vec[_F_Y_]);                                             \
    } else {                                                                   \
      wilson_dslash_y_recv<<<gridDim, blockDim, 0, qcu_stream>>>(              \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_send_vec[_F_Y_],                       \
          device_send_vec[_B_Y_]);                                             \
    }                                                                          \
    if (grid_1dim[_Z_] != 1) {                                                 \
      wilson_dslash_z_recv<<<gridDim, blockDim, 0, qcu_stream>>>(              \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_recv_vec[_B_Z_],                       \
          device_recv_vec[_F_Z_]);                                             \
    } else {                                                                   \
      wilson_dslash_z_recv<<<gridDim, blockDim, 0, qcu_stream>>>(              \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_send_vec[_F_Z_],                       \
          device_send_vec[_B_Z_]);                                             \
    }                                                                          \
    if (grid_1dim[_T_] != 1) {                                                 \
      wilson_dslash_t_recv<<<gridDim, blockDim, 0, qcu_stream>>>(              \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_recv_vec[_B_T_],                       \
          device_recv_vec[_F_T_]);                                             \
    } else {                                                                   \
      wilson_dslash_t_recv<<<gridDim, blockDim, 0, qcu_stream>>>(              \
          gauge, fermion_out, lat_1dim[_X_], lat_1dim[_Y_], lat_1dim[_Z_],     \
          lat_1dim[_T_], parity, device_send_vec[_F_T_],                       \
          device_send_vec[_B_T_]);                                             \
    }                                                                          \
    checkCudaErrors(cudaStreamSynchronize(qcu_stream));                        \
  }

#define nccl_dot(device_dot_tmp, host_dot_tmp, val0, val1, tmp, gridDim,       \
                 blockDim)                                                     \
  { mpi_dot(device_dot_tmp, host_dot_tmp, val0, val1, tmp, gridDim, blockDim); }
#define nccl_diff(device_dot_tmp, host_dot_tmp, val0, val1, tmp,               \
                  device_latt_tmp0, tmp0, tmp1, gridDim, blockDim)             \
  {                                                                            \
    mpi_diff(device_dot_tmp, host_dot_tmp, val0, val1, tmp, device_latt_tmp0,  \
             tmp0, tmp1, gridDim, blockDim);                                   \
  }

#define nccl_dslash_eo(dest_e, src_o, node_rank, gridDim, blockDim, gauge,     \
                       lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move, \
                       device_send_vec, device_recv_vec, qcu_nccl_comm,        \
                       qcu_stream)                                             \
  {                                                                            \
    _ncclDslashQcu(gridDim, blockDim, gauge, src_o, dest_e, _EVEN_, lat_1dim,  \
                   lat_3dim12, node_rank, grid_1dim, grid_index_1dim, move,    \
                   device_send_vec, device_recv_vec, qcu_nccl_comm,            \
                   qcu_stream);                                                \
  }

#define nccl_dslash_oe(dest_o, src_e, node_rank, gridDim, blockDim, gauge,     \
                       lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move, \
                       device_send_vec, device_recv_vec, qcu_nccl_comm,        \
                       qcu_stream)                                             \
  {                                                                            \
    _ncclDslashQcu(gridDim, blockDim, gauge, src_e, dest_o, _ODD_, lat_1dim,   \
                   lat_3dim12, node_rank, grid_1dim, grid_index_1dim, move,    \
                   device_send_vec, device_recv_vec, qcu_nccl_comm,            \
                   qcu_stream);                                                \
  }

// src_o-kappa**2*dslash_oe(dslash_eo(src_o))
#define nccl_dslash(dest_o, src_o, kappa, device_latt_tmp0, device_latt_tmp1,  \
                    node_rank, gridDim, blockDim, gauge, lat_1dim, lat_3dim12, \
                    lat_4dim12, grid_1dim, grid_index_1dim, move,              \
                    device_send_vec, device_recv_vec, qcu_nccl_comm,           \
                    qcu_stream)                                                \
  {                                                                            \
    nccl_dslash_eo(device_latt_tmp0, src_o, node_rank, gridDim, blockDim,      \
                   gauge, lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim,    \
                   move, device_send_vec, device_recv_vec, qcu_nccl_comm,      \
                   qcu_stream);                                                \
    nccl_dslash_oe(device_latt_tmp1, device_latt_tmp0, node_rank, gridDim,     \
                   blockDim, gauge, lat_1dim, lat_3dim12, grid_1dim,           \
                   grid_index_1dim, move, device_send_vec, device_recv_vec,    \
                   qcu_nccl_comm, qcu_stream);                                 \
    wilson_bistabcg_give_dest_o<<<gridDim, blockDim>>>(                        \
        dest_o, src_o, device_latt_tmp1, kappa);                               \
  }

#endif