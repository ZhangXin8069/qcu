#ifndef _DEFINE_H
#define _DEFINE_H
#include "./lattice_complex.h"

#define _BLOCK_SIZE_ 256
#define _WARP_SIZE_ 32
#define _a_ 0
#define _b_ 1
#define _c_ 2
#define _d_ 3
#define _tmp0_ 0
#define _tmp1_ 1
#define _rho_prev_ 2
#define _rho_ 3
#define _alpha_ 4
#define _beta_ 5
#define _omega_ 6
#define _send_tmp_ 7
#define _norm2_tmp_ 8
#define _diff_tmp_ 9
#define _vals_size_ 10
#define _X_ 0
#define _Y_ 1
#define _Z_ 2
#define _T_ 3
#define _XCC_ 4
#define _YXCC_ 5
#define _ZYXCC_ 6
#define _TZYXCC_ 7
#define _XSC_ 8
#define _YXSC_ 9
#define _ZYXSC_ 10
#define _TZYXSC_ 11
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
#define _LAT_CC_ 9
#define _LAT_1C_ 3
#define _LAT_2C_ 6
#define _LAT_3C_ 9
#define _LAT_HALF_SC_ 6
#define _LAT_SC_ 12
#define _LAT_SCSC_ 144
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
#define _MAX_ITER_ 1e2
#define _TOL_ 1e-6
#define _KAPPA_ 0.125
#define _MEM_POOL_ 0
#define _CHECK_ERROR_ 1
#define DRAFT
#define LATTICE_CUDA
#define BISTABCG
#define MULTGRID
#define WILSON_DSLASH
#define CLOVER_DSLASH
// #define OVERLAP_DSLASH
#define NCCL_WILSON_DSLASH
// #define NCCL_CLOVER_DSLASH
// #define NCCL_OVERLAP_DSLASH
#define WILSON_BISTABCG
// #define CLOVER_BISTABCG
// #define OVERLAP_BISTABCG
#define NCCL_WILSON_BISTABCG
// #define NCCL_CLOVER_BISTABCG
// #define NCCL_OVERLAP_BISTABCG
// #define WILSON_MULTGRID
// #define CLOVER_MULTGRID
// #define OVERLAP_MULTGRID
// #define NCCL_WILSON_MULTGRID
// #define NCCL_CLOVER_MULTGRID
// #define NCCL_OVERLAP_MULTGRID

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
    if (_CHECK_ERROR_) {                                                       \
      if (err != cudaSuccess) {                                                \
        fprintf(stderr,                                                        \
                "Failed: CUDA error %04d \"%s\" from file <%s>, "              \
                "line %i.\n",                                                  \
                err, cudaGetErrorString(err), __FILE__, __LINE__);             \
        exit(EXIT_FAILURE);                                                    \
      }                                                                        \
    }                                                                          \
  }

#define checkMpiErrors(err)                                                    \
  {                                                                            \
    if (_CHECK_ERROR_) {                                                       \
      if (err != MPI_SUCCESS) {                                                \
        fprintf(stderr,                                                        \
                "Failed: MPI error %04d from file <%s>, "                      \
                "line %i.\n",                                                  \
                err, __FILE__, __LINE__);                                      \
        exit(EXIT_FAILURE);                                                    \
      }                                                                        \
    }                                                                          \
  }

#define checkNcclErrors(err)                                                   \
  {                                                                            \
    if (_CHECK_ERROR_) {                                                       \
      if (err != ncclSuccess) {                                                \
        fprintf(stderr,                                                        \
                "Failed: NCCL error %04d \"%s\" from file <%s>, "              \
                "line %i.\n",                                                  \
                err, ncclGetErrorString(err), __FILE__, __LINE__);             \
        exit(EXIT_FAILURE);                                                    \
      }                                                                        \
    }                                                                          \
  }

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

#define host_zero_vec(lat_3dim_Half_SC, host_send_vec, host_recv_vec, zero)    \
  {                                                                            \
    for (int i = 0; i < _DIM_; i++) {                                          \
      host_give_value(host_send_vec[i * _SR_], zero, lat_3dim_Half_SC[i]);     \
      host_give_value(host_send_vec[i * _SR_ + 1], zero, lat_3dim_Half_SC[i]); \
      host_give_value(host_recv_vec[i * _SR_], zero, lat_3dim_Half_SC[i]);     \
      host_give_value(host_recv_vec[i * _SR_ + 1], zero, lat_3dim_Half_SC[i]); \
    }                                                                          \
  }

#define device_zero_vec(lat_3dim_Half_SC, device_send_vec, device_recv_vec,    \
                        host_send_vec, host_recv_vec, zero)                    \
  {                                                                            \
    host_zero_vec(lat_3dim_Half_SC, host_send_vec, host_recv_vec, zero);       \
    for (int i = 0; i < _DIM_; i++) {                                          \
      cudaMemcpy(device_send_vec[i * _SR_], device_send_vec[i * _SR_],         \
                 sizeof(LatticeComplex) * lat_3dim_Half_SC[i],                 \
                 cudaMemcpyHostToDevice);                                      \
      cudaMemcpy(device_send_vec[i * _SR_ + 1], device_send_vec[i * _SR_ + 1], \
                 sizeof(LatticeComplex) * lat_3dim_Half_SC[i],                 \
                 cudaMemcpyHostToDevice);                                      \
      cudaMemcpy(device_recv_vec[i * _SR_], device_recv_vec[i * _SR_],         \
                 sizeof(LatticeComplex) * lat_3dim_Half_SC[i],                 \
                 cudaMemcpyHostToDevice);                                      \
      cudaMemcpy(device_recv_vec[i * _SR_ + 1], device_recv_vec[i * _SR_ + 1], \
                 sizeof(LatticeComplex) * lat_3dim_Half_SC[i],                 \
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
    for (int c0 = 0; c0 < _LAT_C_; c0++) {                                     \
      for (int c1 = 0; c1 < _LAT_C_; c1++) {                                   \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < _LAT_C_; cc++) {                                 \
          tmp0 += tmp1[c0 * _LAT_C_ + cc] * tmp2[cc * _LAT_C_ + c1];           \
        }                                                                      \
        tmp3[c0 * _LAT_C_ + c1] = tmp0;                                        \
      }                                                                        \
    }                                                                          \
  }

#define mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero)                          \
  {                                                                            \
    for (int c0 = 0; c0 < _LAT_C_; c0++) {                                     \
      for (int c1 = 0; c1 < _LAT_C_; c1++) {                                   \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < _LAT_C_; cc++) {                                 \
          tmp0 += tmp1[c0 * _LAT_C_ + cc] * tmp2[c1 * _LAT_C_ + cc].conj();    \
        }                                                                      \
        tmp3[c0 * _LAT_C_ + c1] = tmp0;                                        \
      }                                                                        \
    }                                                                          \
  }

#define mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero)                          \
  {                                                                            \
    for (int c0 = 0; c0 < _LAT_C_; c0++) {                                     \
      for (int c1 = 0; c1 < _LAT_C_; c1++) {                                   \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < _LAT_C_; cc++) {                                 \
          tmp0 += tmp1[cc * _LAT_C_ + c0].conj() * tmp2[cc * _LAT_C_ + c1];    \
        }                                                                      \
        tmp3[c0 * _LAT_C_ + c1] = tmp0;                                        \
      }                                                                        \
    }                                                                          \
  }

#define mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero)                           \
  {                                                                            \
    for (int c0 = 0; c0 < _LAT_C_; c0++) {                                     \
      for (int c1 = 0; c1 < _LAT_C_; c1++) {                                   \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < _LAT_C_; cc++) {                                 \
          tmp0 +=                                                              \
              tmp1[cc * _LAT_C_ + c0].conj() * tmp2[c1 * _LAT_C_ + cc].conj(); \
        }                                                                      \
        tmp3[c0 * _LAT_C_ + c1] = tmp0;                                        \
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

#define malloc_vec(lat_3dim_Half_SC, device_send_vec, device_recv_vec,         \
                   host_send_vec, host_recv_vec)                               \
  {                                                                            \
    for (int i = 0; i < _DIM_; i++) {                                          \
      cudaMalloc(&device_send_vec[i * _SR_],                                   \
                 lat_3dim_Half_SC[i] * sizeof(LatticeComplex));                \
      cudaMalloc(&device_send_vec[i * _SR_ + 1],                               \
                 lat_3dim_Half_SC[i] * sizeof(LatticeComplex));                \
      cudaMalloc(&device_recv_vec[i * _SR_],                                   \
                 lat_3dim_Half_SC[i] * sizeof(LatticeComplex));                \
      cudaMalloc(&device_recv_vec[i * _SR_ + 1],                               \
                 lat_3dim_Half_SC[i] * sizeof(LatticeComplex));                \
      host_send_vec[i * _SR_] =                                                \
          (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));        \
      host_send_vec[i * _SR_ + 1] =                                            \
          (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));        \
      host_recv_vec[i * _SR_] =                                                \
          (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));        \
      host_recv_vec[i * _SR_ + 1] =                                            \
          (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));        \
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

#endif