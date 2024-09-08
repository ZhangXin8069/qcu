#ifndef _DEFINE_H
#define _DEFINE_H
#include "./lattice_complex.h"
// #define _BLOCK_SIZE_ 128
#define _BLOCK_SIZE_ 16 // for small grid test
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
#define _lat_xyzt_ 10
#define _vals_size_ 11
#define _X_ 0
#define _Y_ 1
#define _Z_ 2
#define _T_ 3
#define _XYZT_ 4
#define _VALS_SIZE_ 5
#define _DIM_ 4
#define _B_X_ 0
#define _F_X_ 1
#define _B_Y_ 2
#define _F_Y_ 3
#define _B_Z_ 4
#define _F_Z_ 5
#define _B_T_ 6
#define _F_T_ 7
#define _BX_BY_ 8
#define _FX_BY_ 9
#define _BX_FY_ 10
#define _FX_FY_ 11
#define _BX_BZ_ 12
#define _FX_BZ_ 13
#define _BX_FZ_ 14
#define _FX_FZ_ 15
#define _BX_BT_ 16
#define _FX_BT_ 17
#define _BX_FT_ 18
#define _FX_FT_ 19
#define _BY_BZ_ 20
#define _FY_BZ_ 21
#define _BY_FZ_ 22
#define _FY_FZ_ 23
#define _BY_BT_ 24
#define _FY_BT_ 25
#define _BY_FT_ 26
#define _FY_FT_ 27
#define _BZ_BT_ 28
#define _FZ_BT_ 29
#define _BZ_FT_ 30
#define _FZ_FT_ 31
#define _B_X_B_Y_ 0
#define _F_X_B_Y_ 1
#define _B_X_F_Y_ 2
#define _F_X_F_Y_ 3
#define _B_X_B_Z_ 4
#define _F_X_B_Z_ 5
#define _B_X_F_Z_ 6
#define _F_X_F_Z_ 7
#define _B_X_B_T_ 8
#define _F_X_B_T_ 9
#define _B_X_F_T_ 10
#define _F_X_F_T_ 11
#define _B_Y_B_Z_ 12
#define _F_Y_B_Z_ 13
#define _B_Y_F_Z_ 14
#define _F_Y_F_Z_ 15
#define _B_Y_B_T_ 16
#define _F_Y_B_T_ 17
#define _B_Y_F_T_ 18
#define _F_Y_F_T_ 19
#define _B_Z_B_T_ 20
#define _F_Z_B_T_ 21
#define _B_Z_F_T_ 22
#define _F_Z_F_T_ 23
#define _WARDS_ 8
#define _WARDS_2DIM_ 24
#define _XY_ 0
#define _XZ_ 1
#define _XT_ 2
#define _YZ_ 3
#define _YT_ 4
#define _ZT_ 5
#define _2DIM_ 6
#define _YZT_ 0
#define _XZT_ 1
#define _XYT_ 2
#define _XYZ_ 3
#define _EVEN_ 0
#define _ODD_ 1
#define _EVEN_ODD_ 2
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
#define _LAT_DCC_ 36
#define _LAT_PDCC_ 72
#define _B_ 0
#define _F_ 1
#define _BF_ 2
#define _REAL_IMAG_ 2
#define _OUTPUT_SIZE_ 10
#define _BACKWARD_ -1
#define _NOWARD_ 0
#define _FORWARD_ 1
#define _SR_ 2
#define _LAT_EXAMPLE_ 32
#define _GRID_EXAMPLE_ 1
#define _MAX_ITER_ 1e3
#define _TOL_ 1e-9
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
#define NCCL_CLOVER_DSLASH
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
// CUDA API error checking
#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);       \
      throw std::runtime_error("CUDA error");                                  \
    }                                                                          \
  } while (0)
// cublas API error checking
#define CUBLAS_CHECK(err)                                                      \
  do {                                                                         \
    cublasStatus_t err_ = (err);                                               \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("cublas error");                                \
    }                                                                          \
  } while (0)
// curand API error checking
#define CURAND_CHECK(err)                                                      \
  do {                                                                         \
    curandStatus_t err_ = (err);                                               \
    if (err_ != CURAND_STATUS_SUCCESS) {                                       \
      std::printf("curand error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("curand error");                                \
    }                                                                          \
  } while (0)
#define give_ptr(U, origin_U, n)                                               \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] = origin_U[i];                                                      \
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
#define give_vals(U, zero, n)                                                  \
  {                                                                            \
    LatticeComplex *tmp_U = static_cast<LatticeComplex *>(U);                  \
    for (int i = 0; i < n; i++) {                                              \
      tmp_U[i] = zero;                                                         \
    }                                                                          \
  }
#define give_rand(input_matrix, size)                                          \
  {                                                                            \
    for (int i = 0; i < size; i++) {                                           \
      input_matrix[i].real = static_cast<double>(rand()) / RAND_MAX;           \
      input_matrix[i].imag = static_cast<double>(rand()) / RAND_MAX;           \
    }                                                                          \
  }
#define give_u(U, tmp_U, lat_tzyx)                                             \
  {                                                                            \
    for (int i = 0; i < _LAT_2C_; i++) {                                       \
      U[i] = tmp_U[i * _LAT_D_ * _EVEN_ODD_ * lat_tzyx];                       \
    }                                                                          \
    U[6] = (U[1] * U[5] - U[2] * U[4]).conj();                                 \
    U[7] = (U[2] * U[3] - U[0] * U[5]).conj();                                 \
    U[8] = (U[0] * U[4] - U[1] * U[3]).conj();                                 \
  }
#define _give_u_comm(parity, U, tmp_U, _lat_tzyx)                              \
  {                                                                            \
    for (int i = 0; i < _LAT_2C_; i++) {                                       \
      U[i] = tmp_U[(i * _LAT_D_ * _EVEN_ODD_ + parity) * _lat_tzyx];           \
    }                                                                          \
    U[6] = (U[1] * U[5] - U[2] * U[4]).conj();                                 \
    U[7] = (U[2] * U[3] - U[0] * U[5]).conj();                                 \
    U[8] = (U[0] * U[4] - U[1] * U[3]).conj();                                 \
  }
#define give_src(src, origin_src, lat_tzyx)                                    \
  {                                                                            \
    for (int i = 0; i < _LAT_SC_; i++) {                                       \
      src[i] = origin_src[i * lat_tzyx];                                       \
    }                                                                          \
  }
#define give_dest(origin_dest, dest, lat_tzyx)                                 \
  {                                                                            \
    for (int i = 0; i < _LAT_SC_; i++) {                                       \
      origin_dest[i * lat_tzyx] = dest[i];                                     \
    }                                                                          \
  }
#define add_dest(origin_dest, dest, lat_tzyx)                                  \
  {                                                                            \
    for (int i = 0; i < _LAT_SC_; i++) {                                       \
      origin_dest[i * lat_tzyx] += dest[i];                                    \
    }                                                                          \
  }
#define add_dest_x(origin_dest, dest, lat_tzyx, _)                             \
  {                                                                            \
    for (int i = 0; i < _LAT_SC_ * _; i++) {                                   \
      origin_dest[i * lat_tzyx] += dest[i];                                    \
    }                                                                          \
  }
#define give_recv(recv, origin_recv, lat_3dim)                                 \
  {                                                                            \
    for (int i = 0; i < _LAT_HALF_SC_; i++) {                                  \
      recv[i] = origin_recv[i * lat_3dim];                                     \
    }                                                                          \
  }
#define give_send(origin_send, send, lat_3dim)                                 \
  {                                                                            \
    for (int i = 0; i < _LAT_HALF_SC_; i++) {                                  \
      origin_send[i * lat_3dim] = send[i];                                     \
    }                                                                          \
  }
#define give_send_x(origin_send, send, lat_3dim, _)                            \
  {                                                                            \
    for (int i = 0; i < _LAT_HALF_SC_ * _; i++) {                              \
      origin_send[i * lat_3dim] = send[i];                                     \
    }                                                                          \
  }
#define give_clr(origin_clr, clr, lat_tzyx)                                    \
  {                                                                            \
    for (int i = 0; i < _LAT_SCSC_; i++) {                                     \
      origin_clr[i * lat_tzyx] = clr[i];                                       \
    }                                                                          \
  }
#define add_clr(origin_clr, clr, lat_tzyx)                                     \
  {                                                                            \
    for (int i = 0; i < _LAT_SCSC_; i++) {                                     \
      origin_clr[i * lat_tzyx] += clr[i];                                      \
    }                                                                          \
  }
#define get_clr(clr, origin_clr, lat_tzyx)                                     \
  {                                                                            \
    for (int i = 0; i < _LAT_SCSC_; i++) {                                     \
      clr[i] = origin_clr[i * lat_tzyx];                                       \
    }                                                                          \
  }
#define add_vals(U, tmp, n)                                                    \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] += tmp[i];                                                          \
    }                                                                          \
  }
#define subt_vals(U, tmp, n)                                                   \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] -= tmp[i];                                                          \
    }                                                                          \
  }
#define mult_vals(U, tmp, n)                                                   \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] *= tmp[i];                                                          \
    }                                                                          \
  }
#define divi_vals(U, tmp, n)                                                   \
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
#define _inverse(input_matrix, inverse_matrix, augmented_matrix, pivot,        \
                 factor, size)                                                 \
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