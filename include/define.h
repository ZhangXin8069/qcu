#ifndef _DEFINE_H
#define _DEFINE_H
#include "./lattice_complex.h"
#define _QCU_BLOCK_SIZE_ 128
// #define _QCU_BLOCK_SIZE_ 16 // for small grid test
#define _qcu_a_ 0
#define _qcu_b_ 1
#define _qcu_c_ 2
#define _qcu_d_ 3
#define _qcu_tmp0_ 0
#define _qcu_tmp1_ 1
#define _qcu_rho_prev_ 2
#define _qcu_rho_ 3
#define _qcu_alpha_ 4
#define _qcu_beta_ 5
#define _qcu_omega_ 6
#define _qcu_send_tmp_ 7
#define _qcu_norm2_tmp_ 8
#define _qcu_diff_tmp_ 9
#define _qcu_lat_4dim_ 10
#define _qcu_vals_size_ 11
#define _QCU_NO_USE_ 0
#define _QCU_USE_ 1
#define _QCU_X_ 0
#define _QCU_Y_ 1
#define _QCU_Z_ 2
#define _QCU_T_ 3
#define _QCU_LAT_X_ 0
#define _QCU_LAT_Y_ 1
#define _QCU_LAT_Z_ 2
#define _QCU_LAT_T_ 3
#define _QCU_LAT_XYZT_ 4
#define _QCU_GRID_X_ 5
#define _QCU_GRID_Y_ 6
#define _QCU_GRID_Z_ 7
#define _QCU_GRID_T_ 8
#define _QCU_PARITY_ 9
#define _QCU_NODE_RANK_ 10
#define _QCU_NODE_SIZE_ 11
#define _QCU_DAGGER_ 12
#define _QCU_VALS_SIZE_ 13
#define _QCU_DIM_ 4
#define _QCU_1DIM_ 4
#define _QCU_2DIM_ 6
#define _QCU_3DIM_ 4
#define _QCU_B_X_ 0
#define _QCU_F_X_ 1
#define _QCU_B_Y_ 2
#define _QCU_F_Y_ 3
#define _QCU_B_Z_ 4
#define _QCU_F_Z_ 5
#define _QCU_B_T_ 6
#define _QCU_F_T_ 7
#define _QCU_BX_BY_ 8
#define _QCU_FX_BY_ 9
#define _QCU_BX_FY_ 10
#define _QCU_FX_FY_ 11
#define _QCU_BX_BZ_ 12
#define _QCU_FX_BZ_ 13
#define _QCU_BX_FZ_ 14
#define _QCU_FX_FZ_ 15
#define _QCU_BX_BT_ 16
#define _QCU_FX_BT_ 17
#define _QCU_BX_FT_ 18
#define _QCU_FX_FT_ 19
#define _QCU_BY_BZ_ 20
#define _QCU_FY_BZ_ 21
#define _QCU_BY_FZ_ 22
#define _QCU_FY_FZ_ 23
#define _QCU_BY_BT_ 24
#define _QCU_FY_BT_ 25
#define _QCU_BY_FT_ 26
#define _QCU_FY_FT_ 27
#define _QCU_BZ_BT_ 28
#define _QCU_FZ_BT_ 29
#define _QCU_BZ_FT_ 30
#define _QCU_FZ_FT_ 31
#define _QCU_B_X_B_Y_ 0
#define _QCU_F_X_B_Y_ 1
#define _QCU_B_X_F_Y_ 2
#define _QCU_F_X_F_Y_ 3
#define _QCU_B_X_B_Z_ 4
#define _QCU_F_X_B_Z_ 5
#define _QCU_B_X_F_Z_ 6
#define _QCU_F_X_F_Z_ 7
#define _QCU_B_X_B_T_ 8
#define _QCU_F_X_B_T_ 9
#define _QCU_B_X_F_T_ 10
#define _QCU_F_X_F_T_ 11
#define _QCU_B_Y_B_Z_ 12
#define _QCU_F_Y_B_Z_ 13
#define _QCU_B_Y_F_Z_ 14
#define _QCU_F_Y_F_Z_ 15
#define _QCU_B_Y_B_T_ 16
#define _QCU_F_Y_B_T_ 17
#define _QCU_B_Y_F_T_ 18
#define _QCU_F_Y_F_T_ 19
#define _QCU_B_Z_B_T_ 20
#define _QCU_F_Z_B_T_ 21
#define _QCU_B_Z_F_T_ 22
#define _QCU_F_Z_F_T_ 23
#define _QCU_WARDS_ 8
#define _QCU_WARDS_2DIM_ 24
#define _QCU_XY_ 0
#define _QCU_XZ_ 1
#define _QCU_XT_ 2
#define _QCU_YZ_ 3
#define _QCU_YT_ 4
#define _QCU_ZT_ 5
#define _QCU_YZT_ 0
#define _QCU_XZT_ 1
#define _QCU_XYT_ 2
#define _QCU_XYZ_ 3
#define _QCU_EVEN_ 0
#define _QCU_ODD_ 1
#define _QCU_EVEN_ODD_ 2
#define _QCU_LAT_C_ 3
#define _QCU_LAT_S_ 4
#define _QCU_LAT_CC_ 9
#define _QCU_LAT_1C_ 3
#define _QCU_LAT_2C_ 6
#define _QCU_LAT_3C_ 9
#define _QCU_LAT_HALF_SC_ 6
#define _QCU_LAT_SC_ 12
#define _QCU_LAT_SCSC_ 144
#define _QCU_LAT_D_ 4
#define _QCU_LAT_DCC_ 36
#define _QCU_LAT_PDCC_ 72
#define _QCU_B_ 0
#define _QCU_F_ 1
#define _QCU_BF_ 2
#define _QCU_REAL_IMAG_ 2
#define _QCU_OUTPUT_SIZE_ 10
#define _QCU_BACKWARD_ -1
#define _QCU_NOWARD_ 0
#define _QCU_FORWARD_ 1
#define _QCU_SR_ 2
#define _QCU_LAT_EXAMPLE_ 32
#define _QCU_GRID_EXAMPLE_ 1
#define _QCU_MAX_ITER_ 1e3
#define _QCU_TOL_ 1e-9
#define _QCU_MEM_POOL_ 0
#define _QCU_CHECK_ERROR_ 1
#define _QCU_DRAFT_
#define _QCU_LATTICE_SET_
#define _QCU_LATTICE_CUDA_
#define _QCU_LATTICE_CG_
#define _QCU_BISTABCG_
#define _QCU_MULTGRID_
#define _QCU_WILSON_DSLASH_
#define _QCU_CLOVER_DSLASH_
#define _QCU_NCCL_WILSON_DSLASH_
#define _QCU_NCCL_CLOVER_DSLASH_
#define _QCU_WILSON_CG_
#define _QCU_WILSON_BISTABCG_
#define _QCU_NCCL_WILSON_CG_
#define _QCU_NCCL_WILSON_BISTABCG_
// CUDA API error checking
#define CUDA_CHECK(err)                                                  \
  do                                                                     \
  {                                                                      \
    cudaError_t err_ = (err);                                            \
    if (err_ != cudaSuccess)                                             \
    {                                                                    \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("CUDA error");                            \
    }                                                                    \
  } while (0)
// cublas API error checking
#define CUBLAS_CHECK(err)                                                  \
  do                                                                       \
  {                                                                        \
    cublasStatus_t err_ = (err);                                           \
    if (err_ != CUBLAS_STATUS_SUCCESS)                                     \
    {                                                                      \
      std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cublas error");                            \
    }                                                                      \
  } while (0)
// curand API error checking
#define CURAND_CHECK(err)                                                  \
  do                                                                       \
  {                                                                        \
    curandStatus_t err_ = (err);                                           \
    if (err_ != CURAND_STATUS_SUCCESS)                                     \
    {                                                                      \
      std::printf("curand error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("curand error");                            \
    }                                                                      \
  } while (0)
#define give_ptr(U, origin_U, n) \
  {                              \
    for (int i = 0; i < n; i++)  \
    {                            \
      U[i] = origin_U[i];        \
    }                            \
  }
#define move_backward(move, y, lat_y) \
  {                                   \
    move = -1 + (y == 0) * lat_y;     \
  }
#define move_forward(move, y, lat_y)     \
  {                                      \
    move = 1 - (y == lat_y - 1) * lat_y; \
  }
#define move_backward_x(move, x, lat_x, eo, parity)  \
  {                                                  \
    move = (-1 + (x == 0) * lat_x) * (eo == parity); \
  }
#define move_forward_x(move, x, lat_x, eo, parity)          \
  {                                                         \
    move = (1 - (x == lat_x - 1) * lat_x) * (eo != parity); \
  }
#define checkCudaErrors(err)                                       \
  {                                                                \
    if (_QCU_CHECK_ERROR_)                                         \
    {                                                              \
      if (err != cudaSuccess)                                      \
      {                                                            \
        fprintf(stderr,                                            \
                "Failed: CUDA error %04d \"%s\" from file <%s>, "  \
                "line %i.\n",                                      \
                err, cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                        \
      }                                                            \
    }                                                              \
  }
#define checkMpiErrors(err)                               \
  {                                                       \
    if (_QCU_CHECK_ERROR_)                                \
    {                                                     \
      if (err != MPI_SUCCESS)                             \
      {                                                   \
        fprintf(stderr,                                   \
                "Failed: MPI error %04d from file <%s>, " \
                "line %i.\n",                             \
                err, __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                               \
      }                                                   \
    }                                                     \
  }
#define checkNcclErrors(err)                                       \
  {                                                                \
    if (_QCU_CHECK_ERROR_)                                         \
    {                                                              \
      if (err != ncclSuccess)                                      \
      {                                                            \
        fprintf(stderr,                                            \
                "Failed: NCCL error %04d \"%s\" from file <%s>, "  \
                "line %i.\n",                                      \
                err, ncclGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                        \
      }                                                            \
    }                                                              \
  }
// little strange, but don't want change
#define give_vals(U, zero, n)                                 \
  {                                                           \
    LatticeComplex *tmp_U = static_cast<LatticeComplex *>(U); \
    for (int i = 0; i < n; i++)                               \
    {                                                         \
      tmp_U[i] = zero;                                        \
    }                                                         \
  }
#define give_rand(input_matrix, size)                                \
  {                                                                  \
    for (int i = 0; i < size; i++)                                   \
    {                                                                \
      input_matrix[i].real = static_cast<double>(rand()) / RAND_MAX; \
      input_matrix[i].imag = static_cast<double>(rand()) / RAND_MAX; \
    }                                                                \
  }
#define give_u(U, tmp_U, lat_tzyx)                               \
  {                                                              \
    for (int i = 0; i < _QCU_LAT_2C_; i++)                       \
    {                                                            \
      U[i] = tmp_U[i * _QCU_LAT_D_ * _QCU_EVEN_ODD_ * lat_tzyx]; \
    }                                                            \
    U[6] = (U[1] * U[5] - U[2] * U[4]).conj();                   \
    U[7] = (U[2] * U[3] - U[0] * U[5]).conj();                   \
    U[8] = (U[0] * U[4] - U[1] * U[3]).conj();                   \
  }
#define _give_u_comm(parity, U, tmp_U, _lat_tzyx)                            \
  {                                                                          \
    for (int i = 0; i < _QCU_LAT_2C_; i++)                                   \
    {                                                                        \
      U[i] = tmp_U[(i * _QCU_LAT_D_ * _QCU_EVEN_ODD_ + parity) * _lat_tzyx]; \
    }                                                                        \
    U[6] = (U[1] * U[5] - U[2] * U[4]).conj();                               \
    U[7] = (U[2] * U[3] - U[0] * U[5]).conj();                               \
    U[8] = (U[0] * U[4] - U[1] * U[3]).conj();                               \
  }
#define give_src(src, origin_src, lat_tzyx) \
  {                                         \
    for (int i = 0; i < _QCU_LAT_SC_; i++)  \
    {                                       \
      src[i] = origin_src[i * lat_tzyx];    \
    }                                       \
  }
#define give_dest(origin_dest, dest, lat_tzyx) \
  {                                            \
    for (int i = 0; i < _QCU_LAT_SC_; i++)     \
    {                                          \
      origin_dest[i * lat_tzyx] = dest[i];     \
    }                                          \
  }
#define add_dest(origin_dest, dest, lat_tzyx) \
  {                                           \
    for (int i = 0; i < _QCU_LAT_SC_; i++)    \
    {                                         \
      origin_dest[i * lat_tzyx] += dest[i];   \
    }                                         \
  }
#define add_dest_x(origin_dest, dest, lat_tzyx, _) \
  {                                                \
    for (int i = 0; i < _QCU_LAT_SC_ * _; i++)     \
    {                                              \
      origin_dest[i * lat_tzyx] += dest[i];        \
    }                                              \
  }
#define give_recv(recv, origin_recv, lat_3dim)  \
  {                                             \
    for (int i = 0; i < _QCU_LAT_HALF_SC_; i++) \
    {                                           \
      recv[i] = origin_recv[i * lat_3dim];      \
    }                                           \
  }
#define give_send(origin_send, send, lat_3dim)  \
  {                                             \
    for (int i = 0; i < _QCU_LAT_HALF_SC_; i++) \
    {                                           \
      origin_send[i * lat_3dim] = send[i];      \
    }                                           \
  }
#define give_send_x(origin_send, send, lat_3dim, _) \
  {                                                 \
    for (int i = 0; i < _QCU_LAT_HALF_SC_ * _; i++) \
    {                                               \
      origin_send[i * lat_3dim] = send[i];          \
    }                                               \
  }
#define give_clr(origin_clr, clr, lat_tzyx)  \
  {                                          \
    for (int i = 0; i < _QCU_LAT_SCSC_; i++) \
    {                                        \
      origin_clr[i * lat_tzyx] = clr[i];     \
    }                                        \
  }
#define add_clr(origin_clr, clr, lat_tzyx)   \
  {                                          \
    for (int i = 0; i < _QCU_LAT_SCSC_; i++) \
    {                                        \
      origin_clr[i * lat_tzyx] += clr[i];    \
    }                                        \
  }
#define get_clr(clr, origin_clr, lat_tzyx)   \
  {                                          \
    for (int i = 0; i < _QCU_LAT_SCSC_; i++) \
    {                                        \
      clr[i] = origin_clr[i * lat_tzyx];     \
    }                                        \
  }
#define add_vals(U, tmp, n)     \
  {                             \
    for (int i = 0; i < n; i++) \
    {                           \
      U[i] += tmp[i];           \
    }                           \
  }
#define subt_vals(U, tmp, n)    \
  {                             \
    for (int i = 0; i < n; i++) \
    {                           \
      U[i] -= tmp[i];           \
    }                           \
  }
#define mult_vals(U, tmp, n)    \
  {                             \
    for (int i = 0; i < n; i++) \
    {                           \
      U[i] *= tmp[i];           \
    }                           \
  }
#define divi_vals(U, tmp, n)    \
  {                             \
    for (int i = 0; i < n; i++) \
    {                           \
      U[i] /= tmp[i];           \
    }                           \
  }
#define mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero)                       \
  {                                                                          \
    for (int c0 = 0; c0 < _QCU_LAT_C_; c0++)                                 \
    {                                                                        \
      for (int c1 = 0; c1 < _QCU_LAT_C_; c1++)                               \
      {                                                                      \
        tmp0 = zero;                                                         \
        for (int cc = 0; cc < _QCU_LAT_C_; cc++)                             \
        {                                                                    \
          tmp0 += tmp1[c0 * _QCU_LAT_C_ + cc] * tmp2[cc * _QCU_LAT_C_ + c1]; \
        }                                                                    \
        tmp3[c0 * _QCU_LAT_C_ + c1] = tmp0;                                  \
      }                                                                      \
    }                                                                        \
  }
#define mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero)                               \
  {                                                                                 \
    for (int c0 = 0; c0 < _QCU_LAT_C_; c0++)                                        \
    {                                                                               \
      for (int c1 = 0; c1 < _QCU_LAT_C_; c1++)                                      \
      {                                                                             \
        tmp0 = zero;                                                                \
        for (int cc = 0; cc < _QCU_LAT_C_; cc++)                                    \
        {                                                                           \
          tmp0 += tmp1[c0 * _QCU_LAT_C_ + cc] * tmp2[c1 * _QCU_LAT_C_ + cc].conj(); \
        }                                                                           \
        tmp3[c0 * _QCU_LAT_C_ + c1] = tmp0;                                         \
      }                                                                             \
    }                                                                               \
  }
#define mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero)                               \
  {                                                                                 \
    for (int c0 = 0; c0 < _QCU_LAT_C_; c0++)                                        \
    {                                                                               \
      for (int c1 = 0; c1 < _QCU_LAT_C_; c1++)                                      \
      {                                                                             \
        tmp0 = zero;                                                                \
        for (int cc = 0; cc < _QCU_LAT_C_; cc++)                                    \
        {                                                                           \
          tmp0 += tmp1[cc * _QCU_LAT_C_ + c0].conj() * tmp2[cc * _QCU_LAT_C_ + c1]; \
        }                                                                           \
        tmp3[c0 * _QCU_LAT_C_ + c1] = tmp0;                                         \
      }                                                                             \
    }                                                                               \
  }
#define mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero)                                   \
  {                                                                                    \
    for (int c0 = 0; c0 < _QCU_LAT_C_; c0++)                                           \
    {                                                                                  \
      for (int c1 = 0; c1 < _QCU_LAT_C_; c1++)                                         \
      {                                                                                \
        tmp0 = zero;                                                                   \
        for (int cc = 0; cc < _QCU_LAT_C_; cc++)                                       \
        {                                                                              \
          tmp0 +=                                                                      \
              tmp1[cc * _QCU_LAT_C_ + c0].conj() * tmp2[c1 * _QCU_LAT_C_ + cc].conj(); \
        }                                                                              \
        tmp3[c0 * _QCU_LAT_C_ + c1] = tmp0;                                            \
      }                                                                                \
    }                                                                                  \
  }
#define _inverse(input_matrix, inverse_matrix, augmented_matrix, pivot,    \
                 factor, size)                                             \
  {                                                                        \
    for (int i = 0; i < size; i++)                                         \
    {                                                                      \
      for (int j = 0; j < size; j++)                                       \
      {                                                                    \
        inverse_matrix[i * size + j] = input_matrix[i * size + j];         \
        augmented_matrix[i * 2 * size + j] = inverse_matrix[i * size + j]; \
      }                                                                    \
      augmented_matrix[i * 2 * size + size + i] = 1.0;                     \
    }                                                                      \
    for (int i = 0; i < size; i++)                                         \
    {                                                                      \
      pivot = augmented_matrix[i * 2 * size + i];                          \
      for (int j = 0; j < 2 * size; j++)                                   \
      {                                                                    \
        augmented_matrix[i * 2 * size + j] /= pivot;                       \
      }                                                                    \
      for (int j = 0; j < size; j++)                                       \
      {                                                                    \
        if (j != i)                                                        \
        {                                                                  \
          factor = augmented_matrix[j * 2 * size + i];                     \
          for (int k = 0; k < 2 * size; ++k)                               \
          {                                                                \
            augmented_matrix[j * 2 * size + k] -=                          \
                factor * augmented_matrix[i * 2 * size + k];               \
          }                                                                \
        }                                                                  \
      }                                                                    \
    }                                                                      \
    for (int i = 0; i < size; i++)                                         \
    {                                                                      \
      for (int j = 0; j < size; j++)                                       \
      {                                                                    \
        inverse_matrix[i * size + j] =                                     \
            augmented_matrix[i * 2 * size + size + j];                     \
      }                                                                    \
    }                                                                      \
  }
#define malloc_vec(lat_3dim_Half_SC, device_send_vec, device_recv_vec,  \
                   host_send_vec, host_recv_vec)                        \
  {                                                                     \
    for (int i = 0; i < _QCU_DIM_; i++)                                 \
    {                                                                   \
      cudaMalloc(&device_send_vec[i * _QCU_SR_],                        \
                 lat_3dim_Half_SC[i] * sizeof(LatticeComplex));         \
      cudaMalloc(&device_send_vec[i * _QCU_SR_ + 1],                    \
                 lat_3dim_Half_SC[i] * sizeof(LatticeComplex));         \
      cudaMalloc(&device_recv_vec[i * _QCU_SR_],                        \
                 lat_3dim_Half_SC[i] * sizeof(LatticeComplex));         \
      cudaMalloc(&device_recv_vec[i * _QCU_SR_ + 1],                    \
                 lat_3dim_Half_SC[i] * sizeof(LatticeComplex));         \
      host_send_vec[i * _QCU_SR_] =                                     \
          (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex)); \
      host_send_vec[i * _QCU_SR_ + 1] =                                 \
          (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex)); \
      host_recv_vec[i * _QCU_SR_] =                                     \
          (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex)); \
      host_recv_vec[i * _QCU_SR_ + 1] =                                 \
          (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex)); \
    }                                                                   \
  }
#define free_vec(device_send_vec, device_recv_vec, host_send_vec, \
                 host_recv_vec)                                   \
  {                                                               \
    for (int i = 0; i < _QCU_WARDS_; i++)                         \
    {                                                             \
      cudaFree(device_send_vec[i]);                               \
      cudaFree(device_recv_vec[i]);                               \
      free(host_send_vec[i]);                                     \
      free(host_recv_vec[i]);                                     \
    }                                                             \
  }
#endif
