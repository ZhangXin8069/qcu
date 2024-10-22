#ifndef _LATTICE_SET_H
#define _LATTICE_SET_H
#include "./lattice_cuda.h"
#pragma once
// clang-format off
#include "./define.h"
// clang-format on
__global__ void give_param(void *device_param, int vals_index, int val);
struct LatticeSet {
  int lat_2dim[_QCU_2DIM_];
  int lat_3dim[_QCU_3DIM_];
  int lat_4dim;
  int lat_3dim_Half_SC[_QCU_3DIM_];
  int lat_3dim_SC[_QCU_3DIM_];
  int lat_4dim_SC;
  int lat_4dim_DCC;
  dim3 gridDim_3dim[_QCU_3DIM_];
  dim3 gridDim_2dim[_QCU_2DIM_];
  dim3 gridDim;
  dim3 blockDim;
  ncclUniqueId nccl_id;
  ncclComm_t nccl_comm;
  cublasHandle_t cublasH;
  cudaStream_t stream;
  cublasHandle_t cublasHs[_QCU_DIM_];
  cudaStream_t streams[_QCU_DIM_];
  cudaStream_t stream_dims[_QCU_DIM_];
  float time;
  double mass;
  cudaEvent_t start, stop;
  cudaError_t err;
  int move[_QCU_BF_];
  int move_wards[_QCU_WARDS_ + _QCU_WARDS_2DIM_];
  int grid_2dim[_QCU_2DIM_];
  int grid_3dim[_QCU_3DIM_];
  int grid_index_1dim[_QCU_1DIM_];
  MPI_Request send_request[_QCU_WARDS_];
  MPI_Request recv_request[_QCU_WARDS_];
  void *host_send_vec[_QCU_WARDS_];
  void *host_recv_vec[_QCU_WARDS_];
  int host_params[_QCU_VALS_SIZE_];
  void *device_send_vec[_QCU_WARDS_];
  void *device_recv_vec[_QCU_WARDS_];
  void *device_u_1dim_send_vec[_QCU_WARDS_];
  void *device_u_1dim_recv_vec[_QCU_WARDS_];
  void *device_u_2dim_send_vec[_QCU_2DIM_ * _QCU_BF_ * _QCU_BF_];
  void *device_u_2dim_recv_vec[_QCU_2DIM_ * _QCU_BF_ * _QCU_BF_];
  void *device_params;
  void *device_params_even_no_dag;
  void *device_params_odd_no_dag;
  void *device_params_even_dag;
  void *device_params_odd_dag;
  void give(int *_param_lat_size, int *_grid_lat_size) {
    host_params[_QCU_LAT_X_] = _param_lat_size[_QCU_X_] / _QCU_EVEN_ODD_; // even-odd
    host_params[_QCU_LAT_Y_] = _param_lat_size[_QCU_Y_];
    host_params[_QCU_LAT_Z_] = _param_lat_size[_QCU_Z_];
    host_params[_QCU_LAT_T_] = _param_lat_size[_QCU_T_];
    host_params[_QCU_GRID_X_] = _grid_lat_size[_QCU_X_];
    host_params[_QCU_GRID_Y_] = _grid_lat_size[_QCU_Y_];
    host_params[_QCU_GRID_Z_] = _grid_lat_size[_QCU_Z_];
    host_params[_QCU_GRID_T_] = _grid_lat_size[_QCU_T_];
  }
  void give(int *_param_lat_size, int *_grid_lat_size, int parity) {
    host_params[_QCU_LAT_X_] = _param_lat_size[_QCU_X_] / _QCU_EVEN_ODD_; // even-odd
    host_params[_QCU_LAT_Y_] = _param_lat_size[_QCU_Y_];
    host_params[_QCU_LAT_Z_] = _param_lat_size[_QCU_Z_];
    host_params[_QCU_LAT_T_] = _param_lat_size[_QCU_T_];
    host_params[_QCU_GRID_X_] = _grid_lat_size[_QCU_X_];
    host_params[_QCU_GRID_Y_] = _grid_lat_size[_QCU_Y_];
    host_params[_QCU_GRID_Z_] = _grid_lat_size[_QCU_Z_];
    host_params[_QCU_GRID_T_] = _grid_lat_size[_QCU_T_];
    host_params[_QCU_PARITY_] = parity;
  }
  void give(int *_param_lat_size, int parity) {
    host_params[_QCU_LAT_X_] = _param_lat_size[_QCU_X_] / _QCU_EVEN_ODD_; // even-odd
    host_params[_QCU_LAT_Y_] = _param_lat_size[_QCU_Y_];
    host_params[_QCU_LAT_Z_] = _param_lat_size[_QCU_Z_];
    host_params[_QCU_LAT_T_] = _param_lat_size[_QCU_T_];
    host_params[_QCU_GRID_X_] = _QCU_GRID_EXAMPLE_;
    host_params[_QCU_GRID_Y_] = _QCU_GRID_EXAMPLE_;
    host_params[_QCU_GRID_Z_] = _QCU_GRID_EXAMPLE_;
    host_params[_QCU_GRID_T_] = _QCU_GRID_EXAMPLE_;
    host_params[_QCU_PARITY_] = parity;
  }
  void give(int parity) {
    host_params[_QCU_LAT_X_] = _QCU_LAT_EXAMPLE_;
    host_params[_QCU_LAT_Y_] = _QCU_LAT_EXAMPLE_;
    host_params[_QCU_LAT_Z_] = _QCU_LAT_EXAMPLE_;
    host_params[_QCU_LAT_T_] = _QCU_LAT_EXAMPLE_;
    host_params[_QCU_GRID_X_] = _QCU_GRID_EXAMPLE_;
    host_params[_QCU_GRID_Y_] = _QCU_GRID_EXAMPLE_;
    host_params[_QCU_GRID_Z_] = _QCU_GRID_EXAMPLE_;
    host_params[_QCU_GRID_T_] = _QCU_GRID_EXAMPLE_;
    host_params[_QCU_PARITY_] = parity;
  }
  void init() {
    {
      blockDim = _BLOCK_SIZE_;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);
      cudaEventSynchronize(start);
      checkMpiErrors(MPI_Comm_rank(MPI_COMM_WORLD, host_params +_QCU_NODE_RANK_));
      checkMpiErrors(MPI_Comm_size(MPI_COMM_WORLD, host_params + _QCU_NODE_SIZE_));
      grid_index_1dim[_QCU_X_] = host_params[_QCU_NODE_RANK_] / host_params[_QCU_GRID_T_] /
                             host_params[_QCU_GRID_Z_] / host_params[_QCU_GRID_Y_];
      grid_index_1dim[_QCU_Y_] = host_params[_QCU_NODE_RANK_] / host_params[_QCU_GRID_T_] /
                             host_params[_QCU_GRID_Z_] % host_params[_QCU_GRID_Y_];
      grid_index_1dim[_QCU_Z_] = host_params[_QCU_NODE_RANK_] / host_params[_QCU_GRID_T_] %
                             host_params[_QCU_GRID_Z_];
      grid_index_1dim[_QCU_T_] = host_params[_QCU_NODE_RANK_] % host_params[_QCU_GRID_T_];
      grid_2dim[_QCU_XY_] = host_params[_QCU_GRID_X_] * host_params[_QCU_GRID_Y_];
      grid_2dim[_QCU_XZ_] = host_params[_QCU_GRID_X_] * host_params[_QCU_GRID_Z_];
      grid_2dim[_QCU_XT_] = host_params[_QCU_GRID_X_] * host_params[_QCU_GRID_T_];
      grid_2dim[_QCU_YZ_] = host_params[_QCU_GRID_Y_] * host_params[_QCU_GRID_Z_];
      grid_2dim[_QCU_YT_] = host_params[_QCU_GRID_Y_] * host_params[_QCU_GRID_T_];
      grid_2dim[_QCU_ZT_] = host_params[_QCU_GRID_Z_] * host_params[_QCU_GRID_T_];
      grid_3dim[_QCU_YZT_] =
          host_params[_QCU_GRID_Y_] * host_params[_QCU_GRID_Z_] * host_params[_QCU_GRID_T_];
      grid_3dim[_QCU_XZT_] =
          host_params[_QCU_GRID_X_] * host_params[_QCU_GRID_Z_] * host_params[_QCU_GRID_T_];
      grid_3dim[_QCU_XYT_] =
          host_params[_QCU_GRID_X_] * host_params[_QCU_GRID_Y_] * host_params[_QCU_GRID_T_];
      grid_3dim[_QCU_XYZ_] =
          host_params[_QCU_GRID_X_] * host_params[_QCU_GRID_Y_] * host_params[_QCU_GRID_Z_];
      lat_2dim[_QCU_XY_] = host_params[_QCU_LAT_X_] * host_params[_QCU_LAT_Y_];
      lat_2dim[_QCU_XZ_] = host_params[_QCU_LAT_X_] * host_params[_QCU_LAT_Z_];
      lat_2dim[_QCU_XT_] = host_params[_QCU_LAT_X_] * host_params[_QCU_LAT_T_];
      lat_2dim[_QCU_YZ_] = host_params[_QCU_LAT_Y_] * host_params[_QCU_LAT_Z_];
      lat_2dim[_QCU_YT_] = host_params[_QCU_LAT_Y_] * host_params[_QCU_LAT_T_];
      lat_2dim[_QCU_ZT_] = host_params[_QCU_LAT_Z_] * host_params[_QCU_LAT_T_];
      gridDim_2dim[_QCU_XY_] = lat_2dim[_QCU_XY_] / _BLOCK_SIZE_;
      gridDim_2dim[_QCU_XZ_] = lat_2dim[_QCU_XZ_] / _BLOCK_SIZE_;
      gridDim_2dim[_QCU_XT_] = lat_2dim[_QCU_XT_] / _BLOCK_SIZE_;
      gridDim_2dim[_QCU_YZ_] = lat_2dim[_QCU_YZ_] / _BLOCK_SIZE_;
      gridDim_2dim[_QCU_YT_] = lat_2dim[_QCU_YT_] / _BLOCK_SIZE_;
      gridDim_2dim[_QCU_ZT_] = lat_2dim[_QCU_ZT_] / _BLOCK_SIZE_;
      lat_3dim[_QCU_YZT_] =
          host_params[_QCU_LAT_Y_] * host_params[_QCU_LAT_Z_] * host_params[_QCU_LAT_T_];
      lat_3dim[_QCU_XZT_] =
          host_params[_QCU_LAT_X_] * host_params[_QCU_LAT_Z_] * host_params[_QCU_LAT_T_];
      lat_3dim[_QCU_XYT_] =
          host_params[_QCU_LAT_X_] * host_params[_QCU_LAT_Y_] * host_params[_QCU_LAT_T_];
      lat_3dim[_QCU_XYZ_] =
          host_params[_QCU_LAT_X_] * host_params[_QCU_LAT_Y_] * host_params[_QCU_LAT_Z_];
      gridDim_3dim[_QCU_YZT_] = lat_3dim[_QCU_YZT_] / _BLOCK_SIZE_;
      gridDim_3dim[_QCU_XZT_] = lat_3dim[_QCU_XZT_] / _BLOCK_SIZE_;
      gridDim_3dim[_QCU_XYT_] = lat_3dim[_QCU_XYT_] / _BLOCK_SIZE_;
      gridDim_3dim[_QCU_XYZ_] = lat_3dim[_QCU_XYZ_] / _BLOCK_SIZE_;
      lat_4dim = lat_3dim[_QCU_XYZ_] * host_params[_QCU_LAT_T_];
      host_params[_QCU_LAT_XYZT_] = lat_4dim;
      lat_4dim_SC = lat_4dim * _QCU_LAT_SC_;
      lat_4dim_DCC = lat_4dim * _QCU_LAT_DCC_;
      gridDim = lat_4dim / _BLOCK_SIZE_;
    }
    {
      move_backward(move_wards[_QCU_B_X_], grid_index_1dim[_QCU_X_],
                    host_params[_QCU_GRID_X_]);
      move_backward(move_wards[_QCU_B_Y_], grid_index_1dim[_QCU_Y_],
                    host_params[_QCU_GRID_Y_]);
      move_backward(move_wards[_QCU_B_Z_], grid_index_1dim[_QCU_Z_],
                    host_params[_QCU_GRID_Z_]);
      move_backward(move_wards[_QCU_B_T_], grid_index_1dim[_QCU_T_],
                    host_params[_QCU_GRID_T_]);
      move_forward(move_wards[_QCU_F_X_], grid_index_1dim[_QCU_X_],
                   host_params[_QCU_GRID_X_]);
      move_forward(move_wards[_QCU_F_Y_], grid_index_1dim[_QCU_Y_],
                   host_params[_QCU_GRID_Y_]);
      move_forward(move_wards[_QCU_F_Z_], grid_index_1dim[_QCU_Z_],
                   host_params[_QCU_GRID_Z_]);
      move_forward(move_wards[_QCU_F_T_], grid_index_1dim[_QCU_T_],
                   host_params[_QCU_GRID_T_]);
      move_wards[_QCU_B_X_] = host_params[_QCU_NODE_RANK_] + move_wards[_QCU_B_X_];
      move_wards[_QCU_B_Y_] =
          host_params[_QCU_NODE_RANK_] + move_wards[_QCU_B_Y_] * host_params[_QCU_GRID_X_];
      move_wards[_QCU_B_Z_] =
          host_params[_QCU_NODE_RANK_] + move_wards[_QCU_B_Z_] * grid_2dim[_QCU_XY_];
      move_wards[_QCU_B_T_] =
          host_params[_QCU_NODE_RANK_] + move_wards[_QCU_B_T_] * grid_3dim[_QCU_XYZ_];
      move_wards[_QCU_F_X_] = host_params[_QCU_NODE_RANK_] + move_wards[_QCU_F_X_];
      move_wards[_QCU_F_Y_] =
          host_params[_QCU_NODE_RANK_] + move_wards[_QCU_F_Y_] * host_params[_QCU_GRID_X_];
      move_wards[_QCU_F_Z_] =
          host_params[_QCU_NODE_RANK_] + move_wards[_QCU_F_Z_] * grid_2dim[_QCU_XY_];
      move_wards[_QCU_F_T_] =
          host_params[_QCU_NODE_RANK_] + move_wards[_QCU_F_T_] * grid_3dim[_QCU_XYZ_];
      int tmp;
      { // BB
        move_backward(tmp, grid_index_1dim[_QCU_Y_], host_params[_QCU_GRID_Y_]);
        move_wards[_QCU_BX_BY_] = move_wards[_QCU_B_X_] + tmp * host_params[_QCU_GRID_X_];
        move_backward(tmp, grid_index_1dim[_QCU_Z_], host_params[_QCU_GRID_Z_]);
        move_wards[_QCU_BX_BZ_] = move_wards[_QCU_B_X_] + tmp * grid_2dim[_QCU_XY_];
        move_backward(tmp, grid_index_1dim[_QCU_T_], host_params[_QCU_GRID_T_]);
        move_wards[_QCU_BX_BT_] = move_wards[_QCU_B_X_] + tmp * grid_3dim[_QCU_XYZ_];
        move_backward(tmp, grid_index_1dim[_QCU_Z_], host_params[_QCU_GRID_Z_]);
        move_wards[_QCU_BY_BZ_] = move_wards[_QCU_B_Y_] + tmp * grid_2dim[_QCU_XY_];
        move_backward(tmp, grid_index_1dim[_QCU_T_], host_params[_QCU_GRID_T_]);
        move_wards[_QCU_BY_BT_] = move_wards[_QCU_B_Y_] + tmp * grid_3dim[_QCU_XYZ_];
        move_backward(tmp, grid_index_1dim[_QCU_T_], host_params[_QCU_GRID_T_]);
        move_wards[_QCU_BZ_BT_] = move_wards[_QCU_B_Z_] + tmp * grid_3dim[_QCU_XYZ_];
      }
      { // FB
        move_backward(tmp, grid_index_1dim[_QCU_Y_], host_params[_QCU_GRID_Y_]);
        move_wards[_QCU_FX_BY_] = move_wards[_QCU_F_X_] + tmp * host_params[_QCU_GRID_X_];
        move_backward(tmp, grid_index_1dim[_QCU_Z_], host_params[_QCU_GRID_Z_]);
        move_wards[_QCU_FX_BZ_] = move_wards[_QCU_F_X_] + tmp * grid_2dim[_QCU_XY_];
        move_backward(tmp, grid_index_1dim[_QCU_T_], host_params[_QCU_GRID_T_]);
        move_wards[_QCU_FX_BT_] = move_wards[_QCU_F_X_] + tmp * grid_3dim[_QCU_XYZ_];
        move_backward(tmp, grid_index_1dim[_QCU_Z_], host_params[_QCU_GRID_Z_]);
        move_wards[_QCU_FY_BZ_] = move_wards[_QCU_F_Y_] + tmp * grid_2dim[_QCU_XY_];
        move_backward(tmp, grid_index_1dim[_QCU_T_], host_params[_QCU_GRID_T_]);
        move_wards[_QCU_FY_BT_] = move_wards[_QCU_F_Y_] + tmp * grid_3dim[_QCU_XYZ_];
        move_backward(tmp, grid_index_1dim[_QCU_T_], host_params[_QCU_GRID_T_]);
        move_wards[_QCU_FZ_BT_] = move_wards[_QCU_F_Z_] + tmp * grid_3dim[_QCU_XYZ_];
      }
      { // BF
        move_forward(tmp, grid_index_1dim[_QCU_Y_], host_params[_QCU_GRID_Y_]);
        move_wards[_QCU_BX_FY_] = move_wards[_QCU_B_X_] + tmp * host_params[_QCU_GRID_X_];
        move_forward(tmp, grid_index_1dim[_QCU_Z_], host_params[_QCU_GRID_Z_]);
        move_wards[_QCU_BX_FZ_] = move_wards[_QCU_B_X_] + tmp * grid_2dim[_QCU_XY_];
        move_forward(tmp, grid_index_1dim[_QCU_T_], host_params[_QCU_GRID_T_]);
        move_wards[_QCU_BX_FT_] = move_wards[_QCU_B_X_] + tmp * grid_3dim[_QCU_XYZ_];
        move_forward(tmp, grid_index_1dim[_QCU_Z_], host_params[_QCU_GRID_Z_]);
        move_wards[_QCU_BY_FZ_] = move_wards[_QCU_B_Y_] + tmp * grid_2dim[_QCU_XY_];
        move_forward(tmp, grid_index_1dim[_QCU_T_], host_params[_QCU_GRID_T_]);
        move_wards[_QCU_BY_FT_] = move_wards[_QCU_B_Y_] + tmp * grid_3dim[_QCU_XYZ_];
        move_forward(tmp, grid_index_1dim[_QCU_T_], host_params[_QCU_GRID_T_]);
        move_wards[_QCU_BZ_FT_] = move_wards[_QCU_B_Z_] + tmp * grid_3dim[_QCU_XYZ_];
      }
      { // FF
        move_forward(tmp, grid_index_1dim[_QCU_Y_], host_params[_QCU_GRID_Y_]);
        move_wards[_QCU_FX_FY_] = move_wards[_QCU_F_X_] + tmp * host_params[_QCU_GRID_X_];
        move_forward(tmp, grid_index_1dim[_QCU_Z_], host_params[_QCU_GRID_Z_]);
        move_wards[_QCU_FX_FZ_] = move_wards[_QCU_F_X_] + tmp * grid_2dim[_QCU_XY_];
        move_forward(tmp, grid_index_1dim[_QCU_T_], host_params[_QCU_GRID_T_]);
        move_wards[_QCU_FX_FT_] = move_wards[_QCU_F_X_] + tmp * grid_3dim[_QCU_XYZ_];
        move_forward(tmp, grid_index_1dim[_QCU_Z_], host_params[_QCU_GRID_Z_]);
        move_wards[_QCU_FY_FZ_] = move_wards[_QCU_F_Y_] + tmp * grid_2dim[_QCU_XY_];
        move_forward(tmp, grid_index_1dim[_QCU_T_], host_params[_QCU_GRID_T_]);
        move_wards[_QCU_FY_FT_] = move_wards[_QCU_F_Y_] + tmp * grid_3dim[_QCU_XYZ_];
        move_forward(tmp, grid_index_1dim[_QCU_T_], host_params[_QCU_GRID_T_]);
        move_wards[_QCU_FZ_FT_] = move_wards[_QCU_F_Z_] + tmp * grid_3dim[_QCU_XYZ_];
      }
    }
    {
      // nccl set
      if (host_params[_QCU_NODE_RANK_] == 0) {
        checkNcclErrors(ncclGetUniqueId(&nccl_id));
      }
      checkMpiErrors(MPI_Bcast((void *)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0,
                               MPI_COMM_WORLD));
      checkNcclErrors(ncclCommInitRank(&nccl_comm, host_params[_QCU_NODE_SIZE_],
                                       nccl_id, host_params[_QCU_NODE_RANK_]));
    }
    { // set stream and malloc vec
      CUBLAS_CHECK(cublasCreate(&cublasH));
      checkCudaErrors(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUBLAS_CHECK(cublasSetStream(cublasH, stream));
      for (int i = 0; i < _QCU_DIM_; i++) {
        CUBLAS_CHECK(cublasCreate(&cublasHs[i]));
        checkCudaErrors(
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        // checkCudaErrors(cudaStreamCreate(&streams[i]));
        checkCudaErrors(
            cudaStreamCreateWithFlags(&stream_dims[i], cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(cublasHs[i], streams[i]));
        lat_3dim_Half_SC[i] = lat_3dim[i] * _QCU_LAT_HALF_SC_;
        lat_3dim_SC[i] = lat_3dim_Half_SC[i] * 2;
        checkCudaErrors(cudaMallocAsync(
            &device_send_vec[i * _QCU_BF_],
            lat_3dim_Half_SC[i] * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_send_vec[i * _QCU_BF_ + 1],
            lat_3dim_Half_SC[i] * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_recv_vec[i * _QCU_BF_],
            lat_3dim_Half_SC[i] * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_recv_vec[i * _QCU_BF_ + 1],
            lat_3dim_Half_SC[i] * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_1dim_send_vec[i * _QCU_BF_],
            lat_3dim[i] * _QCU_LAT_PDCC_ * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_1dim_send_vec[i * _QCU_BF_ + 1],
            lat_3dim[i] * _QCU_LAT_PDCC_ * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_1dim_recv_vec[i * _QCU_BF_],
            lat_3dim[i] * _QCU_LAT_PDCC_ * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_1dim_recv_vec[i * _QCU_BF_ + 1],
            lat_3dim[i] * _QCU_LAT_PDCC_ * sizeof(LatticeComplex), stream));
        host_send_vec[i * _QCU_BF_] =
            (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));
        host_send_vec[i * _QCU_BF_ + 1] =
            (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));
        host_recv_vec[i * _QCU_BF_] =
            (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));
        host_recv_vec[i * _QCU_BF_ + 1] =
            (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));
      }
      for (int i = 0; i < _QCU_2DIM_; i++) {
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_send_vec[i * _QCU_BF_ * _QCU_BF_ + 0],
            lat_2dim[_QCU_2DIM_ - 1 - i] * _QCU_LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_recv_vec[i * _QCU_BF_ * _QCU_BF_ + 0],
            lat_2dim[_QCU_2DIM_ - 1 - i] * _QCU_LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_send_vec[i * _QCU_BF_ * _QCU_BF_ + 1],
            lat_2dim[_QCU_2DIM_ - 1 - i] * _QCU_LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_recv_vec[i * _QCU_BF_ * _QCU_BF_ + 1],
            lat_2dim[_QCU_2DIM_ - 1 - i] * _QCU_LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_send_vec[i * _QCU_BF_ * _QCU_BF_ + 2],
            lat_2dim[_QCU_2DIM_ - 1 - i] * _QCU_LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_recv_vec[i * _QCU_BF_ * _QCU_BF_ + 2],
            lat_2dim[_QCU_2DIM_ - 1 - i] * _QCU_LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_send_vec[i * _QCU_BF_ * _QCU_BF_ + 3],
            lat_2dim[_QCU_2DIM_ - 1 - i] * _QCU_LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_recv_vec[i * _QCU_BF_ * _QCU_BF_ + 3],
            lat_2dim[_QCU_2DIM_ - 1 - i] * _QCU_LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
      }
    }
    {
      checkCudaErrors(
          cudaMallocAsync(&device_params, _QCU_VALS_SIZE_ * sizeof(int), stream));
      checkCudaErrors(cudaMallocAsync(&device_params_even_no_dag,
                                      _QCU_VALS_SIZE_ * sizeof(int), stream));
      checkCudaErrors(cudaMallocAsync(&device_params_odd_no_dag,
                                      _QCU_VALS_SIZE_ * sizeof(int), stream));
      checkCudaErrors(cudaMallocAsync(&device_params_even_dag,
                                      _QCU_VALS_SIZE_ * sizeof(int), stream));
      checkCudaErrors(cudaMallocAsync(&device_params_odd_dag,
                                      _QCU_VALS_SIZE_ * sizeof(int), stream));
      checkCudaErrors(cudaMemcpyAsync(device_params, host_params,
                                      _QCU_VALS_SIZE_ * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));
      checkCudaErrors(cudaMemcpyAsync(device_params_even_no_dag, host_params,
                                      _QCU_VALS_SIZE_ * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));
      give_param<<<1, 1, 0, stream>>>(device_params_even_no_dag, _QCU_PARITY_,
                                      _QCU_EVEN_);
      give_param<<<1, 1, 0, stream>>>(device_params_even_no_dag, _QCU_DAGGER_,
                                      _QCU_NO_USE_);
      checkCudaErrors(cudaMemcpyAsync(device_params_odd_no_dag, host_params,
                                      _QCU_VALS_SIZE_ * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));
      give_param<<<1, 1, 0, stream>>>(device_params_odd_no_dag, _QCU_PARITY_,
                                      _QCU_ODD_);
      give_param<<<1, 1, 0, stream>>>(device_params_odd_no_dag, _QCU_DAGGER_,
                                      _QCU_NO_USE_);
      checkCudaErrors(cudaMemcpyAsync(device_params_even_dag, host_params,
                                      _QCU_VALS_SIZE_ * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));
      give_param<<<1, 1, 0, stream>>>(device_params_even_dag, _QCU_PARITY_, _QCU_EVEN_);
      give_param<<<1, 1, 0, stream>>>(device_params_even_dag, _QCU_DAGGER_, _QCU_USE_);
      checkCudaErrors(cudaMemcpyAsync(device_params_odd_dag, host_params,
                                      _QCU_VALS_SIZE_ * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));
      give_param<<<1, 1, 0, stream>>>(device_params_odd_dag, _QCU_PARITY_, _QCU_ODD_);
      give_param<<<1, 1, 0, stream>>>(device_params_odd_dag, _QCU_DAGGER_, _QCU_USE_);
    }
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  double kappa() {
    /*
    a=1(always\ ignore)
    r=1(in\ code\ written\ as\ coeff\_r)
    C_{SW}=1(in\ code\ written\ as\ coeff\_t)
    \kappa=\frac{1}{2m_q a+8r}
    or\ just\ define(m=-3.5):\\ \kappa=1(in\ code\ written\ as\ kappa)
    */
    // mass = -2.5;
    mass = 0.0;
    return 1 / (2 * mass + 8);
  }
  float get_time() {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    return time; // ms
  }
  void end() {
    checkCudaErrors(cudaStreamSynchronize(stream));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    checkCudaErrors(cudaFreeAsync(device_params, stream));
    checkCudaErrors(cudaFreeAsync(device_params_even_no_dag, stream));
    checkCudaErrors(cudaFreeAsync(device_params_odd_no_dag, stream));
    checkCudaErrors(cudaFreeAsync(device_params_even_dag, stream));
    checkCudaErrors(cudaFreeAsync(device_params_odd_dag, stream));
    for (int i = 0; i < _QCU_DIM_; i++) {
      checkCudaErrors(cudaStreamSynchronize(streams[i]));
      checkCudaErrors(cudaStreamSynchronize(stream_dims[i]));
      CUBLAS_CHECK(cublasDestroy(cublasHs[i]));
      checkCudaErrors(cudaStreamDestroy(streams[i]));
      checkCudaErrors(cudaStreamDestroy(stream_dims[i]));
      checkCudaErrors(cudaFreeAsync(device_send_vec[i * _QCU_BF_], stream));
      checkCudaErrors(cudaFreeAsync(device_send_vec[i * _QCU_BF_ + 1], stream));
      checkCudaErrors(cudaFreeAsync(device_recv_vec[i * _QCU_BF_], stream));
      checkCudaErrors(cudaFreeAsync(device_recv_vec[i * _QCU_BF_ + 1], stream));
      checkCudaErrors(cudaFreeAsync(device_u_1dim_send_vec[i * _QCU_BF_], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_1dim_send_vec[i * _QCU_BF_ + 1], stream));
      checkCudaErrors(cudaFreeAsync(device_u_1dim_recv_vec[i * _QCU_BF_], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_1dim_recv_vec[i * _QCU_BF_ + 1], stream));
      free(host_send_vec[i * _QCU_BF_]);
      free(host_recv_vec[i * _QCU_BF_ + 1]);
    }
    for (int i = 0; i < _QCU_2DIM_; i++) {
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_send_vec[i * _QCU_BF_ * _QCU_BF_ + 0], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_recv_vec[i * _QCU_BF_ * _QCU_BF_ + 0], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_send_vec[i * _QCU_BF_ * _QCU_BF_ + 1], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_recv_vec[i * _QCU_BF_ * _QCU_BF_ + 1], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_send_vec[i * _QCU_BF_ * _QCU_BF_ + 2], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_recv_vec[i * _QCU_BF_ * _QCU_BF_ + 2], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_send_vec[i * _QCU_BF_ * _QCU_BF_ + 3], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_recv_vec[i * _QCU_BF_ * _QCU_BF_ + 3], stream));
    }
    CUBLAS_CHECK(cublasDestroy(cublasH));
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaStreamDestroy(stream));
    checkNcclErrors(ncclCommDestroy(nccl_comm));
    // CUDA_CHECK(cudaDeviceReset());// don't use this !
  }
  void _print() {
    printf("gridDim.x               :%d\n", gridDim.x);
    printf("blockDim.x              :%d\n", blockDim.x);
    printf("host_params[_QCU_LAT_X_]    :%d\n", host_params[_QCU_LAT_X_]);
    printf("host_params[_QCU_LAT_Y_]    :%d\n", host_params[_QCU_LAT_Y_]);
    printf("host_params[_QCU_LAT_Z_]    :%d\n", host_params[_QCU_LAT_Z_]);
    printf("host_params[_QCU_LAT_T_]    :%d\n", host_params[_QCU_LAT_T_]);
    printf("host_params[_QCU_LAT_XYZT_] :%d\n", host_params[_QCU_LAT_XYZT_]);
    printf("host_params[_QCU_GRID_X_]   :%d\n", host_params[_QCU_GRID_X_]);
    printf("host_params[_QCU_GRID_Y_]   :%d\n", host_params[_QCU_GRID_Y_]);
    printf("host_params[_QCU_GRID_Z_]   :%d\n", host_params[_QCU_GRID_Z_]);
    printf("host_params[_QCU_GRID_T_]   :%d\n", host_params[_QCU_GRID_T_]);
    printf("host_params[_QCU_PARITY_]   :%d\n", host_params[_QCU_PARITY_]);
    printf("host_params[_QCU_NODE_RANK_]:%d\n", host_params[_QCU_NODE_RANK_]);
    printf("host_params[_QCU_NODE_SIZE_]:%d\n", host_params[_QCU_NODE_SIZE_]);
    printf("lat_2dim[_QCU_XY_]          :%d\n", lat_2dim[_QCU_XY_]);
    printf("lat_2dim[_QCU_XZ_]          :%d\n", lat_2dim[_QCU_XZ_]);
    printf("lat_2dim[_QCU_XT_]          :%d\n", lat_2dim[_QCU_XT_]);
    printf("lat_2dim[_QCU_YZ_]          :%d\n", lat_2dim[_QCU_YZ_]);
    printf("lat_2dim[_QCU_YT_]          :%d\n", lat_2dim[_QCU_YT_]);
    printf("lat_2dim[_QCU_ZT_]          :%d\n", lat_2dim[_QCU_ZT_]);
    printf("lat_3dim[_QCU_YZT_]         :%d\n", lat_3dim[_QCU_YZT_]);
    printf("lat_3dim[_QCU_XZT_]         :%d\n", lat_3dim[_QCU_XZT_]);
    printf("lat_3dim[_QCU_XYT_]         :%d\n", lat_3dim[_QCU_XYT_]);
    printf("lat_3dim[_QCU_XYZ_]         :%d\n", lat_3dim[_QCU_XYZ_]);
    printf("lat_4dim                :%d\n", lat_4dim);
    printf("grid_2dim[_QCU_XY_]         :%d\n", grid_2dim[_QCU_XY_]);
    printf("grid_2dim[_QCU_XZ_]         :%d\n", grid_2dim[_QCU_XZ_]);
    printf("grid_2dim[_QCU_XT_]         :%d\n", grid_2dim[_QCU_XT_]);
    printf("grid_2dim[_QCU_YZ_]         :%d\n", grid_2dim[_QCU_YZ_]);
    printf("grid_2dim[_QCU_YT_]         :%d\n", grid_2dim[_QCU_YT_]);
    printf("grid_2dim[_QCU_ZT_]         :%d\n", grid_2dim[_QCU_ZT_]);
    printf("grid_3dim[_QCU_YZT_]        :%d\n", grid_3dim[_QCU_YZT_]);
    printf("grid_3dim[_QCU_XZT_]        :%d\n", grid_3dim[_QCU_XZT_]);
    printf("grid_3dim[_QCU_XYT_]        :%d\n", grid_3dim[_QCU_XYT_]);
    printf("grid_3dim[_QCU_XYZ_]        :%d\n", grid_3dim[_QCU_XYZ_]);
    printf("grid_index_1dim[_QCU_X_]    :%d\n", grid_index_1dim[_QCU_X_]);
    printf("grid_index_1dim[_QCU_Y_]    :%d\n", grid_index_1dim[_QCU_Y_]);
    printf("grid_index_1dim[_QCU_Z_]    :%d\n", grid_index_1dim[_QCU_Z_]);
    printf("grid_index_1dim[_QCU_T_]    :%d\n", grid_index_1dim[_QCU_T_]);
    printf("move_wards[_QCU_B_X_]       :%d\n", move_wards[_QCU_B_X_]);
    printf("move_wards[_QCU_B_Y_]       :%d\n", move_wards[_QCU_B_Y_]);
    printf("move_wards[_QCU_B_Z_]       :%d\n", move_wards[_QCU_B_Z_]);
    printf("move_wards[_QCU_B_T_]       :%d\n", move_wards[_QCU_B_T_]);
    printf("move_wards[_QCU_F_X_]       :%d\n", move_wards[_QCU_F_X_]);
    printf("move_wards[_QCU_F_Y_]       :%d\n", move_wards[_QCU_F_Y_]);
    printf("move_wards[_QCU_F_Z_]       :%d\n", move_wards[_QCU_F_Z_]);
    printf("move_wards[_QCU_F_T_]       :%d\n", move_wards[_QCU_F_T_]);
    printf("move_wards[_QCU_BX_BY_]     :%d\n", move_wards[_QCU_BX_BY_]);
    printf("move_wards[_QCU_BX_BZ_]     :%d\n", move_wards[_QCU_BX_BZ_]);
    printf("move_wards[_QCU_BX_BT_]     :%d\n", move_wards[_QCU_BX_BT_]);
    printf("move_wards[_QCU_BY_BZ_]     :%d\n", move_wards[_QCU_BY_BZ_]);
    printf("move_wards[_QCU_BY_BT_]     :%d\n", move_wards[_QCU_BY_BT_]);
    printf("move_wards[_QCU_BZ_BT_]     :%d\n", move_wards[_QCU_BZ_BT_]);
    printf("move_wards[_QCU_FX_BY_]     :%d\n", move_wards[_QCU_FX_BY_]);
    printf("move_wards[_QCU_FX_BZ_]     :%d\n", move_wards[_QCU_FX_BZ_]);
    printf("move_wards[_QCU_FX_BT_]     :%d\n", move_wards[_QCU_FX_BT_]);
    printf("move_wards[_QCU_FY_BZ_]     :%d\n", move_wards[_QCU_FY_BZ_]);
    printf("move_wards[_QCU_FY_BT_]     :%d\n", move_wards[_QCU_FY_BT_]);
    printf("move_wards[_QCU_FZ_BT_]     :%d\n", move_wards[_QCU_FZ_BT_]);
    printf("move_wards[_QCU_BX_FY_]     :%d\n", move_wards[_QCU_BX_FY_]);
    printf("move_wards[_QCU_BX_FZ_]     :%d\n", move_wards[_QCU_BX_FZ_]);
    printf("move_wards[_QCU_BX_FT_]     :%d\n", move_wards[_QCU_BX_FT_]);
    printf("move_wards[_QCU_BY_FZ_]     :%d\n", move_wards[_QCU_BY_FZ_]);
    printf("move_wards[_QCU_BY_FT_]     :%d\n", move_wards[_QCU_BY_FT_]);
    printf("move_wards[_QCU_BZ_FT_]     :%d\n", move_wards[_QCU_BZ_FT_]);
    printf("move_wards[_QCU_FX_FY_]     :%d\n", move_wards[_QCU_FX_FY_]);
    printf("move_wards[_QCU_FX_FZ_]     :%d\n", move_wards[_QCU_FX_FZ_]);
    printf("move_wards[_QCU_FX_FT_]     :%d\n", move_wards[_QCU_FX_FT_]);
    printf("move_wards[_QCU_FY_FZ_]     :%d\n", move_wards[_QCU_FY_FZ_]);
    printf("move_wards[_QCU_FY_FT_]     :%d\n", move_wards[_QCU_FY_FT_]);
    printf("move_wards[_QCU_FZ_FT_]     :%d\n", move_wards[_QCU_FZ_FT_]);
  }
};
#endif