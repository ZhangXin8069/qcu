#ifndef _LATTICE_SET_H
#define _LATTICE_SET_H
#pragma once
// clang-format off
#include "./define.h"
// clang-format on
struct LatticeSet {
  int lat_1dim[_DIM_];
  int lat_3dim[_DIM_];
  int lat_4dim;
  int lat_3dim_Half_SC[_DIM_];
  int lat_3dim_SC[_DIM_];
  int lat_4dim_SC;
  dim3 gridDim_3dim[_DIM_];
  dim3 gridDim;
  dim3 blockDim;
  ncclUniqueId nccl_id;
  ncclComm_t nccl_comm;
  cudaStream_t stream;
  cublasHandle_t cublasHs[_DIM_];
  cudaStream_t streams[_DIM_];
  cudaStream_t stream_dims[_DIM_];
  cudaError_t err;
  int node_rank, node_size;
  int move[_BF_];
  int move_wards[_WARDS_];
  int grid_1dim[_DIM_];
  int grid_3dim[_DIM_];
  int grid_index_1dim[_DIM_];
  MPI_Request send_request[_WARDS_];
  MPI_Request recv_request[_WARDS_];
  void *host_send_vec[_WARDS_];
  void *host_recv_vec[_WARDS_];
  int host_xyztsc[_DIM_ * _LAT_C_];
  void *device_send_vec[_WARDS_];
  void *device_recv_vec[_WARDS_];
  void *device_xyztsc;
  void give(int *_param_lat_size, int *_grid_lat_size) {
    lat_1dim[_X_] = _param_lat_size[_X_] / _EVENODD_; // even-odd
    lat_1dim[_Y_] = _param_lat_size[_Y_];
    lat_1dim[_Z_] = _param_lat_size[_Z_];
    lat_1dim[_T_] = _param_lat_size[_T_];
    grid_1dim[_X_] = _grid_lat_size[_X_];
    grid_1dim[_Y_] = _grid_lat_size[_Y_];
    grid_1dim[_Z_] = _grid_lat_size[_Z_];
    grid_1dim[_T_] = _grid_lat_size[_T_];
  }
  void give(int *_param_lat_size) {
    lat_1dim[_X_] = _param_lat_size[_X_] >> 1; // even-odd
    lat_1dim[_Y_] = _param_lat_size[_Y_];
    lat_1dim[_Z_] = _param_lat_size[_Z_];
    lat_1dim[_T_] = _param_lat_size[_T_];
    grid_1dim[_X_] = _GRID_EXAMPLE_;
    grid_1dim[_Y_] = _GRID_EXAMPLE_;
    grid_1dim[_Z_] = _GRID_EXAMPLE_;
    grid_1dim[_T_] = _GRID_EXAMPLE_;
  }
  void give() {
    lat_1dim[_X_] = _LAT_EXAMPLE_;
    lat_1dim[_Y_] = _LAT_EXAMPLE_;
    lat_1dim[_Z_] = _LAT_EXAMPLE_;
    lat_1dim[_T_] = _LAT_EXAMPLE_;
    grid_1dim[_X_] = _GRID_EXAMPLE_;
    grid_1dim[_Y_] = _GRID_EXAMPLE_;
    grid_1dim[_Z_] = _GRID_EXAMPLE_;
    grid_1dim[_T_] = _GRID_EXAMPLE_;
  }
  void init() {
    {
      blockDim = _BLOCK_SIZE_;
      checkMpiErrors(MPI_Comm_rank(MPI_COMM_WORLD, &node_rank));
      checkMpiErrors(MPI_Comm_size(MPI_COMM_WORLD, &node_size));
      if (node_rank == 0) {
        checkNcclErrors(ncclGetUniqueId(&nccl_id));
      }
      checkMpiErrors(MPI_Bcast((void *)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0,
                               MPI_COMM_WORLD));
      checkNcclErrors(
          ncclCommInitRank(&nccl_comm, node_size, nccl_id, node_rank));
      checkCudaErrors(cudaStreamCreate(&stream));
      grid_index_1dim[_X_] =
          node_rank / grid_1dim[_T_] / grid_1dim[_Z_] / grid_1dim[_Y_];
      grid_index_1dim[_Y_] =
          node_rank / grid_1dim[_T_] / grid_1dim[_Z_] % grid_1dim[_Y_];
      grid_index_1dim[_Z_] = node_rank / grid_1dim[_T_] % grid_1dim[_Z_];
      grid_index_1dim[_T_] = node_rank % grid_1dim[_T_];
      grid_3dim[_YZT_] = grid_1dim[_Y_] * grid_1dim[_Z_] * grid_1dim[_T_];
      grid_3dim[_XZT_] = grid_1dim[_X_] * grid_1dim[_Z_] * grid_1dim[_T_];
      grid_3dim[_XYT_] = grid_1dim[_X_] * grid_1dim[_Y_] * grid_1dim[_T_];
      grid_3dim[_XYZ_] = grid_1dim[_X_] * grid_1dim[_Y_] * grid_1dim[_Z_];
      lat_3dim[_YZT_] = lat_1dim[_Y_] * lat_1dim[_Z_] * lat_1dim[_T_];
      lat_3dim[_XZT_] = lat_1dim[_X_] * lat_1dim[_Z_] * lat_1dim[_T_];
      lat_3dim[_XYT_] = lat_1dim[_X_] * lat_1dim[_Y_] * lat_1dim[_T_];
      lat_3dim[_XYZ_] = lat_1dim[_X_] * lat_1dim[_Y_] * lat_1dim[_Z_];
      gridDim_3dim[_YZT_] = lat_3dim[_YZT_] / _BLOCK_SIZE_;
      gridDim_3dim[_XZT_] = lat_3dim[_XZT_] / _BLOCK_SIZE_;
      gridDim_3dim[_XYT_] = lat_3dim[_XYT_] / _BLOCK_SIZE_;
      gridDim_3dim[_XYZ_] = lat_3dim[_XYZ_] / _BLOCK_SIZE_;
      lat_4dim = lat_3dim[_XYZ_] * lat_1dim[_T_];
      lat_4dim_SC = lat_4dim * _LAT_SC_;
      gridDim = lat_4dim / _BLOCK_SIZE_;
    }
    {
      move_backward(move_wards[_B_X_], grid_index_1dim[_X_], grid_1dim[_X_]);
      move_backward(move_wards[_B_Y_], grid_index_1dim[_Y_], grid_1dim[_Y_]);
      move_backward(move_wards[_B_Z_], grid_index_1dim[_Z_], grid_1dim[_Z_]);
      move_backward(move_wards[_B_T_], grid_index_1dim[_T_], grid_1dim[_T_]);
      move_forward(move_wards[_F_X_], grid_index_1dim[_X_], grid_1dim[_X_]);
      move_forward(move_wards[_F_Y_], grid_index_1dim[_Y_], grid_1dim[_Y_]);
      move_forward(move_wards[_F_Z_], grid_index_1dim[_Z_], grid_1dim[_Z_]);
      move_forward(move_wards[_F_T_], grid_index_1dim[_T_], grid_1dim[_T_]);
      move_wards[_B_X_] = node_rank + move_wards[_B_X_] * grid_3dim[_YZT_];
      move_wards[_B_Y_] = node_rank + move_wards[_B_Y_] * grid_3dim[_XZT_];
      move_wards[_B_Z_] = node_rank + move_wards[_B_Z_] * grid_3dim[_XYT_];
      move_wards[_B_T_] = node_rank + move_wards[_B_T_] * grid_3dim[_XYZ_];
      move_wards[_F_X_] = node_rank + move_wards[_F_X_] * grid_3dim[_YZT_];
      move_wards[_F_Y_] = node_rank + move_wards[_F_Y_] * grid_3dim[_XZT_];
      move_wards[_F_Z_] = node_rank + move_wards[_F_Z_] * grid_3dim[_XYT_];
      move_wards[_F_T_] = node_rank + move_wards[_F_T_] * grid_3dim[_XYZ_];
    }
    {
      for (int i = 0; i < _DIM_; i++) {
        CUBLAS_CHECK(cublasCreate(&cublasHs[i]));
        checkCudaErrors(
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        // checkCudaErrors(cudaStreamCreate(&streams[i]));
        CUBLAS_CHECK(cublasSetStream(cublasHs[i], streams[i]));
        checkCudaErrors(cudaStreamCreate(&stream_dims[i]));
        lat_3dim_Half_SC[i] = lat_3dim[i] * _LAT_HALF_SC_;
        lat_3dim_SC[i] = lat_3dim_Half_SC[i] * 2;
        checkCudaErrors(cudaMallocAsync(
            &device_send_vec[i * _SR_],
            lat_3dim_Half_SC[i] * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_send_vec[i * _SR_ + 1],
            lat_3dim_Half_SC[i] * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_recv_vec[i * _SR_],
            lat_3dim_Half_SC[i] * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_recv_vec[i * _SR_ + 1],
            lat_3dim_Half_SC[i] * sizeof(LatticeComplex), stream));
        host_send_vec[i * _SR_] =
            (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));
        host_send_vec[i * _SR_ + 1] =
            (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));
        host_recv_vec[i * _SR_] =
            (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));
        host_recv_vec[i * _SR_ + 1] =
            (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));
      }
    }
    {
      checkCudaErrors(cudaMallocAsync(&device_xyztsc,
                                      _DIM_ * _LAT_C_ * sizeof(int), stream));
      host_xyztsc[_X_] = lat_1dim[_X_];
      host_xyztsc[_Y_] = lat_1dim[_Y_];
      host_xyztsc[_Z_] = lat_1dim[_Z_];
      host_xyztsc[_T_] = lat_1dim[_T_];
      host_xyztsc[_XCC_] = lat_1dim[_X_] * _LAT_CC_;
      host_xyztsc[_YXCC_] = lat_1dim[_Y_] * host_xyztsc[_XCC_];
      host_xyztsc[_ZYXCC_] = lat_1dim[_Z_] * host_xyztsc[_YXCC_];
      host_xyztsc[_TZYXCC_] = lat_1dim[_T_] * host_xyztsc[_ZYXCC_];
      host_xyztsc[_XSC_] = lat_1dim[_X_] * _LAT_SC_;
      host_xyztsc[_YXSC_] = lat_1dim[_Y_] * host_xyztsc[_XSC_];
      host_xyztsc[_ZYXSC_] = lat_1dim[_Z_] * host_xyztsc[_YXSC_];
      host_xyztsc[_TZYXSC_] = lat_1dim[_T_] * host_xyztsc[_ZYXSC_];
      checkCudaErrors(cudaMemcpyAsync(device_xyztsc, host_xyztsc,
                                      _DIM_ * _LAT_C_ * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));
    }
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  void end() {
    for (int i = 0; i < _DIM_; i++) {
      CUBLAS_CHECK(cublasDestroy(cublasHs[i]));
      checkCudaErrors(cudaStreamDestroy(streams[i]));
      checkCudaErrors(cudaStreamDestroy(stream_dims[i]));
      checkCudaErrors(cudaFreeAsync(device_send_vec[i * _SR_], stream));
      checkCudaErrors(cudaFreeAsync(device_send_vec[i * _SR_ + 1], stream));
      checkCudaErrors(cudaFreeAsync(device_recv_vec[i * _SR_], stream));
      checkCudaErrors(cudaFreeAsync(device_recv_vec[i * _SR_ + 1], stream));
      free(host_send_vec[i * _SR_]);
      free(host_recv_vec[i * _SR_ + 1]);
    }
    checkCudaErrors(cudaFreeAsync(device_xyztsc, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaStreamDestroy(stream));
    checkNcclErrors(ncclCommDestroy(nccl_comm));
    // CUDA_CHECK(cudaDeviceReset());// don't use this !
  }
  void _print() {
    printf("node_rank        :%d\n", node_rank);
    printf("node_size        :%d\n", node_size);
    printf("gridDim.x        :%d\n", gridDim.x);
    printf("blockDim.x       :%d\n", blockDim.x);
    printf("lat_1dim[_X_]    :%d\n", lat_1dim[_X_]);
    printf("lat_1dim[_Y_]    :%d\n", lat_1dim[_Y_]);
    printf("lat_1dim[_Z_]    :%d\n", lat_1dim[_Z_]);
    printf("lat_1dim[_T_]    :%d\n", lat_1dim[_T_]);
    printf("grid_1dim[_X_]   :%d\n", grid_1dim[_X_]);
    printf("grid_1dim[_Y_]   :%d\n", grid_1dim[_Y_]);
    printf("grid_1dim[_Z_]   :%d\n", grid_1dim[_Z_]);
    printf("grid_1dim[_T_]   :%d\n", grid_1dim[_T_]);
    printf("lat_3dim[_YZT_]  :%d\n", lat_3dim[_YZT_]);
    printf("lat_3dim[_XZT_]  :%d\n", lat_3dim[_XZT_]);
    printf("lat_3dim[_XYT_]  :%d\n", lat_3dim[_XYT_]);
    printf("lat_3dim[_XYZ_]  :%d\n", lat_3dim[_XYZ_]);
    printf("lat_4dim         :%d\n", lat_4dim);
    printf("lat_4dim_SC       :%d\n", lat_4dim_SC);
    printf("lat_3dim_Half_SC[_YZT_] :%d\n", lat_3dim_Half_SC[_YZT_]);
    printf("lat_3dim_Half_SC[_XZT_] :%d\n", lat_3dim_Half_SC[_XZT_]);
    printf("lat_3dim_Half_SC[_XYT_] :%d\n", lat_3dim_Half_SC[_XYT_]);
    printf("lat_3dim_Half_SC[_XYZ_] :%d\n", lat_3dim_Half_SC[_XYZ_]);
    printf("lat_3dim_SC[_YZT_]:%d\n", lat_3dim_SC[_YZT_]);
    printf("lat_3dim_SC[_XZT_]:%d\n", lat_3dim_SC[_XZT_]);
    printf("lat_3dim_SC[_XYT_]:%d\n", lat_3dim_SC[_XYT_]);
    printf("lat_3dim_SC[_XYZ_]:%d\n", lat_3dim_SC[_XYZ_]);
  }
};
#endif