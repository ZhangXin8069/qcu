#ifndef _LATTICE_SET_H
#define _LATTICE_SET_H
#include "./lattice_cuda.h"
#pragma once
// clang-format off
#include "./define.h"
// clang-format on
__global__ void give_param(void *device_param, const int vals_index, int val);
struct LatticeSet {
  int lat_2dim[_2DIM_];
  int lat_3dim[_3DIM_];
  int lat_4dim;
  int lat_3dim_Half_SC[_3DIM_];
  int lat_3dim_SC[_3DIM_];
  int lat_4dim_SC;
  int lat_4dim_DCC;
  dim3 gridDim_3dim[_3DIM_];
  dim3 gridDim_2dim[_2DIM_];
  dim3 gridDim;
  dim3 blockDim;
  ncclUniqueId nccl_id;
  ncclComm_t nccl_comm;
  cublasHandle_t cublasH;
  cudaStream_t stream;
  cublasHandle_t cublasHs[_DIM_];
  cudaStream_t streams[_DIM_];
  cudaStream_t stream_dims[_DIM_];
  float time;
  cudaEvent_t start, stop;
  cudaError_t err;
  int node_rank;
  int node_size;
  int move[_BF_];
  int move_wards[_WARDS_ + _WARDS_2DIM_];
  int grid_2dim[_2DIM_];
  int grid_3dim[_3DIM_];
  int grid_index_1dim[_1DIM_];
  MPI_Request send_request[_WARDS_];
  MPI_Request recv_request[_WARDS_];
  void *host_send_vec[_WARDS_];
  void *host_recv_vec[_WARDS_];
  int host_params[_VALS_SIZE_];
  void *device_send_vec[_WARDS_];
  void *device_recv_vec[_WARDS_];
  void *device_u_1dim_send_vec[_WARDS_];
  void *device_u_1dim_recv_vec[_WARDS_];
  void *device_u_2dim_send_vec[_2DIM_ * _BF_ * _BF_];
  void *device_u_2dim_recv_vec[_2DIM_ * _BF_ * _BF_];
  void *device_params;
  void give(int *_param_lat_size, int *_grid_lat_size) {
    host_params[_LAT_X_] = _param_lat_size[_X_] / _EVEN_ODD_; // even-odd
    host_params[_LAT_Y_] = _param_lat_size[_Y_];
    host_params[_LAT_Z_] = _param_lat_size[_Z_];
    host_params[_LAT_T_] = _param_lat_size[_T_];
    host_params[_GRID_X_] = _grid_lat_size[_X_];
    host_params[_GRID_Y_] = _grid_lat_size[_Y_];
    host_params[_GRID_Z_] = _grid_lat_size[_Z_];
    host_params[_GRID_T_] = _grid_lat_size[_T_];
  }
  void give(int *_param_lat_size, int *_grid_lat_size, int parity) {
    host_params[_LAT_X_] = _param_lat_size[_X_] / _EVEN_ODD_; // even-odd
    host_params[_LAT_Y_] = _param_lat_size[_Y_];
    host_params[_LAT_Z_] = _param_lat_size[_Z_];
    host_params[_LAT_T_] = _param_lat_size[_T_];
    host_params[_GRID_X_] = _grid_lat_size[_X_];
    host_params[_GRID_Y_] = _grid_lat_size[_Y_];
    host_params[_GRID_Z_] = _grid_lat_size[_Z_];
    host_params[_GRID_T_] = _grid_lat_size[_T_];
    host_params[_PARITY_] = parity;
  }
  void give(int *_param_lat_size, int parity) {
    host_params[_LAT_X_] = _param_lat_size[_X_] / _EVEN_ODD_; // even-odd
    host_params[_LAT_Y_] = _param_lat_size[_Y_];
    host_params[_LAT_Z_] = _param_lat_size[_Z_];
    host_params[_LAT_T_] = _param_lat_size[_T_];
    host_params[_GRID_X_] = _GRID_EXAMPLE_;
    host_params[_GRID_Y_] = _GRID_EXAMPLE_;
    host_params[_GRID_Z_] = _GRID_EXAMPLE_;
    host_params[_GRID_T_] = _GRID_EXAMPLE_;
    host_params[_PARITY_] = parity;
  }
  void give(int parity) {
    host_params[_LAT_X_] = _LAT_EXAMPLE_;
    host_params[_LAT_Y_] = _LAT_EXAMPLE_;
    host_params[_LAT_Z_] = _LAT_EXAMPLE_;
    host_params[_LAT_T_] = _LAT_EXAMPLE_;
    host_params[_GRID_X_] = _GRID_EXAMPLE_;
    host_params[_GRID_Y_] = _GRID_EXAMPLE_;
    host_params[_GRID_Z_] = _GRID_EXAMPLE_;
    host_params[_GRID_T_] = _GRID_EXAMPLE_;
    host_params[_PARITY_] = parity;
  }
  void init() {
    {
      blockDim = _BLOCK_SIZE_;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);
      cudaEventSynchronize(start);
      checkMpiErrors(MPI_Comm_rank(MPI_COMM_WORLD, &node_rank));
      checkMpiErrors(MPI_Comm_size(MPI_COMM_WORLD, &node_size));
      grid_index_1dim[_X_] = node_rank / host_params[_GRID_T_] /
                             host_params[_GRID_Z_] / host_params[_GRID_Y_];
      grid_index_1dim[_Y_] = node_rank / host_params[_GRID_T_] /
                             host_params[_GRID_Z_] % host_params[_GRID_Y_];
      grid_index_1dim[_Z_] =
          node_rank / host_params[_GRID_T_] % host_params[_GRID_Z_];
      grid_index_1dim[_T_] = node_rank % host_params[_GRID_T_];
      grid_2dim[_XY_] = host_params[_GRID_X_] * host_params[_GRID_Y_];
      grid_2dim[_XZ_] = host_params[_GRID_X_] * host_params[_GRID_Z_];
      grid_2dim[_XT_] = host_params[_GRID_X_] * host_params[_GRID_T_];
      grid_2dim[_YZ_] = host_params[_GRID_Y_] * host_params[_GRID_Z_];
      grid_2dim[_YT_] = host_params[_GRID_Y_] * host_params[_GRID_T_];
      grid_2dim[_ZT_] = host_params[_GRID_Z_] * host_params[_GRID_T_];
      grid_3dim[_YZT_] =
          host_params[_GRID_Y_] * host_params[_GRID_Z_] * host_params[_GRID_T_];
      grid_3dim[_XZT_] =
          host_params[_GRID_X_] * host_params[_GRID_Z_] * host_params[_GRID_T_];
      grid_3dim[_XYT_] =
          host_params[_GRID_X_] * host_params[_GRID_Y_] * host_params[_GRID_T_];
      grid_3dim[_XYZ_] =
          host_params[_GRID_X_] * host_params[_GRID_Y_] * host_params[_GRID_Z_];
      lat_2dim[_XY_] = host_params[_LAT_X_] * host_params[_LAT_Y_];
      lat_2dim[_XZ_] = host_params[_LAT_X_] * host_params[_LAT_Z_];
      lat_2dim[_XT_] = host_params[_LAT_X_] * host_params[_LAT_T_];
      lat_2dim[_YZ_] = host_params[_LAT_Y_] * host_params[_LAT_Z_];
      lat_2dim[_YT_] = host_params[_LAT_Y_] * host_params[_LAT_T_];
      lat_2dim[_ZT_] = host_params[_LAT_Z_] * host_params[_LAT_T_];
      gridDim_2dim[_XY_] = lat_2dim[_XY_] / _BLOCK_SIZE_;
      gridDim_2dim[_XZ_] = lat_2dim[_XZ_] / _BLOCK_SIZE_;
      gridDim_2dim[_XT_] = lat_2dim[_XT_] / _BLOCK_SIZE_;
      gridDim_2dim[_YZ_] = lat_2dim[_YZ_] / _BLOCK_SIZE_;
      gridDim_2dim[_YT_] = lat_2dim[_YT_] / _BLOCK_SIZE_;
      gridDim_2dim[_ZT_] = lat_2dim[_ZT_] / _BLOCK_SIZE_;
      lat_3dim[_YZT_] =
          host_params[_LAT_Y_] * host_params[_LAT_Z_] * host_params[_LAT_T_];
      lat_3dim[_XZT_] =
          host_params[_LAT_X_] * host_params[_LAT_Z_] * host_params[_LAT_T_];
      lat_3dim[_XYT_] =
          host_params[_LAT_X_] * host_params[_LAT_Y_] * host_params[_LAT_T_];
      lat_3dim[_XYZ_] =
          host_params[_LAT_X_] * host_params[_LAT_Y_] * host_params[_LAT_Z_];
      gridDim_3dim[_YZT_] = lat_3dim[_YZT_] / _BLOCK_SIZE_;
      gridDim_3dim[_XZT_] = lat_3dim[_XZT_] / _BLOCK_SIZE_;
      gridDim_3dim[_XYT_] = lat_3dim[_XYT_] / _BLOCK_SIZE_;
      gridDim_3dim[_XYZ_] = lat_3dim[_XYZ_] / _BLOCK_SIZE_;
      lat_4dim = lat_3dim[_XYZ_] * host_params[_LAT_T_];
      lat_4dim_SC = lat_4dim * _LAT_SC_;
      lat_4dim_DCC = lat_4dim * _LAT_DCC_;
      gridDim = lat_4dim / _BLOCK_SIZE_;
    }
    {
      move_backward(move_wards[_B_X_], grid_index_1dim[_X_],
                    host_params[_GRID_X_]);
      move_backward(move_wards[_B_Y_], grid_index_1dim[_Y_],
                    host_params[_GRID_Y_]);
      move_backward(move_wards[_B_Z_], grid_index_1dim[_Z_],
                    host_params[_GRID_Z_]);
      move_backward(move_wards[_B_T_], grid_index_1dim[_T_],
                    host_params[_GRID_T_]);
      move_forward(move_wards[_F_X_], grid_index_1dim[_X_],
                   host_params[_GRID_X_]);
      move_forward(move_wards[_F_Y_], grid_index_1dim[_Y_],
                   host_params[_GRID_Y_]);
      move_forward(move_wards[_F_Z_], grid_index_1dim[_Z_],
                   host_params[_GRID_Z_]);
      move_forward(move_wards[_F_T_], grid_index_1dim[_T_],
                   host_params[_GRID_T_]);
      move_wards[_B_X_] = node_rank + move_wards[_B_X_];
      move_wards[_B_Y_] = node_rank + move_wards[_B_Y_] * host_params[_GRID_X_];
      move_wards[_B_Z_] = node_rank + move_wards[_B_Z_] * grid_2dim[_XY_];
      move_wards[_B_T_] = node_rank + move_wards[_B_T_] * grid_3dim[_XYZ_];
      move_wards[_F_X_] = node_rank + move_wards[_F_X_];
      move_wards[_F_Y_] = node_rank + move_wards[_F_Y_] * host_params[_GRID_X_];
      move_wards[_F_Z_] = node_rank + move_wards[_F_Z_] * grid_2dim[_XY_];
      move_wards[_F_T_] = node_rank + move_wards[_F_T_] * grid_3dim[_XYZ_];
      int tmp;
      { // BB
        move_backward(tmp, grid_index_1dim[_Y_], host_params[_GRID_Y_]);
        move_wards[_BX_BY_] = move_wards[_B_X_] + tmp * host_params[_GRID_X_];
        move_backward(tmp, grid_index_1dim[_Z_], host_params[_GRID_Z_]);
        move_wards[_BX_BZ_] = move_wards[_B_X_] + tmp * grid_2dim[_XY_];
        move_backward(tmp, grid_index_1dim[_T_], host_params[_GRID_T_]);
        move_wards[_BX_BT_] = move_wards[_B_X_] + tmp * grid_3dim[_XYZ_];
        move_backward(tmp, grid_index_1dim[_Z_], host_params[_GRID_Z_]);
        move_wards[_BY_BZ_] = move_wards[_B_Y_] + tmp * grid_2dim[_XY_];
        move_backward(tmp, grid_index_1dim[_T_], host_params[_GRID_T_]);
        move_wards[_BY_BT_] = move_wards[_B_Y_] + tmp * grid_3dim[_XYZ_];
        move_backward(tmp, grid_index_1dim[_T_], host_params[_GRID_T_]);
        move_wards[_BZ_BT_] = move_wards[_B_Z_] + tmp * grid_3dim[_XYZ_];
      }
      { // FB
        move_backward(tmp, grid_index_1dim[_Y_], host_params[_GRID_Y_]);
        move_wards[_FX_BY_] = move_wards[_F_X_] + tmp * host_params[_GRID_X_];
        move_backward(tmp, grid_index_1dim[_Z_], host_params[_GRID_Z_]);
        move_wards[_FX_BZ_] = move_wards[_F_X_] + tmp * grid_2dim[_XY_];
        move_backward(tmp, grid_index_1dim[_T_], host_params[_GRID_T_]);
        move_wards[_FX_BT_] = move_wards[_F_X_] + tmp * grid_3dim[_XYZ_];
        move_backward(tmp, grid_index_1dim[_Z_], host_params[_GRID_Z_]);
        move_wards[_FY_BZ_] = move_wards[_F_Y_] + tmp * grid_2dim[_XY_];
        move_backward(tmp, grid_index_1dim[_T_], host_params[_GRID_T_]);
        move_wards[_FY_BT_] = move_wards[_F_Y_] + tmp * grid_3dim[_XYZ_];
        move_backward(tmp, grid_index_1dim[_T_], host_params[_GRID_T_]);
        move_wards[_FZ_BT_] = move_wards[_F_Z_] + tmp * grid_3dim[_XYZ_];
      }
      { // BF
        move_forward(tmp, grid_index_1dim[_Y_], host_params[_GRID_Y_]);
        move_wards[_BX_FY_] = move_wards[_B_X_] + tmp * host_params[_GRID_X_];
        move_forward(tmp, grid_index_1dim[_Z_], host_params[_GRID_Z_]);
        move_wards[_BX_FZ_] = move_wards[_B_X_] + tmp * grid_2dim[_XY_];
        move_forward(tmp, grid_index_1dim[_T_], host_params[_GRID_T_]);
        move_wards[_BX_FT_] = move_wards[_B_X_] + tmp * grid_3dim[_XYZ_];
        move_forward(tmp, grid_index_1dim[_Z_], host_params[_GRID_Z_]);
        move_wards[_BY_FZ_] = move_wards[_B_Y_] + tmp * grid_2dim[_XY_];
        move_forward(tmp, grid_index_1dim[_T_], host_params[_GRID_T_]);
        move_wards[_BY_FT_] = move_wards[_B_Y_] + tmp * grid_3dim[_XYZ_];
        move_forward(tmp, grid_index_1dim[_T_], host_params[_GRID_T_]);
        move_wards[_BZ_FT_] = move_wards[_B_Z_] + tmp * grid_3dim[_XYZ_];
      }
      { // FF
        move_forward(tmp, grid_index_1dim[_Y_], host_params[_GRID_Y_]);
        move_wards[_FX_FY_] = move_wards[_F_X_] + tmp * host_params[_GRID_X_];
        move_forward(tmp, grid_index_1dim[_Z_], host_params[_GRID_Z_]);
        move_wards[_FX_FZ_] = move_wards[_F_X_] + tmp * grid_2dim[_XY_];
        move_forward(tmp, grid_index_1dim[_T_], host_params[_GRID_T_]);
        move_wards[_FX_FT_] = move_wards[_F_X_] + tmp * grid_3dim[_XYZ_];
        move_forward(tmp, grid_index_1dim[_Z_], host_params[_GRID_Z_]);
        move_wards[_FY_FZ_] = move_wards[_F_Y_] + tmp * grid_2dim[_XY_];
        move_forward(tmp, grid_index_1dim[_T_], host_params[_GRID_T_]);
        move_wards[_FY_FT_] = move_wards[_F_Y_] + tmp * grid_3dim[_XYZ_];
        move_forward(tmp, grid_index_1dim[_T_], host_params[_GRID_T_]);
        move_wards[_FZ_FT_] = move_wards[_F_Z_] + tmp * grid_3dim[_XYZ_];
      }
    }
    {
      // nccl set
      if (node_rank == 0) {
        checkNcclErrors(ncclGetUniqueId(&nccl_id));
      }
      checkMpiErrors(MPI_Bcast((void *)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0,
                               MPI_COMM_WORLD));
      checkNcclErrors(
          ncclCommInitRank(&nccl_comm, node_size, nccl_id, node_rank));
    }
    { // set stream and malloc vec
      CUBLAS_CHECK(cublasCreate(&cublasH));
      checkCudaErrors(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUBLAS_CHECK(cublasSetStream(cublasH, stream));
      for (int i = 0; i < _DIM_; i++) {
        CUBLAS_CHECK(cublasCreate(&cublasHs[i]));
        checkCudaErrors(
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        // checkCudaErrors(cudaStreamCreate(&streams[i]));
        checkCudaErrors(
            cudaStreamCreateWithFlags(&stream_dims[i], cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(cublasHs[i], streams[i]));
        lat_3dim_Half_SC[i] = lat_3dim[i] * _LAT_HALF_SC_;
        lat_3dim_SC[i] = lat_3dim_Half_SC[i] * 2;
        checkCudaErrors(cudaMallocAsync(
            &device_send_vec[i * _BF_],
            lat_3dim_Half_SC[i] * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_send_vec[i * _BF_ + 1],
            lat_3dim_Half_SC[i] * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_recv_vec[i * _BF_],
            lat_3dim_Half_SC[i] * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_recv_vec[i * _BF_ + 1],
            lat_3dim_Half_SC[i] * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_1dim_send_vec[i * _BF_],
            lat_3dim[i] * _LAT_PDCC_ * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_1dim_send_vec[i * _BF_ + 1],
            lat_3dim[i] * _LAT_PDCC_ * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_1dim_recv_vec[i * _BF_],
            lat_3dim[i] * _LAT_PDCC_ * sizeof(LatticeComplex), stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_1dim_recv_vec[i * _BF_ + 1],
            lat_3dim[i] * _LAT_PDCC_ * sizeof(LatticeComplex), stream));
        host_send_vec[i * _BF_] =
            (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));
        host_send_vec[i * _BF_ + 1] =
            (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));
        host_recv_vec[i * _BF_] =
            (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));
        host_recv_vec[i * _BF_ + 1] =
            (void *)malloc(lat_3dim_Half_SC[i] * sizeof(LatticeComplex));
      }
      for (int i = 0; i < _2DIM_; i++) {
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_send_vec[i * _BF_ * _BF_ + 0],
            lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_recv_vec[i * _BF_ * _BF_ + 0],
            lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_send_vec[i * _BF_ * _BF_ + 1],
            lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_recv_vec[i * _BF_ * _BF_ + 1],
            lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_send_vec[i * _BF_ * _BF_ + 2],
            lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_recv_vec[i * _BF_ * _BF_ + 2],
            lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_send_vec[i * _BF_ * _BF_ + 3],
            lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
        checkCudaErrors(cudaMallocAsync(
            &device_u_2dim_recv_vec[i * _BF_ * _BF_ + 3],
            lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
            stream));
      }
    }
    {
      checkCudaErrors(
          cudaMallocAsync(&device_params, _VALS_SIZE_ * sizeof(int), stream));
      host_params[_LAT_X_] = host_params[_LAT_X_];
      host_params[_LAT_Y_] = host_params[_LAT_Y_];
      host_params[_LAT_Z_] = host_params[_LAT_Z_];
      host_params[_LAT_T_] = host_params[_LAT_T_];
      host_params[_LAT_XYZT_] = host_params[_LAT_X_] * host_params[_LAT_Y_] *
                                host_params[_LAT_Z_] * host_params[_LAT_T_];
      host_params[_NODE_RANK_] = node_rank;
      checkCudaErrors(cudaMemcpyAsync(device_params, host_params,
                                      _VALS_SIZE_ * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));
    }
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  void use_even() {
    checkCudaErrors(cudaStreamSynchronize(stream));
    give_param<<<1, 1, 0, stream>>>(device_params, _PARITY_, _EVEN_);
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  void use_odd() {
    checkCudaErrors(cudaStreamSynchronize(stream));
    give_param<<<1, 1, 0, stream>>>(device_params, _PARITY_, _ODD_);
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  void use_parity(const int parity) {
    checkCudaErrors(cudaStreamSynchronize(stream));
    give_param<<<1, 1, 0, stream>>>(device_params, _PARITY_, parity);
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  void use_dagger() {
    checkCudaErrors(cudaStreamSynchronize(stream));
    give_param<<<1, 1, 0, stream>>>(device_params, _DAGGER_, _USE_);
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  void no_use_dagger() {
    checkCudaErrors(cudaStreamSynchronize(stream));
    give_param<<<1, 1, 0, stream>>>(device_params, _DAGGER_, _NO_USE_);
    checkCudaErrors(cudaStreamSynchronize(stream));
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
    for (int i = 0; i < _DIM_; i++) {
      checkCudaErrors(cudaStreamSynchronize(streams[i]));
      checkCudaErrors(cudaStreamSynchronize(stream_dims[i]));
      CUBLAS_CHECK(cublasDestroy(cublasHs[i]));
      checkCudaErrors(cudaStreamDestroy(streams[i]));
      checkCudaErrors(cudaStreamDestroy(stream_dims[i]));
      checkCudaErrors(cudaFreeAsync(device_send_vec[i * _BF_], stream));
      checkCudaErrors(cudaFreeAsync(device_send_vec[i * _BF_ + 1], stream));
      checkCudaErrors(cudaFreeAsync(device_recv_vec[i * _BF_], stream));
      checkCudaErrors(cudaFreeAsync(device_recv_vec[i * _BF_ + 1], stream));
      checkCudaErrors(cudaFreeAsync(device_u_1dim_send_vec[i * _BF_], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_1dim_send_vec[i * _BF_ + 1], stream));
      checkCudaErrors(cudaFreeAsync(device_u_1dim_recv_vec[i * _BF_], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_1dim_recv_vec[i * _BF_ + 1], stream));
      free(host_send_vec[i * _BF_]);
      free(host_recv_vec[i * _BF_ + 1]);
    }
    for (int i = 0; i < _2DIM_; i++) {
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_send_vec[i * _BF_ * _BF_ + 0], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_recv_vec[i * _BF_ * _BF_ + 0], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_send_vec[i * _BF_ * _BF_ + 1], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_recv_vec[i * _BF_ * _BF_ + 1], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_send_vec[i * _BF_ * _BF_ + 2], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_recv_vec[i * _BF_ * _BF_ + 2], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_send_vec[i * _BF_ * _BF_ + 3], stream));
      checkCudaErrors(
          cudaFreeAsync(device_u_2dim_recv_vec[i * _BF_ * _BF_ + 3], stream));
    }
    CUBLAS_CHECK(cublasDestroy(cublasH));
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaStreamDestroy(stream));
    checkNcclErrors(ncclCommDestroy(nccl_comm));
    // CUDA_CHECK(cudaDeviceReset());// don't use this !
  }
  // clang-format off
  void _print() {
    printf("node_rank            :%d\n", node_rank);
    printf("node_size            :%d\n", node_size);
    printf("gridDim.x            :%d\n", gridDim.x);
    printf("blockDim.x           :%d\n", blockDim.x);
    printf("host_params[_LAT_X_] :%d\n", host_params[_LAT_X_]);
    printf("host_params[_LAT_Y_] :%d\n", host_params[_LAT_Y_]);
    printf("host_params[_LAT_Z_] :%d\n", host_params[_LAT_Z_]);
    printf("host_params[_LAT_T_] :%d\n", host_params[_LAT_T_]);
    printf("lat_2dim[_XY_]       :%d\n", lat_2dim[_XY_]);
    printf("lat_2dim[_XZ_]       :%d\n", lat_2dim[_XZ_]);
    printf("lat_2dim[_XT_]       :%d\n", lat_2dim[_XT_]);
    printf("lat_2dim[_YZ_]       :%d\n", lat_2dim[_YZ_]);
    printf("lat_2dim[_YT_]       :%d\n", lat_2dim[_YT_]);
    printf("lat_2dim[_ZT_]       :%d\n", lat_2dim[_ZT_]);
    printf("lat_3dim[_YZT_]      :%d\n", lat_3dim[_YZT_]);
    printf("lat_3dim[_XZT_]      :%d\n", lat_3dim[_XZT_]);
    printf("lat_3dim[_XYT_]      :%d\n", lat_3dim[_XYT_]);
    printf("lat_3dim[_XYZ_]      :%d\n", lat_3dim[_XYZ_]);
    printf("lat_4dim             :%d\n", lat_4dim);
    printf("host_params[_GRID_X_]:%d\n", host_params[_GRID_X_]);
    printf("host_params[_GRID_Y_]:%d\n", host_params[_GRID_Y_]);
    printf("host_params[_GRID_Z_]:%d\n", host_params[_GRID_Z_]);
    printf("host_params[_GRID_T_]:%d\n", host_params[_GRID_T_]);
    printf("grid_2dim[_XY_]      :%d\n", grid_2dim[_XY_]);
    printf("grid_2dim[_XZ_]      :%d\n", grid_2dim[_XZ_]);
    printf("grid_2dim[_XT_]      :%d\n", grid_2dim[_XT_]);
    printf("grid_2dim[_YZ_]      :%d\n", grid_2dim[_YZ_]);
    printf("grid_2dim[_YT_]      :%d\n", grid_2dim[_YT_]);
    printf("grid_2dim[_ZT_]      :%d\n", grid_2dim[_ZT_]);
    printf("grid_3dim[_YZT_]     :%d\n", grid_3dim[_YZT_]);
    printf("grid_3dim[_XZT_]     :%d\n", grid_3dim[_XZT_]);
    printf("grid_3dim[_XYT_]     :%d\n", grid_3dim[_XYT_]);
    printf("grid_3dim[_XYZ_]     :%d\n", grid_3dim[_XYZ_]);
    printf("grid_index_1dim[_X_] :%d\n", grid_index_1dim[_X_]);
    printf("grid_index_1dim[_Y_] :%d\n", grid_index_1dim[_Y_]);
    printf("grid_index_1dim[_Z_] :%d\n", grid_index_1dim[_Z_]);
    printf("grid_index_1dim[_T_] :%d\n", grid_index_1dim[_T_]);
    printf("move_wards[_B_X_]    :%d\n", move_wards[_B_X_]);
    printf("move_wards[_B_Y_]    :%d\n", move_wards[_B_Y_]);
    printf("move_wards[_B_Z_]    :%d\n", move_wards[_B_Z_]);
    printf("move_wards[_B_T_]    :%d\n", move_wards[_B_T_]);
    printf("move_wards[_F_X_]    :%d\n", move_wards[_F_X_]);
    printf("move_wards[_F_Y_]    :%d\n", move_wards[_F_Y_]);
    printf("move_wards[_F_Z_]    :%d\n", move_wards[_F_Z_]);
    printf("move_wards[_F_T_]    :%d\n", move_wards[_F_T_]);
    printf("move_wards[_BX_BY_]  :%d\n", move_wards[_BX_BY_]);
    printf("move_wards[_BX_BZ_]  :%d\n", move_wards[_BX_BZ_]);
    printf("move_wards[_BX_BT_]  :%d\n", move_wards[_BX_BT_]);
    printf("move_wards[_BY_BZ_]  :%d\n", move_wards[_BY_BZ_]);
    printf("move_wards[_BY_BT_]  :%d\n", move_wards[_BY_BT_]);
    printf("move_wards[_BZ_BT_]  :%d\n", move_wards[_BZ_BT_]);
    printf("move_wards[_FX_BY_]  :%d\n", move_wards[_FX_BY_]);
    printf("move_wards[_FX_BZ_]  :%d\n", move_wards[_FX_BZ_]);
    printf("move_wards[_FX_BT_]  :%d\n", move_wards[_FX_BT_]);
    printf("move_wards[_FY_BZ_]  :%d\n", move_wards[_FY_BZ_]);
    printf("move_wards[_FY_BT_]  :%d\n", move_wards[_FY_BT_]);
    printf("move_wards[_FZ_BT_]  :%d\n", move_wards[_FZ_BT_]);
    printf("move_wards[_BX_FY_]  :%d\n", move_wards[_BX_FY_]);
    printf("move_wards[_BX_FZ_]  :%d\n", move_wards[_BX_FZ_]);
    printf("move_wards[_BX_FT_]  :%d\n", move_wards[_BX_FT_]);
    printf("move_wards[_BY_FZ_]  :%d\n", move_wards[_BY_FZ_]);
    printf("move_wards[_BY_FT_]  :%d\n", move_wards[_BY_FT_]);
    printf("move_wards[_BZ_FT_]  :%d\n", move_wards[_BZ_FT_]);
    printf("move_wards[_FX_FY_]  :%d\n", move_wards[_FX_FY_]);
    printf("move_wards[_FX_FZ_]  :%d\n", move_wards[_FX_FZ_]);
    printf("move_wards[_FX_FT_]  :%d\n", move_wards[_FX_FT_]);
    printf("move_wards[_FY_FZ_]  :%d\n", move_wards[_FY_FZ_]);
    printf("move_wards[_FY_FT_]  :%d\n", move_wards[_FY_FT_]);
    printf("move_wards[_FZ_FT_]  :%d\n", move_wards[_FZ_FT_]);
  }
};
#endif