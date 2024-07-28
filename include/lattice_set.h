#ifndef _LATTICE_SET_H
#define _LATTICE_SET_H
#include "./qcu.h"

struct LatticeSet {
  int lat_1dim[_DIM_];
  int lat_3dim[_DIM_];
  int lat_4dim;
  int lat_3dim6[_DIM_];
  int lat_3dim12[_DIM_];
  int lat_4dim12;
  cudaError_t err;
  dim3 gridDim;
  dim3 blockDim;
  ncclUniqueId qcu_nccl_id;
  ncclComm_t qcu_nccl_comm;
  cudaStream_t qcu_stream;
  cudaStream_t qcu_streams[_WARDS_];
  int node_rank, node_size;
  int move[_BF_];
  int grid_1dim[_DIM_];
  int grid_index_1dim[_DIM_];
  void *host_send_vec[_WARDS_];
  void *host_recv_vec[_WARDS_];
  void *device_send_vec[_WARDS_];
  void *device_recv_vec[_WARDS_];

  void _init(int *_param_lat_size, int *_grid_lat_size) {
    lat_1dim[_X_] = _param_lat_size[_X_] >> 1; // even-odd
    lat_1dim[_Y_] = _param_lat_size[_Y_];
    lat_1dim[_Z_] = _param_lat_size[_Z_];
    lat_1dim[_T_] = _param_lat_size[_T_];
    grid_1dim[_X_] = _grid_lat_size[_X_];
    grid_1dim[_Y_] = _grid_lat_size[_Y_];
    grid_1dim[_Z_] = _grid_lat_size[_Z_];
    grid_1dim[_T_] = _grid_lat_size[_T_];
  }
  void _init_example() {
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
    blockDim = _BLOCK_SIZE_;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &node_size);
    if (node_rank == 0) {
      ncclGetUniqueId(&qcu_nccl_id);
    }
    MPI_Bcast((void *)&qcu_nccl_id, sizeof(qcu_nccl_id), MPI_BYTE, 0,
              MPI_COMM_WORLD);
    ncclCommInitRank(&qcu_nccl_comm, node_size, qcu_nccl_id, node_rank);
    cudaStreamCreate(&qcu_stream);
    grid_index_1dim[_X_] =
        node_rank / grid_1dim[_T_] / grid_1dim[_Z_] / grid_1dim[_Y_];
    grid_index_1dim[_Y_] =
        node_rank / grid_1dim[_T_] / grid_1dim[_Z_] % grid_1dim[_Y_];
    grid_index_1dim[_Z_] = node_rank / grid_1dim[_T_] % grid_1dim[_Z_];
    grid_index_1dim[_T_] = node_rank % grid_1dim[_T_];
    lat_3dim[_YZT_] = lat_1dim[_Y_] * lat_1dim[_Z_] * lat_1dim[_T_];
    lat_3dim[_XZT_] = lat_1dim[_X_] * lat_1dim[_Z_] * lat_1dim[_T_];
    lat_3dim[_XYT_] = lat_1dim[_X_] * lat_1dim[_Y_] * lat_1dim[_T_];
    lat_3dim[_XYZ_] = lat_1dim[_X_] * lat_1dim[_Y_] * lat_1dim[_Z_];
    lat_4dim = lat_3dim[_XYZ_] * lat_1dim[_T_];
    for (int i = 0; i < _DIM_; i++) {
      cudaStreamCreate(&qcu_streams[i]);
      lat_3dim6[i] = lat_3dim[i] * 6;
      lat_3dim12[i] = lat_3dim6[i] * 2;
      cudaMalloc(&device_send_vec[i * _SR_],
                 lat_3dim6[i] * sizeof(LatticeComplex));
      cudaMalloc(&device_send_vec[i * _SR_ + 1],
                 lat_3dim6[i] * sizeof(LatticeComplex));
      cudaMalloc(&device_recv_vec[i * _SR_],
                 lat_3dim6[i] * sizeof(LatticeComplex));
      cudaMalloc(&device_recv_vec[i * _SR_ + 1],
                 lat_3dim6[i] * sizeof(LatticeComplex));
      host_send_vec[i * _SR_] =
          (void *)malloc(lat_3dim6[i] * sizeof(LatticeComplex));
      host_send_vec[i * _SR_ + 1] =
          (void *)malloc(lat_3dim6[i] * sizeof(LatticeComplex));
      host_recv_vec[i * _SR_] =
          (void *)malloc(lat_3dim6[i] * sizeof(LatticeComplex));
      host_recv_vec[i * _SR_ + 1] =
          (void *)malloc(lat_3dim6[i] * sizeof(LatticeComplex));
    }
    lat_4dim12 = lat_4dim * 12;
    gridDim = lat_4dim / _BLOCK_SIZE_;
  }
  void end() {
    cudaStreamDestroy(qcu_stream);
    ncclCommDestroy(qcu_nccl_comm);
    for (int i = 0; i < _WARDS_; i++) {
      cudaFree(device_send_vec[i]);
      cudaFree(device_recv_vec[i]);
      free(host_send_vec[i]);
      free(host_recv_vec[i]);
      cudaStreamDestroy(qcu_streams[i]);
    }
  }
  void _print() {
    printf("lat_1dim[_X_] :%d\n", lat_1dim[_X_]);
    printf("lat_1dim[_Y_] :%d\n", lat_1dim[_Y_]);
    printf("lat_1dim[_Z_] :%d\n", lat_1dim[_Z_]);
    printf("lat_1dim[_T_] :%d\n", lat_1dim[_T_]);
    printf("grid_1dim[_X_]:%d\n", grid_1dim[_X_]);
    printf("grid_1dim[_Y_]:%d\n", grid_1dim[_Y_]);
    printf("grid_1dim[_Z_]:%d\n", grid_1dim[_Z_]);
    printf("grid_1dim[_T_]:%d\n", grid_1dim[_T_]);
  }
};

#endif