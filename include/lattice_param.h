#ifndef _LATTICE_SET_H
#define _LATTICE_SET_H
#include "./qcu.h"

struct LatticeParam {
  LatticeParam(LATTICE_TEMPLATE param){}
};

struct LatticeSet {
  // LatticeSet(QcuParam *param, QcuParam *grid){}
  //     : real(real), imag(imag) {}
  // double real;
  // double imag;

  //   int lat_1dim[_DIM_];
  // int lat_3dim[_DIM_];
  // int lat_4dim;
  // give_dims(param, lat_1dim, lat_3dim, lat_4dim);
  // int lat_3dim6[_DIM_];
  // int lat_3dim12[_DIM_];
  // for (int i = 0; i < _DIM_; i++) {
  //   lat_3dim6[i] = lat_3dim[i] * 6;
  //   lat_3dim12[i] = lat_3dim6[i] * 2;
  // }
  // int lat_4dim12 = lat_4dim * 12;
  // cudaError_t err;
  // dim3 gridDim(lat_4dim / BLOCK_SIZE);
  // dim3 blockDim(BLOCK_SIZE);
  // ncclUniqueId qcu_nccl_id;
  // ncclComm_t qcu_nccl_comm;
  // cudaStream_t qcu_stream;
  // int node_rank, node_size;
  // int move[_BF_];
  // int grid_1dim[_DIM_];
  // int grid_index_1dim[_DIM_];
  // give_grid(grid, node_rank, grid_1dim, grid_index_1dim);
  // void *host_send_vec[_WARDS_];
  // void *host_recv_vec[_WARDS_];
  // void *device_send_vec[_WARDS_];
  // void *device_recv_vec[_WARDS_];
  // malloc_vec(lat_3dim6, device_send_vec, device_recv_vec, host_send_vec,
  //            host_recv_vec);
  // // define end
  // // initializing MPI
  // MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &node_rank));
  // MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &node_size));
  // // initializing NCCL
  // if (node_rank == 0) {
  //   ncclGetUniqueId(&qcu_nccl_id);
  // }
  // MPICHECK(MPI_Bcast((void *)&qcu_nccl_id, sizeof(qcu_nccl_id), MPI_BYTE, 0,
  //                    MPI_COMM_WORLD));
  // NCCLCHECK(
  //     ncclCommInitRank(&qcu_nccl_comm, node_size, qcu_nccl_id, node_rank));
  // CUDACHECK(cudaStreamCreate(&qcu_stream));
};

#endif