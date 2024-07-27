#include "../../include/qcu.h"
#ifdef NCCL_WILSON_DSLASH
void ncclDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                   QcuParam *param, int parity, QcuParam *grid) {
  // define for nccl_wilson_dslash
  int lat_1dim[_DIM_];
  int lat_3dim[_DIM_];
  int lat_4dim;
  give_dims(param, lat_1dim, lat_3dim, lat_4dim);
  int lat_3dim6[_DIM_];
  int lat_3dim12[_DIM_];
  for (int i = 0; i < _DIM_; i++) {
    lat_3dim6[i] = lat_3dim[i] * 6;
    lat_3dim12[i] = lat_3dim6[i] * 2;
  }
  cudaError_t err;
  dim3 gridDim(lat_4dim / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  int node_rank;
  int move[_BF_];
  int grid_1dim[_DIM_];
  int grid_index_1dim[_DIM_];
  give_grid(grid, node_rank, grid_1dim, grid_index_1dim);
  void *host_send_vec[_WARDS_];
  void *host_recv_vec[_WARDS_];
  void *device_send_vec[_WARDS_];
  void *device_recv_vec[_WARDS_];
  malloc_vec(lat_3dim6, device_send_vec, device_recv_vec, host_send_vec,
             host_recv_vec);
  // define end
  // initializing MPI
  int node_size, localRank = 0;
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &node_size));
  // calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[node_size];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[node_rank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
                         sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p = 0; p < node_size; p++) {
    if (p == node_rank)
      break;
    if (hostHashs[p] == hostHashs[node_rank])
      localRank++;
  }
  ncclUniqueId nccl_id;
  ncclComm_t nccl_comm;
  cudaStream_t stream;
  // get NCCL unique nccl_id at rank 0 and broadcast it to all others
  if (node_rank == 0)
    ncclGetUniqueId(&nccl_id);
  MPICHECK(MPI_Bcast((void *)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0,
                     MPI_COMM_WORLD));
  // picking a GPU based on localRank
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaStreamCreate(&stream));
  // initializing NCCL
  NCCLCHECK(ncclCommInitRank(&nccl_comm, node_size, nccl_id, node_rank));
  // define end
  checkCudaErrors(cudaDeviceSynchronize());
  auto start = std::chrono::high_resolution_clock::now();
  // nccl wilson dslash
  _ncclDslashQcu(gridDim, blockDim, gauge, fermion_in, fermion_out, parity,
                 lat_1dim, lat_3dim12, node_rank, grid_1dim, grid_index_1dim,
                 move, device_send_vec, device_recv_vec, nccl_comm, stream);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  err = cudaGetLastError();
  checkCudaErrors(err);
  printf("nccl wilson dslash total time: (without malloc free memcpy) :%.9lf "
         "sec\n",
         double(duration) / 1e9);
  // free
  free_vec(device_send_vec, device_recv_vec, host_send_vec, host_recv_vec);
  // finalizing NCCL
  ncclCommDestroy(nccl_comm);
}
#endif