#include "../../include/qcu.h"
#ifdef MPI_WILSON_DSLASH
void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                  QcuParam *param, int parity, QcuParam *grid) {
  // define for mpi_wilson_dslash
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
  MPI_Request send_request[_WARDS_];
  MPI_Request recv_request[_WARDS_];
  void *host_send_vec[_WARDS_];
  void *host_recv_vec[_WARDS_];
  void *device_send_vec[_WARDS_];
  void *device_recv_vec[_WARDS_];
  malloc_vec(lat_3dim6, device_send_vec, device_recv_vec, host_send_vec, host_recv_vec);
  // define end
  checkCudaErrors(cudaDeviceSynchronize());
  auto start = std::chrono::high_resolution_clock::now();
  // mpi wilson dslash
  _mpiDslashQcu(gridDim, blockDim, gauge, fermion_in, fermion_out, parity,
                lat_1dim, lat_3dim12, node_rank, grid_1dim, grid_index_1dim,
                move, send_request, recv_request, device_send_vec, device_recv_vec, host_send_vec, host_recv_vec);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  err = cudaGetLastError();
  checkCudaErrors(err);
  printf("mpi wilson dslash total time: (without malloc free memcpy) :%.9lf "
         "sec\n",
         double(duration) / 1e9);
  // free
  free_vec(device_send_vec, device_recv_vec, host_send_vec, host_recv_vec);
}
#endif