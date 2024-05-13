#pragma optimize(5)
#include "../../include/qcu.h"
#ifdef MPI_WILSON_DSLASH
void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                  QcuParam *param, int parity, QcuParam *grid) {
  // define for mpi_wilson_dslash
  int lat_1dim[DIM];
  int lat_3dim[DIM];
  int lat_4dim;
  give_dims(param, lat_1dim, lat_3dim, lat_4dim);
  int lat_3dim6[DIM];
  int lat_3dim12[DIM];
  for (int i = 0; i < DIM; i++) {
    lat_3dim6[i] = lat_3dim[i] * 6;
    lat_3dim12[i] = lat_3dim6[i] * 2;
  }
  cudaError_t err;
  dim3 gridDim(lat_4dim / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  int node_rank;
  int move[BF];
  int grid_1dim[DIM];
  int grid_index_1dim[DIM];
  give_grid(grid, node_rank, grid_1dim, grid_index_1dim);
  MPI_Request send_request[WARDS];
  MPI_Request recv_request[WARDS];
  void *host_send_vec[WARDS];
  void *host_recv_vec[WARDS];
  void *device_send_vec[WARDS];
  void *device_recv_vec[WARDS];
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