
#pragma optimize(5)
#include "../include/qcu.h"
// #define DEBUG_MPI_WILSON_CG
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  // define for mpi_wilson_dslash
  int lat_1dim[DIM];
  int lat_3dim[DIM];
  int lat_4dim;
  lat_1dim[X] = LAT_EXAMPLE >> 1;
  lat_1dim[Y] = LAT_EXAMPLE;
  lat_1dim[Z] = LAT_EXAMPLE;
  lat_1dim[T] = LAT_EXAMPLE;
  lat_3dim[YZT] = lat_1dim[Y] * lat_1dim[Z] * lat_1dim[T];
  lat_3dim[XZT] = lat_1dim[X] * lat_1dim[Z] * lat_1dim[T];
  lat_3dim[XYT] = lat_1dim[X] * lat_1dim[Y] * lat_1dim[T];
  lat_3dim[XYZ] = lat_1dim[X] * lat_1dim[Y] * lat_1dim[Z];
  lat_4dim = lat_3dim[XYZ] * lat_1dim[T];
  int lat_3dim6[DIM];
  int lat_3dim12[DIM];
  for (int i = 0; i < DIM; i++) {
    lat_3dim6[i] = lat_3dim[i] * 6;
    lat_3dim12[i] = lat_3dim6[i] * 2;
  }
  cudaError_t err;
  dim3 gridDim(lat_4dim / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  int node_rank, node_size;
  int move[BF];
  int grid_1dim[DIM];
  int grid_index_1dim[DIM];
  MPI_Comm_rank(MPI_COMM_WORLD, &node_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &node_size);
  grid_1dim[X] = node_size;
  grid_1dim[Y] = GRID_EXAMPLE;
  grid_1dim[Z] = GRID_EXAMPLE;
  grid_1dim[T] = GRID_EXAMPLE;
  grid_index_1dim[X] = node_rank / grid_1dim[T] / grid_1dim[Z] / grid_1dim[Y];
  grid_index_1dim[Y] = node_rank / grid_1dim[T] / grid_1dim[Z] % grid_1dim[Y];
  grid_index_1dim[Z] = node_rank / grid_1dim[T] % grid_1dim[Z];
  grid_index_1dim[T] = node_rank % grid_1dim[T];
  MPI_Request send_request[WARDS];
  MPI_Request recv_request[WARDS];
  void *send_vec[WARDS];
  void *recv_vec[WARDS];
  malloc_vec(lat_3dim6, send_vec, recv_vec);
  // define end
  // define gauge and fermion
  LatticeComplex *gauge, *fermion_in, *fermion_out;
  cudaMallocManaged(&gauge, lat_4dim * LAT_D * LAT_C * LAT_C * EVENODD *
                                sizeof(LatticeComplex));
  cudaMallocManaged(&fermion_in, lat_4dim * LAT_S * LAT_C * EVENODD *
                                     sizeof(LatticeComplex));
  cudaMallocManaged(&fermion_out, lat_4dim * LAT_S * LAT_C * EVENODD *
                                      sizeof(LatticeComplex));
  give_rand(gauge, lat_4dim * LAT_D * LAT_C * LAT_C * EVENODD);
  give_rand(fermion_in, lat_4dim * LAT_S * LAT_C * EVENODD);
  // define end
  // define for mpi_wilson_dslash
  checkCudaErrors(cudaDeviceSynchronize());
  auto start = std::chrono::high_resolution_clock::now();
  // mpi wilson dslash
  printf("%d-DEBUG\n", node_rank);
  _mpiDslashQcu(gridDim, blockDim, gauge, fermion_in, fermion_out, EVEN,
                lat_1dim, lat_3dim12, node_rank, grid_1dim, grid_index_1dim,
                move, send_request, recv_request, send_vec, recv_vec);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  err = cudaGetLastError();
  checkCudaErrors(err);
  printf("mpi wilson dslash total time: (without malloc free memcpy) :%.9lf "
         "sec\n",
         double(duration) / 1e9);
  // free
  free_vec(send_vec, recv_vec);
  cudaFree(gauge);
  cudaFree(fermion_in);
  cudaFree(fermion_out);
  MPI_Finalize();
  return 0;
}
