#include "../../include/qcu.h"
#include "../../include/qcu_cuda.h"
#include <chrono>

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge,
               QcuParam *param, int parity) {
  int lat_x = param->lattice_size[0] >> 1;
  int lat_y = param->lattice_size[1];
  int lat_z = param->lattice_size[2];
  int lat_t = param->lattice_size[3];
  void *clover;
  checkCudaErrors(cudaMalloc(&clover, (lat_t * lat_z * lat_y * lat_x * 144) *
                                          sizeof(LatticeComplex)));
  cudaError_t err;
  dim3 gridDim(lat_x * lat_y * lat_z * lat_t / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  {
    // wilson dslash
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    wilson_dslash<<<gridDim, blockDim>>>(gauge, fermion_in, fermion_out, lat_x,
                                         lat_y, lat_z, lat_t, parity);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf(
        "wilson dslash total time: (without malloc free memcpy) : %.9lf sec\n",
        double(duration) / 1e9);
  }
  {
    // make clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    make_clover<<<gridDim, blockDim>>>(gauge, clover, lat_x, lat_y, lat_z,
                                       lat_t, parity);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("make clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  {
    // inverse clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    inverse_clover<<<gridDim, blockDim>>>(clover, lat_x, lat_y, lat_z);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf(
        "inverse clover total time: (without malloc free memcpy) :%.9lf sec\n ",
        double(duration) / 1e9);
  }
  {
    // give clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    give_clover<<<gridDim, blockDim>>>(clover, fermion_out, lat_x, lat_y,
                                       lat_z);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("give clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  {
    // free
    checkCudaErrors(cudaFree(clover));
  }
}

void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
               QcuParam *param, int parity, QcuParam *grid) {
  int node_size, node_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &node_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &node_rank);
  int lat_x = param->lattice_size[0] >> 1;
  int lat_y = param->lattice_size[1];
  int lat_z = param->lattice_size[2];
  int lat_t = param->lattice_size[3];
  int grid_x = grid->lattice_size[0];
  int grid_y = grid->lattice_size[1];
  int grid_z = grid->lattice_size[2];
  int grid_t = grid->lattice_size[3];
  //void *clover;
  //checkCudaErrors(cudaMalloc(&clover, (lat_t * lat_z * lat_y * lat_x * 144) *
  //                                       sizeof(LatticeComplex)));
  cudaError_t err;
  dim3 gridDim(lat_x * lat_y * lat_z * lat_t / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  {
    printf(
        "grid:x-%d, y-%d, z-%d, t-%d \n", grid_x, grid_y, grid_z, grid_t);
  }
  {
    //mpi wilson dslash
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    mpi_wilson_dslash<<<gridDim, blockDim>>>(gauge, fermion_in, fermion_out, lat_x,
                                         lat_y, lat_z, lat_t, parity, grid_x,
                                         grid_y, grid_z, grid_t);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf(
        "mpi wilson dslash total time: (without malloc free memcpy) : %.9lf sec\n",
        double(duration) / 1e9);
  }
  //{
  //  // make clover
  //  checkCudaErrors(cudaDeviceSynchronize());
  //  auto start = std::chrono::high_resolution_clock::now();
  //  make_clover<<<gridDim, blockDim>>>(gauge, clover, lat_x, lat_y, lat_z,
  //                                     lat_t, parity);
  //  err = cudaGetLastError();
  //  checkCudaErrors(err);
  //  checkCudaErrors(cudaDeviceSynchronize());
  //  auto end = std::chrono::high_resolution_clock::now();
  //  auto duration =
  //      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
  //          .count();
  //  printf("make clover total time: (without malloc free memcpy) :%.9lf sec\n ",
  //         double(duration) / 1e9);
  //}
  //{
  //  // inverse clover
  //  checkCudaErrors(cudaDeviceSynchronize());
  //  auto start = std::chrono::high_resolution_clock::now();
  //  inverse_clover<<<gridDim, blockDim>>>(clover, lat_x, lat_y, lat_z);
  //  err = cudaGetLastError();
  //  checkCudaErrors(err);
  //  checkCudaErrors(cudaDeviceSynchronize());
  //  auto end = std::chrono::high_resolution_clock::now();
  //  auto duration =
  //      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
  //          .count();
  //  printf(
  //      "inverse clover total time: (without malloc free memcpy) :%.9lf sec\n ",
  //      double(duration) / 1e9);
  //}
  //{
  //  // give clover
  //  checkCudaErrors(cudaDeviceSynchronize());
  //  auto start = std::chrono::high_resolution_clock::now();
  //  give_clover<<<gridDim, blockDim>>>(clover, fermion_out, lat_x, lat_y,
  //                                     lat_z);
  //  err = cudaGetLastError();
  //  checkCudaErrors(err);
  //  checkCudaErrors(cudaDeviceSynchronize());
  //  auto end = std::chrono::high_resolution_clock::now();
  //  auto duration =
  //      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
  //          .count();
  //  printf("give clover total time: (without malloc free memcpy) :%.9lf sec\n ",
  //         double(duration) / 1e9);
  //}
  //{
  //  // free
  //  checkCudaErrors(cudaFree(clover));
  //}
}

