#pragma optimize(5)
#include "../include/qcu.h"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  LatticeParam param(32, 32, 32, 64 , 2, 1, 1, 4, EVEN);
  cudaError_t err;
  dim3 gridDim(param.grid_size()/BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  ComplexVector a(param.lat_size());
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  a.init_random(rank);
  auto start = std::chrono::high_resolution_clock::now();
  a=a*a*a*a+a+a+a+a+a+a+a+a+a+a+a+a;
  auto end = std::chrono::high_resolution_clock::now();
  err = cudaGetLastError();
  checkCudaErrors(err);
  checkCudaErrors(cudaDeviceSynchronize());
  // auto end = std::chrono::high_resolution_clock::now();
  std::cout <<"#rank:"<< rank<<"#a:"<< a.to_string() << std::endl;
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("test total time : "
         "%.9lf sec\n",
         double(duration) / 1e9);
  MPI_Finalize();
  return 0;
}