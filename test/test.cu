#pragma optimize(5)
#include "../include/qcu.h"

__global__ void test(LatticePoint point) {
  point.init_3int(blockIdx.x , blockDim.x , threadIdx.x);
  point.move(X,FORWARD);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  LatticeParam param(32, 32, 32, 64 , 2, 1, 1, 4, ODD);
  LatticePoint point;
  LatticePoint b_point;
  LatticePoint f_point;
  LatticePoint ff_point;
  point.init_param(param);
  cudaError_t err;
  dim3 gridDim(param.grid_size()/BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  ComplexVector a(param.lat_size());
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  a.init_random(rank);
  auto start = std::chrono::high_resolution_clock::now();
  // a=a*a*a*a+a+a+a+a+a+a+a+a+a+a+a+a;
  // test<<<gridDim, blockDim>>>(point);
  point.init_3int(50, 128 ,5);
  std::cout <<"#rank:"<< rank<<"#point:"<< point.to_string() << std::endl;
  b_point = point.move(X,BACKWARD);
  std::cout <<"#rank:"<< rank<<"#b_point:"<< b_point.to_string() << std::endl;
  f_point = point.move(X,FORWARD);
  std::cout <<"#rank:"<< rank<<"#f_point:"<< f_point.to_string() << std::endl;
  ff_point = point.move(X,FORWARD);
  std::cout <<"#rank:"<< rank<<"#ff_point:"<< ff_point.to_string() << std::endl;
  auto end = std::chrono::high_resolution_clock::now();
  err = cudaGetLastError();
  checkCudaErrors(err);
  checkCudaErrors(cudaDeviceSynchronize());
  // auto end = std::chrono::high_resolution_clock::now();
  // std::cout <<"#rank:"<< rank<<"#a:"<< a.to_string() << std::endl;
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("test total time : "
         "%.9lf sec\n",
         double(duration) / 1e9);
  MPI_Finalize();
  return 0;
}