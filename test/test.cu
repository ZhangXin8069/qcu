#pragma optimize(5)
#include "../include/qcu.h"

// __global__ void test(LatticeParam param, void *void_a) {
//   // LatticePoint point(param, blockIdx.x, blockDim.x, threadIdx.x);
//   // Complex *a = (Complex*)void_a;
//   // for (int i = 0; i < LAT_C * LAT_S; ++i) {
//   //   a[point.get_index() * LAT_C * LAT_S + i] = i;
//   // }
//   // a[0]=2;
//   __syncthreads();
// }

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  LatticeParam param(32, 32, 32, 64, 2, 1, 1, 4, ODD);
  LatticePoint point;
  LatticePoint b_point;
  LatticePoint f_point;
  LatticePoint ff_point;
  cudaError_t err;
  dim3 gridDim(param.grid_size() / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  ComplexVector a(param.lat_size()*LAT_C*LAT_S);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  a = 1;
  if (rank == 0) {
    std::cout << "#rank:" << rank << "#a:" << a.to_string() << std::endl;
  }
  auto start = std::chrono::high_resolution_clock::now();
  LatticePoint point(param, 50, BLOCK_SIZE, 10);
  // void *void_a = (void*)a._data;
  // test<<<gridDim, blockDim>>>(param , void_a);//???
  // auto end = std::chrono::high_resolution_clock::now();
  err = cudaGetLastError();
  checkCudaErrors(err);
  checkCudaErrors(cudaDeviceSynchronize());
  // // auto end = std::chrono::high_resolution_clock::now();
  if (rank == 0) {
    std::cout << "#rank:" << rank << "#a:" << a.to_string() << std::endl;
    std::cout << "#rank:" << rank << "#point:" << point.to_string() << std::endl;
  }
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("test total time : "
         "%.9lf sec\n",
         double(duration) / 1e9);
  MPI_Finalize();
  return 0;
}
