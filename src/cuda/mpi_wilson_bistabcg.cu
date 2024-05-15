#include <iostream>
#pragma optimize(5)
#include "../../include/qcu.h"
#ifdef MPI_WILSON_BISTABCG
// #define DEBUG_MPI_WILSON_CG
__global__ void generateRandomNumbers(void *device_randomNumbers, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double *randomNumbers =static_cast<double *>(device_randomNumbers);
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        randomNumbers[idx] = curand_uniform(&state);
    }
}

__global__ void generateCustomNumbers(void *device_customNumbers, int n, double real, double imag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex *customNumbers =static_cast<LatticeComplex *>(device_customNumbers);
    if (idx < n) {
        customNumbers[idx].real = real;
        customNumbers[idx].imag = imag;
    }
}

void mpiBistabCgQcu(void *gauge, QcuParam *param, QcuParam *grid) {
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
  malloc_vec(lat_3dim6, device_send_vec, device_recv_vec, host_send_vec,
             host_recv_vec);
  // define end
  // define for mpi_wilson_cg
  int lat_4dim12 = lat_4dim * 12;
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex one(1.0, 0.0);
  LatticeComplex r_norm2(0.0, 0.0);
  const int MAX_ITER(1e3); // 300++?
  const double TOL(1e-6);
  LatticeComplex rho_prev(1.0, 0.0);
  LatticeComplex rho(0.0, 0.0);
  LatticeComplex alpha(1.0, 0.0);
  LatticeComplex omega(1.0, 0.0);
  LatticeComplex beta(0.0, 0.0);
  double kappa = 0.125;
  LatticeComplex tmp(0.0, 0.0);
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex local_result(0.0, 0.0);
  void *ans_e, *ans_o, *x_e, *x_o, *b_e, *b_o, *b__o, *r, *r_tilde,
      *p, *v, *s, *t, *device_latt_tmp0, *device_latt_tmp1;
  cudaMalloc(&ans_e, lat_4dim12 * sizeof(LatticeComplex));
  cudaMalloc(&ans_o, lat_4dim12 * sizeof(LatticeComplex));
  cudaMalloc(&x_e, lat_4dim12 * sizeof(LatticeComplex));
  cudaMalloc(&x_o, lat_4dim12 * sizeof(LatticeComplex));
  cudaMalloc(&b_e, lat_4dim12 * sizeof(LatticeComplex));
  cudaMalloc(&b_o, lat_4dim12 * sizeof(LatticeComplex));
  cudaMalloc(&b__o, lat_4dim12 * sizeof(LatticeComplex));
  cudaMalloc(&r, lat_4dim12 * sizeof(LatticeComplex));
  cudaMalloc(&r_tilde, lat_4dim12 * sizeof(LatticeComplex));
  cudaMalloc(&p, lat_4dim12 * sizeof(LatticeComplex));
  cudaMalloc(&v, lat_4dim12 * sizeof(LatticeComplex));
  cudaMalloc(&s, lat_4dim12 * sizeof(LatticeComplex));
  cudaMalloc(&t, lat_4dim12 * sizeof(LatticeComplex));
  cudaMalloc(&device_latt_tmp0, lat_4dim12 * sizeof(LatticeComplex));
  cudaMalloc(&device_latt_tmp1, lat_4dim12 * sizeof(LatticeComplex));
  LatticeComplex *host_latt_tmp0 = (LatticeComplex *)malloc(lat_4dim12 * sizeof(LatticeComplex));
  LatticeComplex *host_latt_tmp1 = (LatticeComplex *)malloc(lat_4dim12 * sizeof(LatticeComplex));
  checkCudaErrors(cudaDeviceSynchronize());
  auto start = std::chrono::high_resolution_clock::now();
  int lat_4dim24=lat_4dim12*2;
  generateRandomNumbers<<<gridDim, blockDim>>>(device_latt_tmp0, lat_4dim24, node_rank);
  checkCudaErrors(cudaDeviceSynchronize());
  cudaMemcpy(host_latt_tmp0, device_latt_tmp0,sizeof(LatticeComplex)*lat_4dim12, cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  err = cudaGetLastError();
  checkCudaErrors(err);
  printf("mpi wilson bistabcg total time: (without malloc free "
         "memcpy) :%.9lf "
         "sec\n",
         double(duration) / 1e9);
  printf("[0]:%f,%f;[-1]::%f,%f;\n", host_latt_tmp0[0].real, host_latt_tmp0[0].imag,host_latt_tmp0[lat_4dim12-1].real,host_latt_tmp0[lat_4dim12-1].imag);
  // free
  free_vec(device_send_vec, device_recv_vec, host_send_vec, host_recv_vec);
  cudaFree(x_o);
  cudaFree(b__o);
  cudaFree(r);
  cudaFree(r_tilde);
  cudaFree(p);
  cudaFree(v);
  cudaFree(s);
  cudaFree(t);
}
#endif