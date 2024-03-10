#include <iostream>
#pragma optimize(5)
#include "../../include/qcu.h"
#ifdef MPI_WILSON_CG
#define DEBUG_MPI_WILSON_CG
#define TEST_MPI_WILSON_CG 0
#define TEST_MPI_WILSON_CG_USE_WILSON_DSLASH 0

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
  void *send_vec[WARDS];
  void *recv_vec[WARDS];
  malloc_recv(lat_3dim6, send_vec, recv_vec);
  // define end
  // define for mpi_wilson_cg
  int lat_4dim12 = lat_4dim * 12;
  LatticeComplex *dslash_in, *dslash_out, *x, *b, *r, *r_tilde, *p, *v, *s, *t;
  cudaMallocManaged(&x, lat_4dim12 * sizeof(LatticeComplex));
  cudaMallocManaged(&b, lat_4dim12 * sizeof(LatticeComplex));
  cudaMallocManaged(&r, lat_4dim12 * sizeof(LatticeComplex));
  cudaMallocManaged(&r_tilde, lat_4dim12 * sizeof(LatticeComplex));
  cudaMallocManaged(&p, lat_4dim12 * sizeof(LatticeComplex));
  cudaMallocManaged(&v, lat_4dim12 * sizeof(LatticeComplex));
  cudaMallocManaged(&s, lat_4dim12 * sizeof(LatticeComplex));
  cudaMallocManaged(&t, lat_4dim12 * sizeof(LatticeComplex));
  LatticeComplex r_norm2(0.0, 0.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex one(1.0, 0.0);
  const int MAX_ITER(1e2); // 300++?
  const double TOL(1e-6);
  LatticeComplex rho_prev(1.0, 0.0);
  LatticeComplex rho(0.0, 0.0);
  LatticeComplex alpha(1.0, 0.0);
  LatticeComplex omega(1.0, 0.0);
  LatticeComplex beta(0.0, 0.0);
  LatticeComplex tmp(0.0, 0.0);
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex local_result(0.0, 0.0);
  // double Kappa = 0.125;
  double Kappa = 10;
  auto start = std::chrono::high_resolution_clock::now();
  give_rand(x, lat_4dim12); // rand x
  // give_value(x, zero, lat_4dim12 );    // zero x
  // give_rand(b, lat_4dim12 );           // rand b
  give_value(b, one, 1);                 // point b
  give_value(r, zero, lat_4dim12);       // zero r
  give_value(r_tilde, zero, lat_4dim12); // zero r_tilde
  give_value(p, zero, lat_4dim12);       // zero p
  give_value(v, zero, lat_4dim12);       // zero v
  give_value(s, zero, lat_4dim12);       // zero s
  give_value(t, zero, lat_4dim12);       // zero t
  dslash_in = x;
  dslash_out = r;
  // define end
  _dslash(gridDim, blockDim, gauge, fermion_in, fermion_out, parity, lat_1dim,
          lat_3dim12, node_rank, grid_1dim, grid_index_1dim, move, send_request,
          recv_request, send_vec, recv_vec, dslash_in, dslash_out, Kappa, zero,
          one);
  for (int i = 0; i < lat_4dim12; i++) {
    r[i] = b[i] - r[i];
    r_tilde[i] = r[i];
  }
  for (int loop = 0; loop < MAX_ITER; loop++) {
    cg_mpi_dot(local_result, lat_4dim12, r_tilde, r, rho, zero);
#ifdef DEBUG_MPI_WILSON_CG
    std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
              << "##rho:" << rho.real << std::endl;
#endif
    beta = (rho / rho_prev) * (alpha / omega);
#ifdef DEBUG_MPI_WILSON_CG
    std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
              << "##beta:" << beta.real << std::endl;
#endif
    for (int i = 0; i < lat_4dim12; i++) {
      p[i] = r[i] + (p[i] - v[i] * omega) * beta;
    }
    // v = A * p;
    dslash_in = p;
    dslash_out = v;
    _dslash(gridDim, blockDim, gauge, fermion_in, fermion_out, parity, lat_1dim,
            lat_3dim12, node_rank, grid_1dim, grid_index_1dim, move,
            send_request, recv_request, send_vec, recv_vec, dslash_in,
            dslash_out, Kappa, zero, one);
    cg_mpi_dot(local_result, lat_4dim12, r_tilde, v, tmp, zero);
    alpha = rho / tmp;
#ifdef DEBUG_MPI_WILSON_CG
    std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
              << "##alpha:" << alpha.real << std::endl;
#endif
    for (int i = 0; i < lat_4dim12; i++) {
      s[i] = r[i] - v[i] * alpha;
    }
    // t = A * s;
    dslash_in = s;
    dslash_out = t;
    _dslash(gridDim, blockDim, gauge, fermion_in, fermion_out, parity, lat_1dim,
            lat_3dim12, node_rank, grid_1dim, grid_index_1dim, move,
            send_request, recv_request, send_vec, recv_vec, dslash_in,
            dslash_out, Kappa, zero, one);
    cg_mpi_dot(local_result, lat_4dim12, t, s, tmp0, zero);
    cg_mpi_dot(local_result, lat_4dim12, t, t, tmp1, zero);
    omega = tmp0 / tmp1;
#ifdef DEBUG_MPI_WILSON_CG
    std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
              << "##omega:" << omega.real << std::endl;
#endif
    for (int i = 0; i < lat_4dim12; i++) {
      x[i] = x[i] + p[i] * alpha + s[i] * omega;
    }
    for (int i = 0; i < lat_4dim12; i++) {
      r[i] = s[i] - t[i] * omega;
    }
    cg_mpi_dot(local_result, lat_4dim12, r, r, r_norm2, zero);
    std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
              << "##Residual:" << r_norm2.real << std::endl;
    // break;
    if (r_norm2.real < TOL || loop == MAX_ITER - 1) {
      break;
    }
    rho_prev = rho;
  }
  checkCudaErrors(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  err = cudaGetLastError();
  checkCudaErrors(err);
  printf("mpi wilson cg total time: (without malloc free "
         "memcpy) :%.9lf "
         "sec\n",
         double(duration) / 1e9);
  // free
  free_recv(send_vec, recv_vec);
  cudaFree(x);
  cudaFree(b);
  cudaFree(r);
  cudaFree(r_tilde);
  cudaFree(p);
  cudaFree(v);
  cudaFree(s);
  cudaFree(t);
}

#endif