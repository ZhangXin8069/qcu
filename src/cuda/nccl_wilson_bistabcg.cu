#include <iostream>
#pragma optimize(5)
#include "../../include/qcu.h"
#ifdef NCCL_WILSON_BISTABCG
// #define DEBUG_NCCL_WILSON_CG
void ncclBistabCgQcu(void *gauge, QcuParam *param, QcuParam *grid) {
  int node_rank, node_size, localRank = 0;
  // initializing MPI
  // MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &node_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &node_size));
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
  // nccl wilson bistabcg
  {
    // define for nccl_wilson_dslash
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
    int move[BF];
    int grid_1dim[DIM];
    int grid_index_1dim[DIM];
    give_grid(grid, node_rank, grid_1dim, grid_index_1dim);
    void *send_vec[WARDS];
    void *recv_vec[WARDS];
    malloc_vec(lat_3dim6, send_vec, recv_vec);
    // define end
    // define for nccl_wilson_bistabcg
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
    LatticeComplex *tmp;
    cudaMallocManaged(&tmp, sizeof(LatticeComplex));
    LatticeComplex *tmp0;
    cudaMallocManaged(&tmp0, sizeof(LatticeComplex));
    LatticeComplex *tmp1;
    cudaMallocManaged(&tmp1, sizeof(LatticeComplex));
    LatticeComplex *local_result;
    cudaMallocManaged(&local_result, sizeof(LatticeComplex));
    LatticeComplex *ans_e, *ans_o, *x_e, *x_o, *b_e, *b_o, *b__o, *r, *r_tilde,
        *p, *v, *s, *t, *latt_tmp0, *latt_tmp1;
    cudaMallocManaged(&ans_e, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&ans_o, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&x_e, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&x_o, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_e, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_o, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&b__o, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&r, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&r_tilde, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&p, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&v, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&s, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&t, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&latt_tmp0, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&latt_tmp1, lat_4dim12 * sizeof(LatticeComplex));
    // give ans first
    give_rand(ans_e, lat_4dim12);
    give_rand(ans_o, lat_4dim12);
    // give x_o, b_e, b_o ,b__o, r, r_tilde, p, v, s, t, latt_tmp0, latt_tmp1
    give_rand(x_o, lat_4dim12);
    // give_value(x_o, zero, lat_4dim12 );
    give_value(b_e, zero, lat_4dim12);
    give_value(b_o, zero, lat_4dim12);
    give_value(b__o, zero, lat_4dim12);
    give_value(r, zero, lat_4dim12);
    give_value(r_tilde, zero, lat_4dim12);
    give_value(p, zero, lat_4dim12);
    give_value(v, zero, lat_4dim12);
    give_value(s, zero, lat_4dim12);
    give_value(t, zero, lat_4dim12);
    // give b'_o(b__0)
    give_value(latt_tmp0, zero, lat_4dim12);
    nccl_dslash_eo(latt_tmp0, ans_o, node_rank, gridDim, blockDim, gauge,
                   lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move,
                   send_vec, recv_vec, zero, nccl_comm, stream);
    for (int i = 0; i < lat_4dim12; i++) {
      b_e[i] = ans_e[i] - latt_tmp0[i] * kappa; // b_e=anw_e-kappa*D_eo(ans_o)
    }
    give_value(latt_tmp1, zero, lat_4dim12);
    nccl_dslash_oe(latt_tmp1, ans_e, node_rank, gridDim, blockDim, gauge,
                   lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move,
                   send_vec, recv_vec, zero, nccl_comm, stream);
    for (int i = 0; i < lat_4dim12; i++) {
      b_o[i] = ans_o[i] - latt_tmp1[i] * kappa; // b_o=anw_o-kappa*D_oe(ans_e)
    }
    give_value(latt_tmp0, zero, lat_4dim12);
    nccl_dslash_oe(latt_tmp0, b_e, node_rank, gridDim, blockDim, gauge,
                   lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move,
                   send_vec, recv_vec, zero, nccl_comm, stream);
    for (int i = 0; i < lat_4dim12; i++) {
      b__o[i] = b_o[i] + latt_tmp0[i] * kappa; // b__o=b_o+kappa*D_oe(b_e)
    }
    // bistabcg
    nccl_dslash(r, x_o, kappa, latt_tmp0, latt_tmp1, node_rank, gridDim,
                blockDim, gauge, lat_1dim, lat_3dim12, lat_4dim12, grid_1dim,
                grid_index_1dim, move, send_vec, recv_vec, zero, nccl_comm,
                stream);
    for (int i = 0; i < lat_4dim12; i++) {
      r[i] = b__o[i] - r[i];
      r_tilde[i] = r[i];
    }
    // define end
    auto start = std::chrono::high_resolution_clock::now();
    for (int loop = 0; loop < MAX_ITER; loop++) {
      nccl_dot(local_result, lat_4dim12, r_tilde, r, tmp, zero, nccl_comm,
               stream);
      rho = (*tmp);
#ifdef DEBUG_NCCL_WILSON_CG
      std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
                << "##rho:" << rho.real << std::endl;
#endif
      beta = (rho / rho_prev) * (alpha / omega);
#ifdef DEBUG_NCCL_WILSON_CG
      std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
                << "##beta:" << beta.real << std::endl;
#endif
      for (int i = 0; i < lat_4dim12; i++) {
        p[i] = r[i] + (p[i] - v[i] * omega) * beta;
      }
      // v = A * p;
      nccl_dslash(v, p, kappa, latt_tmp0, latt_tmp1, node_rank, gridDim,
                  blockDim, gauge, lat_1dim, lat_3dim12, lat_4dim12, grid_1dim,
                  grid_index_1dim, move, send_vec, recv_vec, zero, nccl_comm,
                  stream);
      nccl_dot(local_result, lat_4dim12, r_tilde, v, tmp, zero, nccl_comm,
               stream);
      alpha = rho / (*tmp);
#ifdef DEBUG_NCCL_WILSON_CG
      std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
                << "##alpha:" << alpha.real << std::endl;
#endif
      for (int i = 0; i < lat_4dim12; i++) {
        s[i] = r[i] - v[i] * alpha;
      }
      // t = A * s;
      nccl_dslash(t, s, kappa, latt_tmp0, latt_tmp1, node_rank, gridDim,
                  blockDim, gauge, lat_1dim, lat_3dim12, lat_4dim12, grid_1dim,
                  grid_index_1dim, move, send_vec, recv_vec, zero, nccl_comm,
                  stream);
      nccl_dot(local_result, lat_4dim12, t, s, tmp0, zero, nccl_comm, stream);
      nccl_dot(local_result, lat_4dim12, t, t, tmp1, zero, nccl_comm, stream);
      omega = (*tmp0) / (*tmp1);
#ifdef DEBUG_NCCL_WILSON_CG
      std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
                << "##omega:" << omega.real << std::endl;
#endif
      for (int i = 0; i < lat_4dim12; i++) {
        x_o[i] = x_o[i] + p[i] * alpha + s[i] * omega;
      }
      for (int i = 0; i < lat_4dim12; i++) {
        r[i] = s[i] - t[i] * omega;
      }
      nccl_dot(local_result, lat_4dim12, r, r, tmp, zero, nccl_comm, stream);
      r_norm2 = (*tmp);
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
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    err = cudaGetLastError();
    checkCudaErrors(err);
    printf("nccl wilson bistabcg total time: (without malloc free "
           "memcpy) :%.9lf "
           "sec\n",
           double(duration) / 1e9);
    nccl_diff(local_result, lat_4dim12, x_o, ans_o, tmp, latt_tmp0, tmp0, tmp1,
              zero, nccl_comm, stream);
    printf("## difference: %.16f ", (*tmp).real);
    // free
    free_vec(send_vec, recv_vec);
    cudaFree(gauge);
    cudaFree(x_o);
    cudaFree(b__o);
    cudaFree(r);
    cudaFree(r_tilde);
    cudaFree(p);
    cudaFree(v);
    cudaFree(s);
    cudaFree(t);
  }
  // finalizing NCCL
  ncclCommDestroy(nccl_comm);
  // finalizing MPI
  // MPICHECK(MPI_Finalize());
  printf("[MPI Rank %d] Success \n", node_rank);
  // return 0;
}
#endif