#include "../../include/qcu.h"
#ifdef NCCL_WILSON_BISTABCG
// #define DEBUG_NCCL_WILSON_CG
void ncclBistabCgQcu(void *gauge, QcuParam *param, QcuParam *grid) {
  int node_rank, node_size, localRank = 0;
  // initializing MPI
  // MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &node_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &node_size));
  // calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[node_size];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[node_rank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
                         sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p = 0; p < node_size; p++) {
    if (p == node_rank)
      break;
    if (hostHashs[p] == hostHashs[node_rank])
      localRank++;
  }
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
  // define for nccl_wilson_dslash
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
  int move[_BF_];
  int grid_1dim[_DIM_];
  int grid_index_1dim[_DIM_];
  give_grid(grid, node_rank, grid_1dim, grid_index_1dim);
  void *host_send_vec[_WARDS_];
  void *host_recv_vec[_WARDS_];
  void *device_send_vec[_WARDS_];
  void *device_recv_vec[_WARDS_];
  malloc_vec(lat_3dim6, device_send_vec, device_recv_vec, host_send_vec,
             host_recv_vec);
  // define end
  // define for nccl_wilson_cg
  int lat_4dim12 = lat_4dim * 12;
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
  void *ans_e, *ans_o, *x_e, *x_o, *b_e, *b_o, *b__o, *r, *r_tilde, *p, *v, *s,
      *t, *device_latt_tmp0, *device_latt_tmp1, *device_dot_tmp;
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
  cudaMalloc(&device_dot_tmp, lat_4dim * sizeof(LatticeComplex));
  LatticeComplex *host_latt_tmp0 =
      (LatticeComplex *)malloc(lat_4dim12 * sizeof(LatticeComplex));
  LatticeComplex *host_latt_tmp1 =
      (LatticeComplex *)malloc(lat_4dim12 * sizeof(LatticeComplex));
  LatticeComplex *host_dot_tmp =
      (LatticeComplex *)malloc(lat_4dim * sizeof(LatticeComplex));
  // give ans first
  give_random_value<<<gridDim, blockDim>>>(ans_e, node_rank + 12138);
  give_random_value<<<gridDim, blockDim>>>(ans_o, node_rank + 83121);
  // give x_o, b_e, b_o ,b__o, r, r_tilde, p, v, s, t
  give_random_value<<<gridDim, blockDim>>>(x_o, node_rank + 66666);
  give_custom_value<<<gridDim, blockDim>>>(b_e, 0.0, 0.0);
  give_custom_value<<<gridDim, blockDim>>>(b_o, 0.0, 0.0);
  give_custom_value<<<gridDim, blockDim>>>(b__o, 0.0, 0.0);
  give_custom_value<<<gridDim, blockDim>>>(r, 0.0, 0.0);
  give_custom_value<<<gridDim, blockDim>>>(r_tilde, 0.0, 0.0);
  give_custom_value<<<gridDim, blockDim>>>(p, 0.0, 0.0);
  give_custom_value<<<gridDim, blockDim>>>(v, 0.0, 0.0);
  give_custom_value<<<gridDim, blockDim>>>(s, 0.0, 0.0);
  give_custom_value<<<gridDim, blockDim>>>(t, 0.0, 0.0);
  // give b'_o(b__0)
  give_custom_value<<<gridDim, blockDim>>>(device_latt_tmp0, 0.0, 0.0);
  checkCudaErrors(cudaDeviceSynchronize());
  nccl_dslash_eo(device_latt_tmp0, ans_o, node_rank, gridDim, blockDim, gauge,
                 lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move,
                 device_send_vec, device_recv_vec, nccl_comm, stream);
  wilson_bistabcg_give_b_e<<<gridDim, blockDim>>>(b_e, ans_e, device_latt_tmp0,
                                                  kappa);
  checkCudaErrors(cudaDeviceSynchronize());
  give_custom_value<<<gridDim, blockDim>>>(device_latt_tmp1, 0.0, 0.0);
  checkCudaErrors(cudaDeviceSynchronize());
  nccl_dslash_oe(device_latt_tmp1, ans_e, node_rank, gridDim, blockDim, gauge,
                 lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move,
                 device_send_vec, device_recv_vec, nccl_comm, stream);
  wilson_bistabcg_give_b_o<<<gridDim, blockDim>>>(b_o, ans_o, device_latt_tmp1,
                                                  kappa);
  checkCudaErrors(cudaDeviceSynchronize());
  give_custom_value<<<gridDim, blockDim>>>(device_latt_tmp0, 0.0, 0.0);
  checkCudaErrors(cudaDeviceSynchronize());
  nccl_dslash_oe(device_latt_tmp0, b_e, node_rank, gridDim, blockDim, gauge,
                 lat_1dim, lat_3dim12, grid_1dim, grid_index_1dim, move,
                 device_send_vec, device_recv_vec, nccl_comm, stream);
  wilson_bistabcg_give_b__0<<<gridDim, blockDim>>>(b__o, b_o, device_latt_tmp0,
                                                   kappa);
  checkCudaErrors(cudaDeviceSynchronize());
  // bistabcg
  nccl_dslash(r, x_o, kappa, device_latt_tmp0, device_latt_tmp1, node_rank,
              gridDim, blockDim, gauge, lat_1dim, lat_3dim12, lat_4dim12,
              grid_1dim, grid_index_1dim, move, device_send_vec,
              device_recv_vec, nccl_comm, stream);
  wilson_bistabcg_give_rr<<<gridDim, blockDim>>>(r, b__o, r_tilde);
  checkCudaErrors(cudaDeviceSynchronize());
  // define end
  auto start = std::chrono::high_resolution_clock::now();
  for (int loop = 0; loop < MAX_ITER; loop++) {
    nccl_dot(device_dot_tmp, host_dot_tmp, r_tilde, r, rho, gridDim, blockDim);
#ifdef DEBUG_NCCL_WILSON_CG
    std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
              << "##rho:" << rho.real << std::endl;
#endif
    beta = (rho / rho_prev) * (alpha / omega);
#ifdef DEBUG_NCCL_WILSON_CG
    std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
              << "##beta:" << beta.real << std::endl;
#endif
    wilson_bistabcg_give_p<<<gridDim, blockDim>>>(p, r, v, omega, beta);
    checkCudaErrors(cudaDeviceSynchronize());
    // v = A * p;
    nccl_dslash(v, p, kappa, device_latt_tmp0, device_latt_tmp1, node_rank,
                gridDim, blockDim, gauge, lat_1dim, lat_3dim12, lat_4dim12,
                grid_1dim, grid_index_1dim, move, device_send_vec,
                device_recv_vec, nccl_comm, stream);
    nccl_dot(device_dot_tmp, host_dot_tmp, r_tilde, v, tmp, gridDim, blockDim);
    alpha = rho / tmp;
#ifdef DEBUG_NCCL_WILSON_CG
    std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
              << "##alpha:" << alpha.real << std::endl;
#endif
    wilson_bistabcg_give_s<<<gridDim, blockDim>>>(s, r, v, alpha);
    checkCudaErrors(cudaDeviceSynchronize());
    // t = A * s;
    nccl_dslash(t, s, kappa, device_latt_tmp0, device_latt_tmp1, node_rank,
                gridDim, blockDim, gauge, lat_1dim, lat_3dim12, lat_4dim12,
                grid_1dim, grid_index_1dim, move, device_send_vec,
                device_recv_vec, nccl_comm, stream);
    nccl_dot(device_dot_tmp, host_dot_tmp, t, s, tmp0, gridDim, blockDim);
    nccl_dot(device_dot_tmp, host_dot_tmp, t, t, tmp1, gridDim, blockDim);
    omega = tmp0 / tmp1;
#ifdef DEBUG_NCCL_WILSON_CG
    std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
              << "##omega:" << omega.real << std::endl;
#endif
    wilson_bistabcg_give_x_o<<<gridDim, blockDim>>>(x_o, p, s, alpha, omega);
    wilson_bistabcg_give_r<<<gridDim, blockDim>>>(r, s, t, omega);
    checkCudaErrors(cudaDeviceSynchronize());
    nccl_dot(device_dot_tmp, host_dot_tmp, r, r, r_norm2, gridDim, blockDim);
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
  printf("nccl wilson bistabcg total time: (without malloc free "
         "memcpy) :%.9lf "
         "sec\n",
         double(duration) / 1e9);
  nccl_diff(device_dot_tmp, host_dot_tmp, x_o, ans_o, tmp, device_latt_tmp0,
            tmp0, tmp1, gridDim, blockDim);
  printf("## difference: %.16f\n", tmp.real);
  // free
  free_vec(device_send_vec, device_recv_vec, host_send_vec, host_recv_vec);
  cudaFree(ans_e);
  cudaFree(ans_o);
  cudaFree(x_o);
  cudaFree(b__o);
  cudaFree(r);
  cudaFree(r_tilde);
  cudaFree(p);
  cudaFree(v);
  cudaFree(s);
  cudaFree(t);
  cudaFree(device_latt_tmp0);
  cudaFree(device_latt_tmp1);
  free(host_latt_tmp0);
  free(host_latt_tmp1);
}
#endif