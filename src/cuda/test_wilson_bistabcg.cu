#include <iostream>
#pragma optimize(5)
#include "../../include/qcu.h"

// #define DEBUG_TEST_WILSON_BISTABCG
#ifdef TEST_WILSON_BISTABCG
void mpiCgQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param,
              int parity, QcuParam *grid) {
  const int lat_x = param->lattice_size[0] >> 1;
  const int lat_y = param->lattice_size[1];
  const int lat_z = param->lattice_size[2];
  const int lat_t = param->lattice_size[3];
  const int lat_yzt6 = lat_y * lat_z * lat_t * 6;
  const int lat_xzt6 = lat_x * lat_z * lat_t * 6;
  const int lat_xyt6 = lat_x * lat_y * lat_t * 6;
  const int lat_xyz6 = lat_x * lat_y * lat_z * 6;
  const int lat_yzt12 = lat_yzt6 * 2;
  const int lat_xzt12 = lat_xzt6 * 2;
  const int lat_xyt12 = lat_xyt6 * 2;
  const int lat_xyz12 = lat_xyz6 * 2;
  const int lat_xyzt12 = lat_xyz6 * lat_t * 2;
  cudaError_t err;
  dim3 gridDim(lat_x * lat_y * lat_z * lat_t / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  {
    // mpi wilson cg
    int node_size, node_rank, move_b, move_f;
    MPI_Comm_size(MPI_COMM_WORLD, &node_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_rank);
    const int grid_x = grid->lattice_size[0];
    const int grid_y = grid->lattice_size[1];
    const int grid_z = grid->lattice_size[2];
    const int grid_t = grid->lattice_size[3];
    LatticeComplex *cg_in, *cg_out, *x, *b, *r, *r_tilde, *p, *v, *s, *t;
    cudaMallocManaged(&x, lat_xyzt12 * sizeof(LatticeComplex));
    cudaMallocManaged(&b, lat_xyzt12 * sizeof(LatticeComplex));
    cudaMallocManaged(&r, lat_xyzt12 * sizeof(LatticeComplex));
    cudaMallocManaged(&r_tilde, lat_xyzt12 * sizeof(LatticeComplex));
    cudaMallocManaged(&p, lat_xyzt12 * sizeof(LatticeComplex));
    cudaMallocManaged(&v, lat_xyzt12 * sizeof(LatticeComplex));
    cudaMallocManaged(&s, lat_xyzt12 * sizeof(LatticeComplex));
    cudaMallocManaged(&t, lat_xyzt12 * sizeof(LatticeComplex));
    // LatticeComplex x_norm2(0.0, 0.0);
    // LatticeComplex b_norm2(0.0, 0.0);
    LatticeComplex r_norm2(0.0, 0.0);
    // LatticeComplex r_tilde_norm2(0.0, 0.0);
    // LatticeComplex p_norm2(0.0, 0.0);
    // LatticeComplex v_norm2(0.0, 0.0);
    // LatticeComplex s_norm2(0.0, 0.0);
    // LatticeComplex t_norm2(0.0, 0.0);
    LatticeComplex zero(0.0, 0.0);
    LatticeComplex one(1.0, 0.0);
    // const int MAX_ITER(1e2); // 300++?
    const int MAX_ITER(1e3); // ?
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
    // above define for TEST_WILSON_dslash and TEST_WILSON_cg
    auto start = std::chrono::high_resolution_clock::now();
    give_rand(x, lat_xyzt12); // rand x
    // give_value(x, zero, lat_xyzt12); // zero x
    give_rand(b, lat_xyzt12); // rand b
    // give_value(b, one, 1); // point b
    give_value(r, zero, lat_xyzt12);       // zero r
    give_value(r_tilde, zero, lat_xyzt12); // zero r_tilde
    give_value(p, zero, lat_xyzt12);       // zero p
    give_value(v, zero, lat_xyzt12);       // zero v
    give_value(s, zero, lat_xyzt12);       // zero s
    give_value(t, zero, lat_xyzt12);       // zero t
    cg_in = x;
    cg_out = r;
    for (int i = 0; i < lat_xyzt12; i++) {
      cg_out[i] = cg_in[i] * 2 + one;
    }
    for (int i = 0; i < lat_xyzt12; i++) {
      r[i] = b[i] - r[i];
      r_tilde[i] = r[i];
    }
    for (int loop = 0; loop < MAX_ITER; loop++) {
      {
        local_result = zero;
        for (int i = 0; i < lat_xyzt12; i++) {
          local_result += r_tilde[i].conj() * r[i];
        }
        MPI_Allreduce(&local_result, &rho, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
#ifdef DEBUG_MPI_WILSON_BISTABCG
      std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
                << "##rho:" << rho.real << std::endl;
#endif
      beta = (rho / rho_prev) * (alpha / omega);
#ifdef DEBUG_MPI_WILSON_BISTABCG
      std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
                << "##beta:" << beta.real << std::endl;
#endif
      for (int i = 0; i < lat_xyzt12; i++) {
        p[i] = r[i] + (p[i] - v[i] * omega) * beta;
      }
      // v = A * p;
      cg_in = p;
      cg_out = v;
      for (int i = 0; i < lat_xyzt12; i++) {
        cg_out[i] = cg_in[i] * 2 + one;
      }
      {
        local_result = zero;
        for (int i = 0; i < lat_xyzt12; i++) {
          local_result += r_tilde[i].conj() * v[i];
        }
        MPI_Allreduce(&local_result, &tmp, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      alpha = rho / tmp;
#ifdef DEBUG_MPI_WILSON_BISTABCG
      std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
                << "##alpha:" << alpha.real << std::endl;
#endif
      for (int i = 0; i < lat_xyzt12; i++) {
        s[i] = r[i] - v[i] * alpha;
      }
      // t = A * s;
      cg_in = s;
      cg_out = t;
      for (int i = 0; i < lat_xyzt12; i++) {
        cg_out[i] = cg_in[i] * 2 + one;
      }
      {
        local_result = zero;
        for (int i = 0; i < lat_xyzt12; i++) {
          local_result += t[i].conj() * s[i];
        }
        MPI_Allreduce(&local_result, &tmp0, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      {
        local_result = zero;
        for (int i = 0; i < lat_xyzt12; i++) {
          local_result += t[i].conj() * t[i];
        }
        MPI_Allreduce(&local_result, &tmp1, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      omega = tmp0 / tmp1;
#ifdef DEBUG_MPI_WILSON_BISTABCG
      std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
                << "##omega:" << omega.real << std::endl;
#endif
      for (int i = 0; i < lat_xyzt12; i++) {
        x[i] = x[i] + p[i] * alpha + s[i] * omega;
      }
      for (int i = 0; i < lat_xyzt12; i++) {
        r[i] = s[i] - t[i] * omega;
      }
      {
        local_result = zero;
        for (int i = 0; i < lat_xyzt12; i++) {
          local_result += r[i].conj() * r[i];
        }
        MPI_Allreduce(&local_result, &r_norm2, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
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
    printf("mpi wilson cg total time: (without malloc free memcpy) :%.9lf "
           "sec\n",
           double(duration) / 1e9);
    {
      // free
      checkCudaErrors(cudaFree(x));
      checkCudaErrors(cudaFree(b));
      checkCudaErrors(cudaFree(r));
      checkCudaErrors(cudaFree(r_tilde));
      checkCudaErrors(cudaFree(p));
      checkCudaErrors(cudaFree(v));
      checkCudaErrors(cudaFree(s));
      checkCudaErrors(cudaFree(t));
    }
  }
}
#endif