#include <iostream>
#pragma optimize(5)
#include "../../include/qcu.h"

#ifdef MPI_WILSON_BISTABCG
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
    const int grid_index_x = node_rank / grid_t / grid_z / grid_y;
    const int grid_index_y = node_rank / grid_t / grid_z % grid_y;
    const int grid_index_z = node_rank / grid_t % grid_z;
    const int grid_index_t = node_rank % grid_t;
    MPI_Request b_x_send_request, b_x_recv_request;
    MPI_Request f_x_send_request, f_x_recv_request;
    MPI_Request b_y_send_request, b_y_recv_request;
    MPI_Request f_y_send_request, f_y_recv_request;
    MPI_Request b_z_send_request, b_z_recv_request;
    MPI_Request f_z_send_request, f_z_recv_request;
    MPI_Request b_t_send_request, b_t_recv_request;
    MPI_Request f_t_send_request, f_t_recv_request;
    void *b_x_send_vec, *b_x_recv_vec;
    void *f_x_send_vec, *f_x_recv_vec;
    void *b_y_send_vec, *b_y_recv_vec;
    void *f_y_send_vec, *f_y_recv_vec;
    void *b_z_send_vec, *b_z_recv_vec;
    void *f_z_send_vec, *f_z_recv_vec;
    void *b_t_send_vec, *b_t_recv_vec;
    void *f_t_send_vec, *f_t_recv_vec;
    cudaMallocManaged(&b_x_send_vec, lat_yzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_x_send_vec, lat_yzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_y_send_vec, lat_xzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_y_send_vec, lat_xzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_z_send_vec, lat_xyt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_z_send_vec, lat_xyt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_t_send_vec, lat_xyz6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_t_send_vec, lat_xyz6 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_x_recv_vec, lat_yzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_x_recv_vec, lat_yzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_y_recv_vec, lat_xzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_y_recv_vec, lat_xzt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_z_recv_vec, lat_xyt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_z_recv_vec, lat_xyt6 * sizeof(LatticeComplex));
    cudaMallocManaged(&b_t_recv_vec, lat_xyz6 * sizeof(LatticeComplex));
    cudaMallocManaged(&f_t_recv_vec, lat_xyz6 * sizeof(LatticeComplex));
    LatticeComplex *cg_in, *cg_out, *x, *b, *r, *r_tilde, *p, *v, *s, *t;
    cudaMallocManaged(&x, lat_xyzt12 * sizeof(LatticeComplex));
    cudaMallocManaged(&b, lat_xyzt12 * sizeof(LatticeComplex));
    cudaMallocManaged(&r, lat_xyzt12 * sizeof(LatticeComplex));
    cudaMallocManaged(&r_tilde, lat_xyzt12 * sizeof(LatticeComplex));
    cudaMallocManaged(&p, lat_xyzt12 * sizeof(LatticeComplex));
    cudaMallocManaged(&v, lat_xyzt12 * sizeof(LatticeComplex));
    cudaMallocManaged(&s, lat_xyzt12 * sizeof(LatticeComplex));
    cudaMallocManaged(&t, lat_xyzt12 * sizeof(LatticeComplex));
    LatticeComplex zero(0.0, 0.0);
    LatticeComplex one(1.0, 0.0);
    const int MAX_ITER(1e2);
    const double TOL(1e-6);
    LatticeComplex rho_prev(1.0, 0.0);
    LatticeComplex rho(0.0, 0.0);
    LatticeComplex alpha(1.0, 0.0);
    LatticeComplex omega(1.0, 0.0);
    LatticeComplex beta(0.0, 0.0);
    LatticeComplex tmp(0.0, 0.0);
    LatticeComplex tmp0(0.0, 0.0);
    LatticeComplex tmp1(0.0, 0.0);
    LatticeComplex r_norm2(0.0, 0.0);
    LatticeComplex local_result = 0;
    // double Kappa = 0.125;
    double Kappa = -7.0;
    // above define for mpi_wilson_dslash and mpi_wilson_cg
    auto start = std::chrono::high_resolution_clock::now();
    give_rand(x, lat_xyzt12); // rand x
    // give_rand(b, lat_xyzt12); // rand source
    // give_value(b, one, 1);    // point source
    cg_in = x;
    cg_out = r;
    // mpi_wilson_dslash
    {
      // clean
      wilson_dslash_clear_dest<<<gridDim, blockDim>>>(cg_out, lat_x, lat_y,
                                                      lat_z);
      // send x
      wilson_dslash_x_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_x,
                                                  lat_y, lat_z, lat_t, parity,
                                                  b_x_send_vec, f_x_send_vec);
      if (grid_x != 1) {
        checkCudaErrors(cudaDeviceSynchronize());
        move_backward(move_b, grid_index_x, grid_x);
        move_forward(move_f, grid_index_x, grid_x);
        move_b = node_rank + move_b * grid_y * grid_z * grid_t;
        move_f = node_rank + move_f * grid_y * grid_z * grid_t;
        MPI_Irecv(b_x_recv_vec, lat_yzt12, MPI_DOUBLE, move_b, 1,
                  MPI_COMM_WORLD, &b_x_recv_request);
        MPI_Irecv(f_x_recv_vec, lat_yzt12, MPI_DOUBLE, move_f, 0,
                  MPI_COMM_WORLD, &f_x_recv_request);
        MPI_Isend(b_x_send_vec, lat_yzt12, MPI_DOUBLE, move_b, 0,
                  MPI_COMM_WORLD, &b_x_send_request);
        MPI_Isend(f_x_send_vec, lat_yzt12, MPI_DOUBLE, move_f, 1,
                  MPI_COMM_WORLD, &f_x_send_request);
      }
      // send y
      wilson_dslash_y_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_x,
                                                  lat_y, lat_z, lat_t, parity,
                                                  b_y_send_vec, f_y_send_vec);
      if (grid_y != 1) {
        checkCudaErrors(cudaDeviceSynchronize());
        move_backward(move_b, grid_index_y, grid_y);
        move_forward(move_f, grid_index_y, grid_y);
        move_b = node_rank + move_b * grid_z * grid_t;
        move_f = node_rank + move_f * grid_z * grid_t;
        MPI_Irecv(b_y_recv_vec, lat_xzt12, MPI_DOUBLE, move_b, 3,
                  MPI_COMM_WORLD, &b_y_recv_request);
        MPI_Irecv(f_y_recv_vec, lat_xzt12, MPI_DOUBLE, move_f, 2,
                  MPI_COMM_WORLD, &f_y_recv_request);
        MPI_Isend(b_y_send_vec, lat_xzt12, MPI_DOUBLE, move_b, 2,
                  MPI_COMM_WORLD, &b_y_send_request);
        MPI_Isend(f_y_send_vec, lat_xzt12, MPI_DOUBLE, move_f, 3,
                  MPI_COMM_WORLD, &f_y_send_request);
      }
      // send z
      wilson_dslash_z_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_x,
                                                  lat_y, lat_z, lat_t, parity,
                                                  b_z_send_vec, f_z_send_vec);
      if (grid_z != 1) {
        checkCudaErrors(cudaDeviceSynchronize());
        move_backward(move_b, grid_index_z, grid_z);
        move_forward(move_f, grid_index_z, grid_z);
        move_b = node_rank + move_b * grid_t;
        move_f = node_rank + move_f * grid_t;
        MPI_Irecv(b_z_recv_vec, lat_xyt12, MPI_DOUBLE, move_b, 5,
                  MPI_COMM_WORLD, &b_z_recv_request);
        MPI_Irecv(f_z_recv_vec, lat_xyt12, MPI_DOUBLE, move_f, 4,
                  MPI_COMM_WORLD, &f_z_recv_request);
        MPI_Isend(b_z_send_vec, lat_xyt12, MPI_DOUBLE, move_b, 4,
                  MPI_COMM_WORLD, &b_z_send_request);
        MPI_Isend(f_z_send_vec, lat_xyt12, MPI_DOUBLE, move_f, 5,
                  MPI_COMM_WORLD, &f_z_send_request);
      }
      // send t
      wilson_dslash_t_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_x,
                                                  lat_y, lat_z, lat_t, parity,
                                                  b_t_send_vec, f_t_send_vec);
      if (grid_t != 1) {
        checkCudaErrors(cudaDeviceSynchronize());
        move_backward(move_b, grid_index_t, grid_t);
        move_forward(move_f, grid_index_t, grid_t);
        move_b = node_rank + move_b;
        move_f = node_rank + move_f;
        MPI_Irecv(b_t_recv_vec, lat_xyz12, MPI_DOUBLE, move_b, 7,
                  MPI_COMM_WORLD, &b_t_recv_request);
        MPI_Irecv(f_t_recv_vec, lat_xyz12, MPI_DOUBLE, move_f, 6,
                  MPI_COMM_WORLD, &f_t_recv_request);
        MPI_Isend(b_t_send_vec, lat_xyz12, MPI_DOUBLE, move_b, 6,
                  MPI_COMM_WORLD, &b_t_send_request);
        MPI_Isend(f_t_send_vec, lat_xyz12, MPI_DOUBLE, move_f, 7,
                  MPI_COMM_WORLD, &f_t_send_request);
      }
      // recv x
      if (grid_x != 1) {
        MPI_Wait(&b_x_recv_request, MPI_STATUS_IGNORE);
        MPI_Wait(&f_x_recv_request, MPI_STATUS_IGNORE);
        wilson_dslash_x_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_x, lat_y,
                                                    lat_z, lat_t, parity,
                                                    b_x_recv_vec, f_x_recv_vec);
      } else {
        checkCudaErrors(cudaDeviceSynchronize());
        wilson_dslash_x_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_x, lat_y,
                                                    lat_z, lat_t, parity,
                                                    f_x_send_vec, b_x_send_vec);
      }
      // recv y
      if (grid_y != 1) {
        MPI_Wait(&b_y_recv_request, MPI_STATUS_IGNORE);
        MPI_Wait(&f_y_recv_request, MPI_STATUS_IGNORE);
        wilson_dslash_y_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_x, lat_y,
                                                    lat_z, lat_t, parity,
                                                    b_y_recv_vec, f_y_recv_vec);
      } else {
        checkCudaErrors(cudaDeviceSynchronize());
        wilson_dslash_y_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_x, lat_y,
                                                    lat_z, lat_t, parity,
                                                    f_y_send_vec, b_y_send_vec);
      }
      // recv z
      if (grid_z != 1) {
        MPI_Wait(&b_z_recv_request, MPI_STATUS_IGNORE);
        MPI_Wait(&f_z_recv_request, MPI_STATUS_IGNORE);
        wilson_dslash_z_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_x, lat_y,
                                                    lat_z, lat_t, parity,
                                                    b_z_recv_vec, f_z_recv_vec);
      } else {
        checkCudaErrors(cudaDeviceSynchronize());
        wilson_dslash_z_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_x, lat_y,
                                                    lat_z, lat_t, parity,
                                                    f_z_send_vec, b_z_send_vec);
      }
      // recv t
      if (grid_t != 1) {
        MPI_Wait(&b_t_recv_request, MPI_STATUS_IGNORE);
        MPI_Wait(&f_t_recv_request, MPI_STATUS_IGNORE);
        wilson_dslash_t_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_x, lat_y,
                                                    lat_z, lat_t, parity,
                                                    b_t_recv_vec, f_t_recv_vec);
      } else {
        checkCudaErrors(cudaDeviceSynchronize());
        wilson_dslash_t_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_x, lat_y,
                                                    lat_z, lat_t, parity,
                                                    f_t_send_vec, b_t_send_vec);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    // kappa
    {
      for (int i = 0; i < lat_xyzt12; i++) {
        cg_out[i] = cg_out[i] - cg_in[i] * Kappa;
      }
    }
    for (int i = 0; i < lat_xyzt12; i++) {
      r[i] = b[i] - r[i];
      r_tilde[i] = r[i];
    }
    for (int loop = 0; loop < MAX_ITER; loop++) {
      {
        for (int i = 0; i < lat_xyzt12; i++) {
          local_result = r_tilde[i].conj() * r[i];
        }
        MPI_Allreduce(&local_result, &rho, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
#ifdef DEBUG
      std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
                << "##rho:" << rho.real << std::endl;
#endif
      beta = (rho / rho_prev) * (alpha / omega);
#ifdef DEBUG
      std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
                << "##beta:" << beta.real << std::endl;
#endif
      for (int i = 0; i < lat_xyzt12; i++) {
        p[i] = r[i] + (p[i] - v[i] * omega) * beta;
      }
      // v = A * p;
      cg_in = p;
      cg_out = v;
      // mpi_wilson_dslash
      {
        // clean
        wilson_dslash_clear_dest<<<gridDim, blockDim>>>(cg_out, lat_x, lat_y,
                                                        lat_z);
        // send x
        wilson_dslash_x_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_x,
                                                    lat_y, lat_z, lat_t, parity,
                                                    b_x_send_vec, f_x_send_vec);
        if (grid_x != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_x, grid_x);
          move_forward(move_f, grid_index_x, grid_x);
          move_b = node_rank + move_b * grid_y * grid_z * grid_t;
          move_f = node_rank + move_f * grid_y * grid_z * grid_t;
          MPI_Irecv(b_x_recv_vec, lat_yzt12, MPI_DOUBLE, move_b, 1,
                    MPI_COMM_WORLD, &b_x_recv_request);
          MPI_Irecv(f_x_recv_vec, lat_yzt12, MPI_DOUBLE, move_f, 0,
                    MPI_COMM_WORLD, &f_x_recv_request);
          MPI_Isend(b_x_send_vec, lat_yzt12, MPI_DOUBLE, move_b, 0,
                    MPI_COMM_WORLD, &b_x_send_request);
          MPI_Isend(f_x_send_vec, lat_yzt12, MPI_DOUBLE, move_f, 1,
                    MPI_COMM_WORLD, &f_x_send_request);
        }
        // send y
        wilson_dslash_y_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_x,
                                                    lat_y, lat_z, lat_t, parity,
                                                    b_y_send_vec, f_y_send_vec);
        if (grid_y != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_y, grid_y);
          move_forward(move_f, grid_index_y, grid_y);
          move_b = node_rank + move_b * grid_z * grid_t;
          move_f = node_rank + move_f * grid_z * grid_t;
          MPI_Irecv(b_y_recv_vec, lat_xzt12, MPI_DOUBLE, move_b, 3,
                    MPI_COMM_WORLD, &b_y_recv_request);
          MPI_Irecv(f_y_recv_vec, lat_xzt12, MPI_DOUBLE, move_f, 2,
                    MPI_COMM_WORLD, &f_y_recv_request);
          MPI_Isend(b_y_send_vec, lat_xzt12, MPI_DOUBLE, move_b, 2,
                    MPI_COMM_WORLD, &b_y_send_request);
          MPI_Isend(f_y_send_vec, lat_xzt12, MPI_DOUBLE, move_f, 3,
                    MPI_COMM_WORLD, &f_y_send_request);
        }
        // send z
        wilson_dslash_z_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_x,
                                                    lat_y, lat_z, lat_t, parity,
                                                    b_z_send_vec, f_z_send_vec);
        if (grid_z != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_z, grid_z);
          move_forward(move_f, grid_index_z, grid_z);
          move_b = node_rank + move_b * grid_t;
          move_f = node_rank + move_f * grid_t;
          MPI_Irecv(b_z_recv_vec, lat_xyt12, MPI_DOUBLE, move_b, 5,
                    MPI_COMM_WORLD, &b_z_recv_request);
          MPI_Irecv(f_z_recv_vec, lat_xyt12, MPI_DOUBLE, move_f, 4,
                    MPI_COMM_WORLD, &f_z_recv_request);
          MPI_Isend(b_z_send_vec, lat_xyt12, MPI_DOUBLE, move_b, 4,
                    MPI_COMM_WORLD, &b_z_send_request);
          MPI_Isend(f_z_send_vec, lat_xyt12, MPI_DOUBLE, move_f, 5,
                    MPI_COMM_WORLD, &f_z_send_request);
        }
        // send t
        wilson_dslash_t_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_x,
                                                    lat_y, lat_z, lat_t, parity,
                                                    b_t_send_vec, f_t_send_vec);
        if (grid_t != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_t, grid_t);
          move_forward(move_f, grid_index_t, grid_t);
          move_b = node_rank + move_b;
          move_f = node_rank + move_f;
          MPI_Irecv(b_t_recv_vec, lat_xyz12, MPI_DOUBLE, move_b, 7,
                    MPI_COMM_WORLD, &b_t_recv_request);
          MPI_Irecv(f_t_recv_vec, lat_xyz12, MPI_DOUBLE, move_f, 6,
                    MPI_COMM_WORLD, &f_t_recv_request);
          MPI_Isend(b_t_send_vec, lat_xyz12, MPI_DOUBLE, move_b, 6,
                    MPI_COMM_WORLD, &b_t_send_request);
          MPI_Isend(f_t_send_vec, lat_xyz12, MPI_DOUBLE, move_f, 7,
                    MPI_COMM_WORLD, &f_t_send_request);
        }
        // recv x
        if (grid_x != 1) {
          MPI_Wait(&b_x_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_x_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_x_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, b_x_recv_vec,
              f_x_recv_vec);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_x_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, f_x_send_vec,
              b_x_send_vec);
        }
        // recv y
        if (grid_y != 1) {
          MPI_Wait(&b_y_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_y_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_y_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, b_y_recv_vec,
              f_y_recv_vec);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_y_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, f_y_send_vec,
              b_y_send_vec);
        }
        // recv z
        if (grid_z != 1) {
          MPI_Wait(&b_z_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_z_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_z_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, b_z_recv_vec,
              f_z_recv_vec);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_z_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, f_z_send_vec,
              b_z_send_vec);
        }
        // recv t
        if (grid_t != 1) {
          MPI_Wait(&b_t_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_t_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_t_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, b_t_recv_vec,
              f_t_recv_vec);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_t_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, f_t_send_vec,
              b_t_send_vec);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
      // kappa
      {
        for (int i = 0; i < lat_xyzt12; i++) {
          cg_out[i] = cg_out[i] - cg_in[i] * Kappa;
        }
      }    
      {
        for (int i = 0; i < lat_xyzt12; i++) {
          local_result = r_tilde[i].conj() * v[i];
        }
        MPI_Allreduce(&local_result, &tmp, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      alpha = rho / tmp;
#ifdef DEBUG
      std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
                << "##alpha:" << alpha.real << std::endl;
#endif
      for (int i = 0; i < lat_xyzt12; i++) {
        s[i] = r[i] - v[i] * alpha;
      }
      // t = A * s;
      cg_in = s;
      cg_out = t;
      // mpi_wilson_dslash
      {
        // clean
        wilson_dslash_clear_dest<<<gridDim, blockDim>>>(cg_out, lat_x, lat_y,
                                                        lat_z);
        // send x
        wilson_dslash_x_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_x,
                                                    lat_y, lat_z, lat_t, parity,
                                                    b_x_send_vec, f_x_send_vec);
        if (grid_x != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_x, grid_x);
          move_forward(move_f, grid_index_x, grid_x);
          move_b = node_rank + move_b * grid_y * grid_z * grid_t;
          move_f = node_rank + move_f * grid_y * grid_z * grid_t;
          MPI_Irecv(b_x_recv_vec, lat_yzt12, MPI_DOUBLE, move_b, 1,
                    MPI_COMM_WORLD, &b_x_recv_request);
          MPI_Irecv(f_x_recv_vec, lat_yzt12, MPI_DOUBLE, move_f, 0,
                    MPI_COMM_WORLD, &f_x_recv_request);
          MPI_Isend(b_x_send_vec, lat_yzt12, MPI_DOUBLE, move_b, 0,
                    MPI_COMM_WORLD, &b_x_send_request);
          MPI_Isend(f_x_send_vec, lat_yzt12, MPI_DOUBLE, move_f, 1,
                    MPI_COMM_WORLD, &f_x_send_request);
        }
        // send y
        wilson_dslash_y_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_x,
                                                    lat_y, lat_z, lat_t, parity,
                                                    b_y_send_vec, f_y_send_vec);
        if (grid_y != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_y, grid_y);
          move_forward(move_f, grid_index_y, grid_y);
          move_b = node_rank + move_b * grid_z * grid_t;
          move_f = node_rank + move_f * grid_z * grid_t;
          MPI_Irecv(b_y_recv_vec, lat_xzt12, MPI_DOUBLE, move_b, 3,
                    MPI_COMM_WORLD, &b_y_recv_request);
          MPI_Irecv(f_y_recv_vec, lat_xzt12, MPI_DOUBLE, move_f, 2,
                    MPI_COMM_WORLD, &f_y_recv_request);
          MPI_Isend(b_y_send_vec, lat_xzt12, MPI_DOUBLE, move_b, 2,
                    MPI_COMM_WORLD, &b_y_send_request);
          MPI_Isend(f_y_send_vec, lat_xzt12, MPI_DOUBLE, move_f, 3,
                    MPI_COMM_WORLD, &f_y_send_request);
        }
        // send z
        wilson_dslash_z_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_x,
                                                    lat_y, lat_z, lat_t, parity,
                                                    b_z_send_vec, f_z_send_vec);
        if (grid_z != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_z, grid_z);
          move_forward(move_f, grid_index_z, grid_z);
          move_b = node_rank + move_b * grid_t;
          move_f = node_rank + move_f * grid_t;
          MPI_Irecv(b_z_recv_vec, lat_xyt12, MPI_DOUBLE, move_b, 5,
                    MPI_COMM_WORLD, &b_z_recv_request);
          MPI_Irecv(f_z_recv_vec, lat_xyt12, MPI_DOUBLE, move_f, 4,
                    MPI_COMM_WORLD, &f_z_recv_request);
          MPI_Isend(b_z_send_vec, lat_xyt12, MPI_DOUBLE, move_b, 4,
                    MPI_COMM_WORLD, &b_z_send_request);
          MPI_Isend(f_z_send_vec, lat_xyt12, MPI_DOUBLE, move_f, 5,
                    MPI_COMM_WORLD, &f_z_send_request);
        }
        // send t
        wilson_dslash_t_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_x,
                                                    lat_y, lat_z, lat_t, parity,
                                                    b_t_send_vec, f_t_send_vec);
        if (grid_t != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_t, grid_t);
          move_forward(move_f, grid_index_t, grid_t);
          move_b = node_rank + move_b;
          move_f = node_rank + move_f;
          MPI_Irecv(b_t_recv_vec, lat_xyz12, MPI_DOUBLE, move_b, 7,
                    MPI_COMM_WORLD, &b_t_recv_request);
          MPI_Irecv(f_t_recv_vec, lat_xyz12, MPI_DOUBLE, move_f, 6,
                    MPI_COMM_WORLD, &f_t_recv_request);
          MPI_Isend(b_t_send_vec, lat_xyz12, MPI_DOUBLE, move_b, 6,
                    MPI_COMM_WORLD, &b_t_send_request);
          MPI_Isend(f_t_send_vec, lat_xyz12, MPI_DOUBLE, move_f, 7,
                    MPI_COMM_WORLD, &f_t_send_request);
        }
        // recv x
        if (grid_x != 1) {
          MPI_Wait(&b_x_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_x_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_x_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, b_x_recv_vec,
              f_x_recv_vec);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_x_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, f_x_send_vec,
              b_x_send_vec);
        }
        // recv y
        if (grid_y != 1) {
          MPI_Wait(&b_y_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_y_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_y_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, b_y_recv_vec,
              f_y_recv_vec);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_y_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, f_y_send_vec,
              b_y_send_vec);
        }
        // recv z
        if (grid_z != 1) {
          MPI_Wait(&b_z_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_z_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_z_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, b_z_recv_vec,
              f_z_recv_vec);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_z_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, f_z_send_vec,
              b_z_send_vec);
        }
        // recv t
        if (grid_t != 1) {
          MPI_Wait(&b_t_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_t_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_t_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, b_t_recv_vec,
              f_t_recv_vec);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_t_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_x, lat_y, lat_z, lat_t, parity, f_t_send_vec,
              b_t_send_vec);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
      // kappa
      {
        for (int i = 0; i < lat_xyzt12; i++) {
          cg_out[i] = cg_out[i] - cg_in[i] * Kappa;
        }
      }
      {
        for (int i = 0; i < lat_xyzt12; i++) {
          local_result = t[i].conj() * s[i];
        }
        MPI_Allreduce(&local_result, &tmp0, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      {
        for (int i = 0; i < lat_xyzt12; i++) {
          local_result = t[i].conj() * t[i];
        }
        MPI_Allreduce(&local_result, &tmp1, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      omega = tmp0 / tmp1;
#ifdef DEBUG
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
        for (int i = 0; i < lat_xyzt12; i++) {
          local_result = r[i].conj() * r[i];
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
      checkCudaErrors(cudaFree(b_x_send_vec));
      checkCudaErrors(cudaFree(f_x_send_vec));
      checkCudaErrors(cudaFree(b_y_send_vec));
      checkCudaErrors(cudaFree(f_y_send_vec));
      checkCudaErrors(cudaFree(b_z_send_vec));
      checkCudaErrors(cudaFree(f_z_send_vec));
      checkCudaErrors(cudaFree(b_t_send_vec));
      checkCudaErrors(cudaFree(f_t_send_vec));
      checkCudaErrors(cudaFree(b_x_recv_vec));
      checkCudaErrors(cudaFree(f_x_recv_vec));
      checkCudaErrors(cudaFree(b_y_recv_vec));
      checkCudaErrors(cudaFree(f_y_recv_vec));
      checkCudaErrors(cudaFree(b_z_recv_vec));
      checkCudaErrors(cudaFree(f_z_recv_vec));
      checkCudaErrors(cudaFree(b_t_recv_vec));
      checkCudaErrors(cudaFree(f_t_recv_vec));
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