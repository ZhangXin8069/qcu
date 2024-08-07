#include <iostream>
#pragma optimize(5)
#include "../../include/qcu.h"

// #define DEBUG_MPI_WILSON_CG
// #define TEST_MPI_WILSON_CG
// #define TEST_MPI_WILSON_CG_USE_WILSON_DSLASH

#ifdef MPI_WILSON_CG
void mpiCgQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param,
              int parity, QcuParam *grid) {
  int lat_1dim[DIM];
  int lat_3dim6[DIM];
  int lat_3dim12[DIM];
  int lat_4dim12;
  give_dims(param, lat_1dim, lat_3dim6, lat_3dim12, lat_4dim12);
  cudaError_t err;
  dim3 gridDim(lat_1dim[X] * lat_1dim[Y] * lat_1dim[Z] * lat_1dim[T] /
               BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  {
    // mpi wilson cg
    int node_rank;
    int move[BF];
    int grid_1dim[DIM];
    int grid_index_1dim[DIM];
    give_grid(grid, node_rank, grid_1dim, grid_index_1dim);
    MPI_Request send_request[WARDS];
    MPI_Request recv_request[WARDS];
    void *send_vec[WARDS];
    void *recv_vec[WARDS];
    cudaMallocManaged(&send_vec[B_X], lat_3dim6[YZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&send_vec[F_X], lat_3dim6[YZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&send_vec[B_Y], lat_3dim6[XZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&send_vec[F_Y], lat_3dim6[XZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&send_vec[B_Z], lat_3dim6[XYT] * sizeof(LatticeComplex));
    cudaMallocManaged(&send_vec[F_Z], lat_3dim6[XYT] * sizeof(LatticeComplex));
    cudaMallocManaged(&send_vec[B_T], lat_3dim6[XYZ] * sizeof(LatticeComplex));
    cudaMallocManaged(&send_vec[F_T], lat_3dim6[XYZ] * sizeof(LatticeComplex));
    cudaMallocManaged(&recv_vec[B_X], lat_3dim6[YZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&recv_vec[F_X], lat_3dim6[YZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&recv_vec[B_Y], lat_3dim6[XZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&recv_vec[F_Y], lat_3dim6[XZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&recv_vec[B_Z], lat_3dim6[XYT] * sizeof(LatticeComplex));
    cudaMallocManaged(&recv_vec[F_Z], lat_3dim6[XYT] * sizeof(LatticeComplex));
    cudaMallocManaged(&recv_vec[B_T], lat_3dim6[XYZ] * sizeof(LatticeComplex));
    cudaMallocManaged(&recv_vec[F_T], lat_3dim6[XYZ] * sizeof(LatticeComplex));
    LatticeComplex *dslash_in, *dslash_out, *x, *b, *r, *r_tilde, *p, *v, *s, *t;
    cudaMallocManaged(&x, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&b, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&r, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&r_tilde, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&p, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&v, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&s, lat_4dim12 * sizeof(LatticeComplex));
    cudaMallocManaged(&t, lat_4dim12 * sizeof(LatticeComplex));
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
    int MAX_ITER(1e2); // 300++?
    double TOL(1e-6);
    LatticeComplex rho_prev(1.0, 0.0);
    LatticeComplex rho(0.0, 0.0);
    LatticeComplex alpha(1.0, 0.0);
    LatticeComplex omega(1.0, 0.0);
    LatticeComplex beta(0.0, 0.0);
    LatticeComplex tmp(0.0, 0.0);
    LatticeComplex tmp0(0.0, 0.0);
    LatticeComplex tmp1(0.0, 0.0);
    LatticeComplex local_result(0.0, 0.0);
    double Kappa = 0.125;
    // above define for mpi_wilson_dslash and mpi_wilson_cg
    auto start = std::chrono::high_resolution_clock::now();
    give_rand(x, lat_4dim12); // rand x
    // give_value(x, zero, lat_4dim12);    // zero x
    // give_rand(b, lat_4dim12);           // rand b
    give_value(b, one, 1);                 // point b
    give_value(r, zero, lat_4dim12);       // zero r
    give_value(r_tilde, zero, lat_4dim12); // zero r_tilde
    give_value(p, zero, lat_4dim12);       // zero p
    give_value(v, zero, lat_4dim12);       // zero v
    give_value(s, zero, lat_4dim12);       // zero s
    give_value(t, zero, lat_4dim12);       // zero t
    dslash_in = x;
    dslash_out = r;
#ifndef TEST_MPI_WILSON_CG
    // clear vecs for mpi_wilson_dslash
    {
      give_value(send_vec[B_X], zero, lat_3dim6[YZT]);
      give_value(send_vec[F_X], zero, lat_3dim6[YZT]);
      give_value(send_vec[B_Y], zero, lat_3dim6[XZT]);
      give_value(send_vec[F_Y], zero, lat_3dim6[XZT]);
      give_value(send_vec[B_Z], zero, lat_3dim6[XYT]);
      give_value(send_vec[F_Z], zero, lat_3dim6[XYT]);
      give_value(send_vec[B_T], zero, lat_3dim6[XYZ]);
      give_value(send_vec[F_T], zero, lat_3dim6[XYZ]);
      give_value(recv_vec[B_X], zero, lat_3dim6[YZT]);
      give_value(recv_vec[F_X], zero, lat_3dim6[YZT]);
      give_value(recv_vec[B_Y], zero, lat_3dim6[XZT]);
      give_value(recv_vec[F_Y], zero, lat_3dim6[XZT]);
      give_value(recv_vec[B_Z], zero, lat_3dim6[XYT]);
      give_value(recv_vec[F_Z], zero, lat_3dim6[XYT]);
      give_value(recv_vec[B_T], zero, lat_3dim6[XYZ]);
      give_value(recv_vec[F_T], zero, lat_3dim6[XYZ]);
    }
    // mpi_wilson_dslash
    {
      // clean
      wilson_dslash_clear_dest<<<gridDim, blockDim>>>(dslash_out, lat_1dim[X],
                                                      lat_1dim[Y], lat_1dim[Z]);
      // send x
      wilson_dslash_x_send<<<gridDim, blockDim>>>(
          gauge, dslash_in, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
          lat_1dim[T], parity, send_vec[B_X], send_vec[F_X]);
      if (grid_1dim[X] != 1) {
        checkCudaErrors(cudaDeviceSynchronize());
        move_backward(move[B], grid_index_1dim[X], grid_1dim[X]);
        move_forward(move[F], grid_index_1dim[X], grid_1dim[X]);
        move[B] =
            node_rank + move[B] * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];
        move[F] =
            node_rank + move[F] * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];
        MPI_Irecv(recv_vec[B_X], lat_3dim12[YZT], MPI_DOUBLE, move[B], 1,
                  MPI_COMM_WORLD, &recv_request[B_X]);
        MPI_Irecv(recv_vec[F_X], lat_3dim12[YZT], MPI_DOUBLE, move[F], 0,
                  MPI_COMM_WORLD, &recv_request[F_X]);
        MPI_Isend(send_vec[B_X], lat_3dim12[YZT], MPI_DOUBLE, move[B], 0,
                  MPI_COMM_WORLD, &send_request[B_X]);
        MPI_Isend(send_vec[F_X], lat_3dim12[YZT], MPI_DOUBLE, move[F], 1,
                  MPI_COMM_WORLD, &send_request[F_T]);
      }
      // send y
      wilson_dslash_y_send<<<gridDim, blockDim>>>(
          gauge, dslash_in, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
          lat_1dim[T], parity, send_vec[B_Y], send_vec[F_Y]);
      if (grid_1dim[Y] != 1) {
        checkCudaErrors(cudaDeviceSynchronize());
        move_backward(move[B], grid_index_1dim[Y], grid_1dim[Y]);
        move_forward(move[F], grid_index_1dim[Y], grid_1dim[Y]);
        move[B] = node_rank + move[B] * grid_1dim[Z] * grid_1dim[T];
        move[F] = node_rank + move[F] * grid_1dim[Z] * grid_1dim[T];
        MPI_Irecv(recv_vec[B_Y], lat_3dim12[XZT], MPI_DOUBLE, move[B], 3,
                  MPI_COMM_WORLD, &recv_request[B_Y]);
        MPI_Irecv(recv_vec[F_Y], lat_3dim12[XZT], MPI_DOUBLE, move[F], 2,
                  MPI_COMM_WORLD, &recv_request[F_Y]);
        MPI_Isend(send_vec[B_Y], lat_3dim12[XZT], MPI_DOUBLE, move[B], 2,
                  MPI_COMM_WORLD, &send_request[B_Y]);
        MPI_Isend(send_vec[F_Y], lat_3dim12[XZT], MPI_DOUBLE, move[F], 3,
                  MPI_COMM_WORLD, &send_request[F_Y]);
      }
      // send z
      wilson_dslash_z_send<<<gridDim, blockDim>>>(
          gauge, dslash_in, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
          lat_1dim[T], parity, send_vec[B_Z], send_vec[F_Z]);
      if (grid_1dim[Z] != 1) {
        checkCudaErrors(cudaDeviceSynchronize());
        move_backward(move[B], grid_index_1dim[Z], grid_1dim[Z]);
        move_forward(move[F], grid_index_1dim[Z], grid_1dim[Z]);
        move[B] = node_rank + move[B] * grid_1dim[T];
        move[F] = node_rank + move[F] * grid_1dim[T];
        MPI_Irecv(recv_vec[B_Z], lat_3dim12[XYT], MPI_DOUBLE, move[B], 5,
                  MPI_COMM_WORLD, &recv_request[B_Z]);
        MPI_Irecv(recv_vec[F_Z], lat_3dim12[XYT], MPI_DOUBLE, move[F], 4,
                  MPI_COMM_WORLD, &recv_request[F_Z]);
        MPI_Isend(send_vec[B_Z], lat_3dim12[XYT], MPI_DOUBLE, move[B], 4,
                  MPI_COMM_WORLD, &send_request[B_Z]);
        MPI_Isend(send_vec[F_Z], lat_3dim12[XYT], MPI_DOUBLE, move[F], 5,
                  MPI_COMM_WORLD, &send_request[F_Z]);
      }
      // send t
      wilson_dslash_t_send<<<gridDim, blockDim>>>(
          gauge, dslash_in, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
          lat_1dim[T], parity, send_vec[B_T], send_vec[F_T]);
      if (grid_1dim[T] != 1) {
        checkCudaErrors(cudaDeviceSynchronize());
        move_backward(move[B], grid_index_1dim[T], grid_1dim[T]);
        move_forward(move[F], grid_index_1dim[T], grid_1dim[T]);
        move[B] = node_rank + move[B];
        move[F] = node_rank + move[F];
        MPI_Irecv(recv_vec[B_T], lat_3dim12[XYZ], MPI_DOUBLE, move[B], 7,
                  MPI_COMM_WORLD, &recv_request[B_T]);
        MPI_Irecv(recv_vec[F_T], lat_3dim12[XYZ], MPI_DOUBLE, move[F], 6,
                  MPI_COMM_WORLD, &recv_request[F_T]);
        MPI_Isend(send_vec[B_T], lat_3dim12[XYZ], MPI_DOUBLE, move[B], 6,
                  MPI_COMM_WORLD, &send_request[B_T]);
        MPI_Isend(send_vec[F_T], lat_3dim12[XYZ], MPI_DOUBLE, move[F], 7,
                  MPI_COMM_WORLD, &send_request[F_T]);
      }
      // recv x
      if (grid_1dim[X] != 1) {
        MPI_Wait(&recv_request[B_X], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_request[F_X], MPI_STATUS_IGNORE);
        wilson_dslash_x_recv<<<gridDim, blockDim>>>(
            gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, recv_vec[B_X], recv_vec[F_X]);
      } else {
        checkCudaErrors(cudaDeviceSynchronize());
        wilson_dslash_x_recv<<<gridDim, blockDim>>>(
            gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, send_vec[F_X], send_vec[B_X]);
      }
      // recv y
      if (grid_1dim[Y] != 1) {
        MPI_Wait(&recv_request[B_Y], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_request[F_Y], MPI_STATUS_IGNORE);
        wilson_dslash_y_recv<<<gridDim, blockDim>>>(
            gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, recv_vec[B_Y], recv_vec[F_Y]);
      } else {
        checkCudaErrors(cudaDeviceSynchronize());
        wilson_dslash_y_recv<<<gridDim, blockDim>>>(
            gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, send_vec[F_Y], send_vec[B_Y]);
      }
      // recv z
      if (grid_1dim[Z] != 1) {
        MPI_Wait(&recv_request[B_Z], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_request[F_Z], MPI_STATUS_IGNORE);
        wilson_dslash_z_recv<<<gridDim, blockDim>>>(
            gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, recv_vec[B_Z], recv_vec[F_Z]);
      } else {
        checkCudaErrors(cudaDeviceSynchronize());
        wilson_dslash_z_recv<<<gridDim, blockDim>>>(
            gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, send_vec[F_Z], send_vec[B_Z]);
      }
      // recv t
      if (grid_1dim[T] != 1) {
        MPI_Wait(&recv_request[B_T], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_request[F_T], MPI_STATUS_IGNORE);
        wilson_dslash_t_recv<<<gridDim, blockDim>>>(
            gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, recv_vec[B_T], recv_vec[F_T]);
      } else {
        checkCudaErrors(cudaDeviceSynchronize());
        wilson_dslash_t_recv<<<gridDim, blockDim>>>(
            gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, send_vec[F_T], send_vec[B_T]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    // kappa
    {
      for (int i = 0; i < lat_4dim12; i++) {
        dslash_out[i] = dslash_in[i] - dslash_out[i] * Kappa;
      }
    }
#else
#ifdef TEST_MPI_WILSON_CG_USE_WILSON_DSLASH
    // wilson_dslash
    wilson_dslash<<<gridDim, blockDim>>>(gauge, dslash_in, dslash_out,
                                         lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
                                         lat_1dim[T], parity);
    // kappa
    {
      for (int i = 0; i < lat_4dim12; i++) {
        dslash_out[i] = dslash_in[i] - dslash_out[i] * Kappa;
      }
    }
#else
    for (int i = 0; i < lat_4dim12; i++) {
      dslash_out[i] = dslash_in[i] * 2 + one;
    }
#endif
#endif
    for (int i = 0; i < lat_4dim12; i++) {
      r[i] = b[i] - r[i];
      r_tilde[i] = r[i];
    }
    for (int loop = 0; loop < MAX_ITER; loop++) {
      {
        local_result = zero;
        for (int i = 0; i < lat_4dim12; i++) {
          local_result += r_tilde[i].conj() * r[i];
        }
        MPI_Allreduce(&local_result, &rho, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
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
#ifndef TEST_MPI_WILSON_CG
      // clear vecs for mpi_wilson_dslash
      {
        give_value(send_vec[B_X], zero, lat_3dim6[YZT]);
        give_value(send_vec[F_X], zero, lat_3dim6[YZT]);
        give_value(send_vec[B_Y], zero, lat_3dim6[XZT]);
        give_value(send_vec[F_Y], zero, lat_3dim6[XZT]);
        give_value(send_vec[B_Z], zero, lat_3dim6[XYT]);
        give_value(send_vec[F_Z], zero, lat_3dim6[XYT]);
        give_value(send_vec[B_T], zero, lat_3dim6[XYZ]);
        give_value(send_vec[F_T], zero, lat_3dim6[XYZ]);
        give_value(recv_vec[B_X], zero, lat_3dim6[YZT]);
        give_value(recv_vec[F_X], zero, lat_3dim6[YZT]);
        give_value(recv_vec[B_Y], zero, lat_3dim6[XZT]);
        give_value(recv_vec[F_Y], zero, lat_3dim6[XZT]);
        give_value(recv_vec[B_Z], zero, lat_3dim6[XYT]);
        give_value(recv_vec[F_Z], zero, lat_3dim6[XYT]);
        give_value(recv_vec[B_T], zero, lat_3dim6[XYZ]);
        give_value(recv_vec[F_T], zero, lat_3dim6[XYZ]);
      }
      // mpi_wilson_dslash
      {
        // clean
        wilson_dslash_clear_dest<<<gridDim, blockDim>>>(
            dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z]);
        // send x
        wilson_dslash_x_send<<<gridDim, blockDim>>>(
            gauge, dslash_in, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, send_vec[B_X], send_vec[F_X]);
        if (grid_1dim[X] != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move[B], grid_index_1dim[X], grid_1dim[X]);
          move_forward(move[F], grid_index_1dim[X], grid_1dim[X]);
          move[B] =
              node_rank + move[B] * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];
          move[F] =
              node_rank + move[F] * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];
          MPI_Irecv(recv_vec[B_X], lat_3dim12[YZT], MPI_DOUBLE, move[B], 1,
                    MPI_COMM_WORLD, &recv_request[B_X]);
          MPI_Irecv(recv_vec[F_X], lat_3dim12[YZT], MPI_DOUBLE, move[F], 0,
                    MPI_COMM_WORLD, &recv_request[F_X]);
          MPI_Isend(send_vec[B_X], lat_3dim12[YZT], MPI_DOUBLE, move[B], 0,
                    MPI_COMM_WORLD, &send_request[B_X]);
          MPI_Isend(send_vec[F_X], lat_3dim12[YZT], MPI_DOUBLE, move[F], 1,
                    MPI_COMM_WORLD, &send_request[F_T]);
        }
        // send y
        wilson_dslash_y_send<<<gridDim, blockDim>>>(
            gauge, dslash_in, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, send_vec[B_Y], send_vec[F_Y]);
        if (grid_1dim[Y] != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move[B], grid_index_1dim[Y], grid_1dim[Y]);
          move_forward(move[F], grid_index_1dim[Y], grid_1dim[Y]);
          move[B] = node_rank + move[B] * grid_1dim[Z] * grid_1dim[T];
          move[F] = node_rank + move[F] * grid_1dim[Z] * grid_1dim[T];
          MPI_Irecv(recv_vec[B_Y], lat_3dim12[XZT], MPI_DOUBLE, move[B], 3,
                    MPI_COMM_WORLD, &recv_request[B_Y]);
          MPI_Irecv(recv_vec[F_Y], lat_3dim12[XZT], MPI_DOUBLE, move[F], 2,
                    MPI_COMM_WORLD, &recv_request[F_Y]);
          MPI_Isend(send_vec[B_Y], lat_3dim12[XZT], MPI_DOUBLE, move[B], 2,
                    MPI_COMM_WORLD, &send_request[B_Y]);
          MPI_Isend(send_vec[F_Y], lat_3dim12[XZT], MPI_DOUBLE, move[F], 3,
                    MPI_COMM_WORLD, &send_request[F_Y]);
        }
        // send z
        wilson_dslash_z_send<<<gridDim, blockDim>>>(
            gauge, dslash_in, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, send_vec[B_Z], send_vec[F_Z]);
        if (grid_1dim[Z] != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move[B], grid_index_1dim[Z], grid_1dim[Z]);
          move_forward(move[F], grid_index_1dim[Z], grid_1dim[Z]);
          move[B] = node_rank + move[B] * grid_1dim[T];
          move[F] = node_rank + move[F] * grid_1dim[T];
          MPI_Irecv(recv_vec[B_Z], lat_3dim12[XYT], MPI_DOUBLE, move[B], 5,
                    MPI_COMM_WORLD, &recv_request[B_Z]);
          MPI_Irecv(recv_vec[F_Z], lat_3dim12[XYT], MPI_DOUBLE, move[F], 4,
                    MPI_COMM_WORLD, &recv_request[F_Z]);
          MPI_Isend(send_vec[B_Z], lat_3dim12[XYT], MPI_DOUBLE, move[B], 4,
                    MPI_COMM_WORLD, &send_request[B_Z]);
          MPI_Isend(send_vec[F_Z], lat_3dim12[XYT], MPI_DOUBLE, move[F], 5,
                    MPI_COMM_WORLD, &send_request[F_Z]);
        }
        // send t
        wilson_dslash_t_send<<<gridDim, blockDim>>>(
            gauge, dslash_in, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, send_vec[B_T], send_vec[F_T]);
        if (grid_1dim[T] != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move[B], grid_index_1dim[T], grid_1dim[T]);
          move_forward(move[F], grid_index_1dim[T], grid_1dim[T]);
          move[B] = node_rank + move[B];
          move[F] = node_rank + move[F];
          MPI_Irecv(recv_vec[B_T], lat_3dim12[XYZ], MPI_DOUBLE, move[B], 7,
                    MPI_COMM_WORLD, &recv_request[B_T]);
          MPI_Irecv(recv_vec[F_T], lat_3dim12[XYZ], MPI_DOUBLE, move[F], 6,
                    MPI_COMM_WORLD, &recv_request[F_T]);
          MPI_Isend(send_vec[B_T], lat_3dim12[XYZ], MPI_DOUBLE, move[B], 6,
                    MPI_COMM_WORLD, &send_request[B_T]);
          MPI_Isend(send_vec[F_T], lat_3dim12[XYZ], MPI_DOUBLE, move[F], 7,
                    MPI_COMM_WORLD, &send_request[F_T]);
        }
        // recv x
        if (grid_1dim[X] != 1) {
          MPI_Wait(&recv_request[B_X], MPI_STATUS_IGNORE);
          MPI_Wait(&recv_request[F_X], MPI_STATUS_IGNORE);
          wilson_dslash_x_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, recv_vec[B_X], recv_vec[F_X]);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_x_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, send_vec[F_X], send_vec[B_X]);
        }
        // recv y
        if (grid_1dim[Y] != 1) {
          MPI_Wait(&recv_request[B_Y], MPI_STATUS_IGNORE);
          MPI_Wait(&recv_request[F_Y], MPI_STATUS_IGNORE);
          wilson_dslash_y_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, recv_vec[B_Y], recv_vec[F_Y]);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_y_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, send_vec[F_Y], send_vec[B_Y]);
        }
        // recv z
        if (grid_1dim[Z] != 1) {
          MPI_Wait(&recv_request[B_Z], MPI_STATUS_IGNORE);
          MPI_Wait(&recv_request[F_Z], MPI_STATUS_IGNORE);
          wilson_dslash_z_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, recv_vec[B_Z], recv_vec[F_Z]);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_z_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, send_vec[F_Z], send_vec[B_Z]);
        }
        // recv t
        if (grid_1dim[T] != 1) {
          MPI_Wait(&recv_request[B_T], MPI_STATUS_IGNORE);
          MPI_Wait(&recv_request[F_T], MPI_STATUS_IGNORE);
          wilson_dslash_t_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, recv_vec[B_T], recv_vec[F_T]);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_t_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, send_vec[F_T], send_vec[B_T]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
      // kappa
      {
        for (int i = 0; i < lat_4dim12; i++) {
          dslash_out[i] = dslash_in[i] - dslash_out[i] * Kappa;
        }
      }
#else
#ifdef TEST_MPI_WILSON_CG_USE_WILSON_DSLASH
      // wilson_dslash
      wilson_dslash<<<gridDim, blockDim>>>(gauge, dslash_in, dslash_out,
                                           lat_1dim[X], lat_1dim[Y],
                                           lat_1dim[Z], lat_1dim[T], parity);
      // kappa
      {
        for (int i = 0; i < lat_4dim12; i++) {
          dslash_out[i] = dslash_in[i] - dslash_out[i] * Kappa;
        }
      }
#else
      for (int i = 0; i < lat_4dim12; i++) {
        dslash_out[i] = dslash_in[i] * 2 + one;
      }
#endif
#endif
      {
        local_result = zero;
        for (int i = 0; i < lat_4dim12; i++) {
          local_result += r_tilde[i].conj() * v[i];
        }
        MPI_Allreduce(&local_result, &tmp, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
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
#ifndef TEST_MPI_WILSON_CG
      // clear vecs for mpi_wilson_dslash
      {
        give_value(send_vec[B_X], zero, lat_3dim6[YZT]);
        give_value(send_vec[F_X], zero, lat_3dim6[YZT]);
        give_value(send_vec[B_Y], zero, lat_3dim6[XZT]);
        give_value(send_vec[F_Y], zero, lat_3dim6[XZT]);
        give_value(send_vec[B_Z], zero, lat_3dim6[XYT]);
        give_value(send_vec[F_Z], zero, lat_3dim6[XYT]);
        give_value(send_vec[B_T], zero, lat_3dim6[XYZ]);
        give_value(send_vec[F_T], zero, lat_3dim6[XYZ]);
        give_value(recv_vec[B_X], zero, lat_3dim6[YZT]);
        give_value(recv_vec[F_X], zero, lat_3dim6[YZT]);
        give_value(recv_vec[B_Y], zero, lat_3dim6[XZT]);
        give_value(recv_vec[F_Y], zero, lat_3dim6[XZT]);
        give_value(recv_vec[B_Z], zero, lat_3dim6[XYT]);
        give_value(recv_vec[F_Z], zero, lat_3dim6[XYT]);
        give_value(recv_vec[B_T], zero, lat_3dim6[XYZ]);
        give_value(recv_vec[F_T], zero, lat_3dim6[XYZ]);
      }
      // mpi_wilson_dslash
      {
        // clean
        wilson_dslash_clear_dest<<<gridDim, blockDim>>>(
            dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z]);
        // send x
        wilson_dslash_x_send<<<gridDim, blockDim>>>(
            gauge, dslash_in, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, send_vec[B_X], send_vec[F_X]);
        if (grid_1dim[X] != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move[B], grid_index_1dim[X], grid_1dim[X]);
          move_forward(move[F], grid_index_1dim[X], grid_1dim[X]);
          move[B] =
              node_rank + move[B] * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];
          move[F] =
              node_rank + move[F] * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];
          MPI_Irecv(recv_vec[B_X], lat_3dim12[YZT], MPI_DOUBLE, move[B], 1,
                    MPI_COMM_WORLD, &recv_request[B_X]);
          MPI_Irecv(recv_vec[F_X], lat_3dim12[YZT], MPI_DOUBLE, move[F], 0,
                    MPI_COMM_WORLD, &recv_request[F_X]);
          MPI_Isend(send_vec[B_X], lat_3dim12[YZT], MPI_DOUBLE, move[B], 0,
                    MPI_COMM_WORLD, &send_request[B_X]);
          MPI_Isend(send_vec[F_X], lat_3dim12[YZT], MPI_DOUBLE, move[F], 1,
                    MPI_COMM_WORLD, &send_request[F_T]);
        }
        // send y
        wilson_dslash_y_send<<<gridDim, blockDim>>>(
            gauge, dslash_in, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, send_vec[B_Y], send_vec[F_Y]);
        if (grid_1dim[Y] != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move[B], grid_index_1dim[Y], grid_1dim[Y]);
          move_forward(move[F], grid_index_1dim[Y], grid_1dim[Y]);
          move[B] = node_rank + move[B] * grid_1dim[Z] * grid_1dim[T];
          move[F] = node_rank + move[F] * grid_1dim[Z] * grid_1dim[T];
          MPI_Irecv(recv_vec[B_Y], lat_3dim12[XZT], MPI_DOUBLE, move[B], 3,
                    MPI_COMM_WORLD, &recv_request[B_Y]);
          MPI_Irecv(recv_vec[F_Y], lat_3dim12[XZT], MPI_DOUBLE, move[F], 2,
                    MPI_COMM_WORLD, &recv_request[F_Y]);
          MPI_Isend(send_vec[B_Y], lat_3dim12[XZT], MPI_DOUBLE, move[B], 2,
                    MPI_COMM_WORLD, &send_request[B_Y]);
          MPI_Isend(send_vec[F_Y], lat_3dim12[XZT], MPI_DOUBLE, move[F], 3,
                    MPI_COMM_WORLD, &send_request[F_Y]);
        }
        // send z
        wilson_dslash_z_send<<<gridDim, blockDim>>>(
            gauge, dslash_in, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, send_vec[B_Z], send_vec[F_Z]);
        if (grid_1dim[Z] != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move[B], grid_index_1dim[Z], grid_1dim[Z]);
          move_forward(move[F], grid_index_1dim[Z], grid_1dim[Z]);
          move[B] = node_rank + move[B] * grid_1dim[T];
          move[F] = node_rank + move[F] * grid_1dim[T];
          MPI_Irecv(recv_vec[B_Z], lat_3dim12[XYT], MPI_DOUBLE, move[B], 5,
                    MPI_COMM_WORLD, &recv_request[B_Z]);
          MPI_Irecv(recv_vec[F_Z], lat_3dim12[XYT], MPI_DOUBLE, move[F], 4,
                    MPI_COMM_WORLD, &recv_request[F_Z]);
          MPI_Isend(send_vec[B_Z], lat_3dim12[XYT], MPI_DOUBLE, move[B], 4,
                    MPI_COMM_WORLD, &send_request[B_Z]);
          MPI_Isend(send_vec[F_Z], lat_3dim12[XYT], MPI_DOUBLE, move[F], 5,
                    MPI_COMM_WORLD, &send_request[F_Z]);
        }
        // send t
        wilson_dslash_t_send<<<gridDim, blockDim>>>(
            gauge, dslash_in, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
            lat_1dim[T], parity, send_vec[B_T], send_vec[F_T]);
        if (grid_1dim[T] != 1) {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move[B], grid_index_1dim[T], grid_1dim[T]);
          move_forward(move[F], grid_index_1dim[T], grid_1dim[T]);
          move[B] = node_rank + move[B];
          move[F] = node_rank + move[F];
          MPI_Irecv(recv_vec[B_T], lat_3dim12[XYZ], MPI_DOUBLE, move[B], 7,
                    MPI_COMM_WORLD, &recv_request[B_T]);
          MPI_Irecv(recv_vec[F_T], lat_3dim12[XYZ], MPI_DOUBLE, move[F], 6,
                    MPI_COMM_WORLD, &recv_request[F_T]);
          MPI_Isend(send_vec[B_T], lat_3dim12[XYZ], MPI_DOUBLE, move[B], 6,
                    MPI_COMM_WORLD, &send_request[B_T]);
          MPI_Isend(send_vec[F_T], lat_3dim12[XYZ], MPI_DOUBLE, move[F], 7,
                    MPI_COMM_WORLD, &send_request[F_T]);
        }
        // recv x
        if (grid_1dim[X] != 1) {
          MPI_Wait(&recv_request[B_X], MPI_STATUS_IGNORE);
          MPI_Wait(&recv_request[F_X], MPI_STATUS_IGNORE);
          wilson_dslash_x_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, recv_vec[B_X], recv_vec[F_X]);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_x_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, send_vec[F_X], send_vec[B_X]);
        }
        // recv y
        if (grid_1dim[Y] != 1) {
          MPI_Wait(&recv_request[B_Y], MPI_STATUS_IGNORE);
          MPI_Wait(&recv_request[F_Y], MPI_STATUS_IGNORE);
          wilson_dslash_y_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, recv_vec[B_Y], recv_vec[F_Y]);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_y_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, send_vec[F_Y], send_vec[B_Y]);
        }
        // recv z
        if (grid_1dim[Z] != 1) {
          MPI_Wait(&recv_request[B_Z], MPI_STATUS_IGNORE);
          MPI_Wait(&recv_request[F_Z], MPI_STATUS_IGNORE);
          wilson_dslash_z_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, recv_vec[B_Z], recv_vec[F_Z]);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_z_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, send_vec[F_Z], send_vec[B_Z]);
        }
        // recv t
        if (grid_1dim[T] != 1) {
          MPI_Wait(&recv_request[B_T], MPI_STATUS_IGNORE);
          MPI_Wait(&recv_request[F_T], MPI_STATUS_IGNORE);
          wilson_dslash_t_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, recv_vec[B_T], recv_vec[F_T]);
        } else {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_t_recv<<<gridDim, blockDim>>>(
              gauge, dslash_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
              lat_1dim[T], parity, send_vec[F_T], send_vec[B_T]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
      // kappa
      {
        for (int i = 0; i < lat_4dim12; i++) {
          dslash_out[i] = dslash_in[i] - dslash_out[i] * Kappa;
        }
      }
#else
#ifdef TEST_MPI_WILSON_CG_USE_WILSON_DSLASH
      // wilson_dslash
      wilson_dslash<<<gridDim, blockDim>>>(gauge, dslash_in, dslash_out,
                                           lat_1dim[X], lat_1dim[Y],
                                           lat_1dim[Z], lat_1dim[T], parity);
      // kappa
      {
        for (int i = 0; i < lat_4dim12; i++) {
          dslash_out[i] = dslash_in[i] - dslash_out[i] * Kappa;
        }
      }
#else
      for (int i = 0; i < lat_4dim12; i++) {
        dslash_out[i] = dslash_in[i] * 2 + one;
      }
#endif
#endif
      {
        local_result = zero;
        for (int i = 0; i < lat_4dim12; i++) {
          local_result += t[i].conj() * s[i];
        }
        MPI_Allreduce(&local_result, &tmp0, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      {
        local_result = zero;
        for (int i = 0; i < lat_4dim12; i++) {
          local_result += t[i].conj() * t[i];
        }
        MPI_Allreduce(&local_result, &tmp1, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
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
      {
        local_result = zero;
        for (int i = 0; i < lat_4dim12; i++) {
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
    // free
    {
      checkCudaErrors(cudaFree(send_vec[B_X]));
      checkCudaErrors(cudaFree(send_vec[F_X]));
      checkCudaErrors(cudaFree(send_vec[B_Y]));
      checkCudaErrors(cudaFree(send_vec[F_Y]));
      checkCudaErrors(cudaFree(send_vec[B_Z]));
      checkCudaErrors(cudaFree(send_vec[F_Z]));
      checkCudaErrors(cudaFree(send_vec[B_T]));
      checkCudaErrors(cudaFree(send_vec[F_T]));
      checkCudaErrors(cudaFree(recv_vec[B_X]));
      checkCudaErrors(cudaFree(recv_vec[F_X]));
      checkCudaErrors(cudaFree(recv_vec[B_Y]));
      checkCudaErrors(cudaFree(recv_vec[F_Y]));
      checkCudaErrors(cudaFree(recv_vec[B_Z]));
      checkCudaErrors(cudaFree(recv_vec[F_Z]));
      checkCudaErrors(cudaFree(recv_vec[B_T]));
      checkCudaErrors(cudaFree(recv_vec[F_T]));
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