#include <iostream>
#pragma optimize(5)
#include "../../include/qcu.h"

// #define DEBUG_MPI_WILSON_CG
// #define TEST_MPI_WILSON_CG
// #define TEST_MPI_WILSON_CG_USE_WILSON_DSLASH

#ifdef MPI_WILSON_CG
void mpiCgQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param,
              int parity, QcuParam *grid)
{
  int lat_1dim[DIM];
  int lat_3dim6[DIM];
  int lat_3dim12[DIM];
  int lat_4dim12;
  give_dims(param, lat_1dim, lat_3dim6, lat_3dim12, lat_4dim12);
  cudaError_t err;
  dim3 gridDim(lat_1dim[X] * lat_1dim[Y] * lat_1dim[Z] * lat_1dim[T] / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  {
    // mpi wilson cg
    int node_rank, move_b, move_f;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_rank);
    int grid_1dim[DIM];
    int grid_index_1dim[DIM];
    give_grid(grid, node_rank, grid_1dim, grid_index_1dim);
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
    cudaMallocManaged(&b_x_send_vec, lat_3dim6[YZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&f_x_send_vec, lat_3dim6[YZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&b_y_send_vec, lat_3dim6[XZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&f_y_send_vec, lat_3dim6[XZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&b_z_send_vec, lat_3dim6[XYT] * sizeof(LatticeComplex));
    cudaMallocManaged(&f_z_send_vec, lat_3dim6[XYT] * sizeof(LatticeComplex));
    cudaMallocManaged(&b_t_send_vec, lat_3dim6[XYZ] * sizeof(LatticeComplex));
    cudaMallocManaged(&f_t_send_vec, lat_3dim6[XYZ] * sizeof(LatticeComplex));
    cudaMallocManaged(&b_x_recv_vec, lat_3dim6[YZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&f_x_recv_vec, lat_3dim6[YZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&b_y_recv_vec, lat_3dim6[XZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&f_y_recv_vec, lat_3dim6[XZT] * sizeof(LatticeComplex));
    cudaMallocManaged(&b_z_recv_vec, lat_3dim6[XYT] * sizeof(LatticeComplex));
    cudaMallocManaged(&f_z_recv_vec, lat_3dim6[XYT] * sizeof(LatticeComplex));
    cudaMallocManaged(&b_t_recv_vec, lat_3dim6[XYZ] * sizeof(LatticeComplex));
    cudaMallocManaged(&f_t_recv_vec, lat_3dim6[XYZ] * sizeof(LatticeComplex));
    LatticeComplex *cg_in, *cg_out, *x, *b, *r, *r_tilde, *p, *v, *s, *t;
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
    cg_in = x;
    cg_out = r;
#ifndef TEST_MPI_WILSON_CG
    // clear vecs for mpi_wilson_dslash
    {
      give_value(b_x_send_vec, zero, lat_3dim6[YZT]);
      give_value(f_x_send_vec, zero, lat_3dim6[YZT]);
      give_value(b_y_send_vec, zero, lat_3dim6[XZT]);
      give_value(f_y_send_vec, zero, lat_3dim6[XZT]);
      give_value(b_z_send_vec, zero, lat_3dim6[XYT]);
      give_value(f_z_send_vec, zero, lat_3dim6[XYT]);
      give_value(b_t_send_vec, zero, lat_3dim6[XYZ]);
      give_value(f_t_send_vec, zero, lat_3dim6[XYZ]);
      give_value(b_x_recv_vec, zero, lat_3dim6[YZT]);
      give_value(f_x_recv_vec, zero, lat_3dim6[YZT]);
      give_value(b_y_recv_vec, zero, lat_3dim6[XZT]);
      give_value(f_y_recv_vec, zero, lat_3dim6[XZT]);
      give_value(b_z_recv_vec, zero, lat_3dim6[XYT]);
      give_value(f_z_recv_vec, zero, lat_3dim6[XYT]);
      give_value(b_t_recv_vec, zero, lat_3dim6[XYZ]);
      give_value(f_t_recv_vec, zero, lat_3dim6[XYZ]);
    }
    // mpi_wilson_dslash
    {
      // clean
      wilson_dslash_clear_dest<<<gridDim, blockDim>>>(cg_out, lat_1dim[X], lat_1dim[Y],
                                                      lat_1dim[Z]);
      // send x
      wilson_dslash_x_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_1dim[X],
                                                  lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity,
                                                  b_x_send_vec, f_x_send_vec);
      if (grid_1dim[X] != 1)
      {
        checkCudaErrors(cudaDeviceSynchronize());
        move_backward(move_b, grid_index_1dim[X], grid_1dim[X]);
        move_forward(move_f, grid_index_1dim[X], grid_1dim[X]);
        move_b = node_rank + move_b * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];
        move_f = node_rank + move_f * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];
        MPI_Irecv(b_x_recv_vec, lat_3dim12[YZT], MPI_DOUBLE, move_b, 1,
                  MPI_COMM_WORLD, &b_x_recv_request);
        MPI_Irecv(f_x_recv_vec, lat_3dim12[YZT], MPI_DOUBLE, move_f, 0,
                  MPI_COMM_WORLD, &f_x_recv_request);
        MPI_Isend(b_x_send_vec, lat_3dim12[YZT], MPI_DOUBLE, move_b, 0,
                  MPI_COMM_WORLD, &b_x_send_request);
        MPI_Isend(f_x_send_vec, lat_3dim12[YZT], MPI_DOUBLE, move_f, 1,
                  MPI_COMM_WORLD, &f_x_send_request);
      }
      // send y
      wilson_dslash_y_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_1dim[X],
                                                  lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity,
                                                  b_y_send_vec, f_y_send_vec);
      if (grid_1dim[Y] != 1)
      {
        checkCudaErrors(cudaDeviceSynchronize());
        move_backward(move_b, grid_index_1dim[Y], grid_1dim[Y]);
        move_forward(move_f, grid_index_1dim[Y], grid_1dim[Y]);
        move_b = node_rank + move_b * grid_1dim[Z] * grid_1dim[T];
        move_f = node_rank + move_f * grid_1dim[Z] * grid_1dim[T];
        MPI_Irecv(b_y_recv_vec, lat_3dim12[XZT], MPI_DOUBLE, move_b, 3,
                  MPI_COMM_WORLD, &b_y_recv_request);
        MPI_Irecv(f_y_recv_vec, lat_3dim12[XZT], MPI_DOUBLE, move_f, 2,
                  MPI_COMM_WORLD, &f_y_recv_request);
        MPI_Isend(b_y_send_vec, lat_3dim12[XZT], MPI_DOUBLE, move_b, 2,
                  MPI_COMM_WORLD, &b_y_send_request);
        MPI_Isend(f_y_send_vec, lat_3dim12[XZT], MPI_DOUBLE, move_f, 3,
                  MPI_COMM_WORLD, &f_y_send_request);
      }
      // send z
      wilson_dslash_z_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_1dim[X],
                                                  lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity,
                                                  b_z_send_vec, f_z_send_vec);
      if (grid_1dim[Z] != 1)
      {
        checkCudaErrors(cudaDeviceSynchronize());
        move_backward(move_b, grid_index_1dim[Z], grid_1dim[Z]);
        move_forward(move_f, grid_index_1dim[Z], grid_1dim[Z]);
        move_b = node_rank + move_b * grid_1dim[T];
        move_f = node_rank + move_f * grid_1dim[T];
        MPI_Irecv(b_z_recv_vec, lat_3dim12[XYT], MPI_DOUBLE, move_b, 5,
                  MPI_COMM_WORLD, &b_z_recv_request);
        MPI_Irecv(f_z_recv_vec, lat_3dim12[XYT], MPI_DOUBLE, move_f, 4,
                  MPI_COMM_WORLD, &f_z_recv_request);
        MPI_Isend(b_z_send_vec, lat_3dim12[XYT], MPI_DOUBLE, move_b, 4,
                  MPI_COMM_WORLD, &b_z_send_request);
        MPI_Isend(f_z_send_vec, lat_3dim12[XYT], MPI_DOUBLE, move_f, 5,
                  MPI_COMM_WORLD, &f_z_send_request);
      }
      // send t
      wilson_dslash_t_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_1dim[X],
                                                  lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity,
                                                  b_t_send_vec, f_t_send_vec);
      if (grid_1dim[T] != 1)
      {
        checkCudaErrors(cudaDeviceSynchronize());
        move_backward(move_b, grid_index_1dim[T], grid_1dim[T]);
        move_forward(move_f, grid_index_1dim[T], grid_1dim[T]);
        move_b = node_rank + move_b;
        move_f = node_rank + move_f;
        MPI_Irecv(b_t_recv_vec, lat_3dim12[XYZ], MPI_DOUBLE, move_b, 7,
                  MPI_COMM_WORLD, &b_t_recv_request);
        MPI_Irecv(f_t_recv_vec, lat_3dim12[XYZ], MPI_DOUBLE, move_f, 6,
                  MPI_COMM_WORLD, &f_t_recv_request);
        MPI_Isend(b_t_send_vec, lat_3dim12[XYZ], MPI_DOUBLE, move_b, 6,
                  MPI_COMM_WORLD, &b_t_send_request);
        MPI_Isend(f_t_send_vec, lat_3dim12[XYZ], MPI_DOUBLE, move_f, 7,
                  MPI_COMM_WORLD, &f_t_send_request);
      }
      // recv x
      if (grid_1dim[X] != 1)
      {
        MPI_Wait(&b_x_recv_request, MPI_STATUS_IGNORE);
        MPI_Wait(&f_x_recv_request, MPI_STATUS_IGNORE);
        wilson_dslash_x_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_1dim[X], lat_1dim[Y],
                                                    lat_1dim[Z], lat_1dim[T], parity,
                                                    b_x_recv_vec, f_x_recv_vec);
      }
      else
      {
        checkCudaErrors(cudaDeviceSynchronize());
        wilson_dslash_x_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_1dim[X], lat_1dim[Y],
                                                    lat_1dim[Z], lat_1dim[T], parity,
                                                    f_x_send_vec, b_x_send_vec);
      }
      // recv y
      if (grid_1dim[Y] != 1)
      {
        MPI_Wait(&b_y_recv_request, MPI_STATUS_IGNORE);
        MPI_Wait(&f_y_recv_request, MPI_STATUS_IGNORE);
        wilson_dslash_y_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_1dim[X], lat_1dim[Y],
                                                    lat_1dim[Z], lat_1dim[T], parity,
                                                    b_y_recv_vec, f_y_recv_vec);
      }
      else
      {
        checkCudaErrors(cudaDeviceSynchronize());
        wilson_dslash_y_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_1dim[X], lat_1dim[Y],
                                                    lat_1dim[Z], lat_1dim[T], parity,
                                                    f_y_send_vec, b_y_send_vec);
      }
      // recv z
      if (grid_1dim[Z] != 1)
      {
        MPI_Wait(&b_z_recv_request, MPI_STATUS_IGNORE);
        MPI_Wait(&f_z_recv_request, MPI_STATUS_IGNORE);
        wilson_dslash_z_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_1dim[X], lat_1dim[Y],
                                                    lat_1dim[Z], lat_1dim[T], parity,
                                                    b_z_recv_vec, f_z_recv_vec);
      }
      else
      {
        checkCudaErrors(cudaDeviceSynchronize());
        wilson_dslash_z_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_1dim[X], lat_1dim[Y],
                                                    lat_1dim[Z], lat_1dim[T], parity,
                                                    f_z_send_vec, b_z_send_vec);
      }
      // recv t
      if (grid_1dim[T] != 1)
      {
        MPI_Wait(&b_t_recv_request, MPI_STATUS_IGNORE);
        MPI_Wait(&f_t_recv_request, MPI_STATUS_IGNORE);
        wilson_dslash_t_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_1dim[X], lat_1dim[Y],
                                                    lat_1dim[Z], lat_1dim[T], parity,
                                                    b_t_recv_vec, f_t_recv_vec);
      }
      else
      {
        checkCudaErrors(cudaDeviceSynchronize());
        wilson_dslash_t_recv<<<gridDim, blockDim>>>(gauge, cg_out, lat_1dim[X], lat_1dim[Y],
                                                    lat_1dim[Z], lat_1dim[T], parity,
                                                    f_t_send_vec, b_t_send_vec);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    // kappa
    {
      for (int i = 0; i < lat_4dim12; i++)
      {
        cg_out[i] = cg_in[i] - cg_out[i] * Kappa;
      }
    }
#else
#ifdef TEST_MPI_WILSON_CG_USE_WILSON_DSLASH
    // wilson_dslash
    wilson_dslash<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_1dim[X], lat_1dim[Y],
                                         lat_1dim[Z], lat_1dim[T], parity);
    // kappa
    {
      for (int i = 0; i < lat_4dim12; i++)
      {
        cg_out[i] = cg_in[i] - cg_out[i] * Kappa;
      }
    }
#else
    for (int i = 0; i < lat_4dim12; i++)
    {
      cg_out[i] = cg_in[i] * 2 + one;
    }
#endif
#endif
    for (int i = 0; i < lat_4dim12; i++)
    {
      r[i] = b[i] - r[i];
      r_tilde[i] = r[i];
    }
    for (int loop = 0; loop < MAX_ITER; loop++)
    {
      {
        local_result = zero;
        for (int i = 0; i < lat_4dim12; i++)
        {
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
      for (int i = 0; i < lat_4dim12; i++)
      {
        p[i] = r[i] + (p[i] - v[i] * omega) * beta;
      }
      // v = A * p;
      cg_in = p;
      cg_out = v;
#ifndef TEST_MPI_WILSON_CG
      // clear vecs for mpi_wilson_dslash
      {
        give_value(b_x_send_vec, zero, lat_3dim6[YZT]);
        give_value(f_x_send_vec, zero, lat_3dim6[YZT]);
        give_value(b_y_send_vec, zero, lat_3dim6[XZT]);
        give_value(f_y_send_vec, zero, lat_3dim6[XZT]);
        give_value(b_z_send_vec, zero, lat_3dim6[XYT]);
        give_value(f_z_send_vec, zero, lat_3dim6[XYT]);
        give_value(b_t_send_vec, zero, lat_3dim6[XYZ]);
        give_value(f_t_send_vec, zero, lat_3dim6[XYZ]);
        give_value(b_x_recv_vec, zero, lat_3dim6[YZT]);
        give_value(f_x_recv_vec, zero, lat_3dim6[YZT]);
        give_value(b_y_recv_vec, zero, lat_3dim6[XZT]);
        give_value(f_y_recv_vec, zero, lat_3dim6[XZT]);
        give_value(b_z_recv_vec, zero, lat_3dim6[XYT]);
        give_value(f_z_recv_vec, zero, lat_3dim6[XYT]);
        give_value(b_t_recv_vec, zero, lat_3dim6[XYZ]);
        give_value(f_t_recv_vec, zero, lat_3dim6[XYZ]);
      }
      // mpi_wilson_dslash
      {
        // clean
        wilson_dslash_clear_dest<<<gridDim, blockDim>>>(cg_out, lat_1dim[X], lat_1dim[Y],
                                                        lat_1dim[Z]);
        // send x
        wilson_dslash_x_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_1dim[X],
                                                    lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity,
                                                    b_x_send_vec, f_x_send_vec);
        if (grid_1dim[X] != 1)
        {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_1dim[X], grid_1dim[X]);
          move_forward(move_f, grid_index_1dim[X], grid_1dim[X]);
          move_b = node_rank + move_b * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];
          move_f = node_rank + move_f * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];
          MPI_Irecv(b_x_recv_vec, lat_3dim12[YZT], MPI_DOUBLE, move_b, 1,
                    MPI_COMM_WORLD, &b_x_recv_request);
          MPI_Irecv(f_x_recv_vec, lat_3dim12[YZT], MPI_DOUBLE, move_f, 0,
                    MPI_COMM_WORLD, &f_x_recv_request);
          MPI_Isend(b_x_send_vec, lat_3dim12[YZT], MPI_DOUBLE, move_b, 0,
                    MPI_COMM_WORLD, &b_x_send_request);
          MPI_Isend(f_x_send_vec, lat_3dim12[YZT], MPI_DOUBLE, move_f, 1,
                    MPI_COMM_WORLD, &f_x_send_request);
        }
        // send y
        wilson_dslash_y_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_1dim[X],
                                                    lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity,
                                                    b_y_send_vec, f_y_send_vec);
        if (grid_1dim[Y] != 1)
        {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_1dim[Y], grid_1dim[Y]);
          move_forward(move_f, grid_index_1dim[Y], grid_1dim[Y]);
          move_b = node_rank + move_b * grid_1dim[Z] * grid_1dim[T];
          move_f = node_rank + move_f * grid_1dim[Z] * grid_1dim[T];
          MPI_Irecv(b_y_recv_vec, lat_3dim12[XZT], MPI_DOUBLE, move_b, 3,
                    MPI_COMM_WORLD, &b_y_recv_request);
          MPI_Irecv(f_y_recv_vec, lat_3dim12[XZT], MPI_DOUBLE, move_f, 2,
                    MPI_COMM_WORLD, &f_y_recv_request);
          MPI_Isend(b_y_send_vec, lat_3dim12[XZT], MPI_DOUBLE, move_b, 2,
                    MPI_COMM_WORLD, &b_y_send_request);
          MPI_Isend(f_y_send_vec, lat_3dim12[XZT], MPI_DOUBLE, move_f, 3,
                    MPI_COMM_WORLD, &f_y_send_request);
        }
        // send z
        wilson_dslash_z_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_1dim[X],
                                                    lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity,
                                                    b_z_send_vec, f_z_send_vec);
        if (grid_1dim[Z] != 1)
        {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_1dim[Z], grid_1dim[Z]);
          move_forward(move_f, grid_index_1dim[Z], grid_1dim[Z]);
          move_b = node_rank + move_b * grid_1dim[T];
          move_f = node_rank + move_f * grid_1dim[T];
          MPI_Irecv(b_z_recv_vec, lat_3dim12[XYT], MPI_DOUBLE, move_b, 5,
                    MPI_COMM_WORLD, &b_z_recv_request);
          MPI_Irecv(f_z_recv_vec, lat_3dim12[XYT], MPI_DOUBLE, move_f, 4,
                    MPI_COMM_WORLD, &f_z_recv_request);
          MPI_Isend(b_z_send_vec, lat_3dim12[XYT], MPI_DOUBLE, move_b, 4,
                    MPI_COMM_WORLD, &b_z_send_request);
          MPI_Isend(f_z_send_vec, lat_3dim12[XYT], MPI_DOUBLE, move_f, 5,
                    MPI_COMM_WORLD, &f_z_send_request);
        }
        // send t
        wilson_dslash_t_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_1dim[X],
                                                    lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity,
                                                    b_t_send_vec, f_t_send_vec);
        if (grid_1dim[T] != 1)
        {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_1dim[T], grid_1dim[T]);
          move_forward(move_f, grid_index_1dim[T], grid_1dim[T]);
          move_b = node_rank + move_b;
          move_f = node_rank + move_f;
          MPI_Irecv(b_t_recv_vec, lat_3dim12[XYZ], MPI_DOUBLE, move_b, 7,
                    MPI_COMM_WORLD, &b_t_recv_request);
          MPI_Irecv(f_t_recv_vec, lat_3dim12[XYZ], MPI_DOUBLE, move_f, 6,
                    MPI_COMM_WORLD, &f_t_recv_request);
          MPI_Isend(b_t_send_vec, lat_3dim12[XYZ], MPI_DOUBLE, move_b, 6,
                    MPI_COMM_WORLD, &b_t_send_request);
          MPI_Isend(f_t_send_vec, lat_3dim12[XYZ], MPI_DOUBLE, move_f, 7,
                    MPI_COMM_WORLD, &f_t_send_request);
        }
        // recv x
        if (grid_1dim[X] != 1)
        {
          MPI_Wait(&b_x_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_x_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_x_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, b_x_recv_vec,
              f_x_recv_vec);
        }
        else
        {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_x_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, f_x_send_vec,
              b_x_send_vec);
        }
        // recv y
        if (grid_1dim[Y] != 1)
        {
          MPI_Wait(&b_y_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_y_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_y_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, b_y_recv_vec,
              f_y_recv_vec);
        }
        else
        {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_y_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, f_y_send_vec,
              b_y_send_vec);
        }
        // recv z
        if (grid_1dim[Z] != 1)
        {
          MPI_Wait(&b_z_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_z_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_z_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, b_z_recv_vec,
              f_z_recv_vec);
        }
        else
        {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_z_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, f_z_send_vec,
              b_z_send_vec);
        }
        // recv t
        if (grid_1dim[T] != 1)
        {
          MPI_Wait(&b_t_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_t_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_t_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, b_t_recv_vec,
              f_t_recv_vec);
        }
        else
        {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_t_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, f_t_send_vec,
              b_t_send_vec);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
      // kappa
      {
        for (int i = 0; i < lat_4dim12; i++)
        {
          cg_out[i] = cg_in[i] - cg_out[i] * Kappa;
        }
      }
#else
#ifdef TEST_MPI_WILSON_CG_USE_WILSON_DSLASH
      // wilson_dslash
      wilson_dslash<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_1dim[X], lat_1dim[Y],
                                           lat_1dim[Z], lat_1dim[T], parity);
      // kappa
      {
        for (int i = 0; i < lat_4dim12; i++)
        {
          cg_out[i] = cg_in[i] - cg_out[i] * Kappa;
        }
      }
#else
      for (int i = 0; i < lat_4dim12; i++)
      {
        cg_out[i] = cg_in[i] * 2 + one;
      }
#endif
#endif
      {
        local_result = zero;
        for (int i = 0; i < lat_4dim12; i++)
        {
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
      for (int i = 0; i < lat_4dim12; i++)
      {
        s[i] = r[i] - v[i] * alpha;
      }
      // t = A * s;
      cg_in = s;
      cg_out = t;
#ifndef TEST_MPI_WILSON_CG
      // clear vecs for mpi_wilson_dslash
      {
        give_value(b_x_send_vec, zero, lat_3dim6[YZT]);
        give_value(f_x_send_vec, zero, lat_3dim6[YZT]);
        give_value(b_y_send_vec, zero, lat_3dim6[XZT]);
        give_value(f_y_send_vec, zero, lat_3dim6[XZT]);
        give_value(b_z_send_vec, zero, lat_3dim6[XYT]);
        give_value(f_z_send_vec, zero, lat_3dim6[XYT]);
        give_value(b_t_send_vec, zero, lat_3dim6[XYZ]);
        give_value(f_t_send_vec, zero, lat_3dim6[XYZ]);
        give_value(b_x_recv_vec, zero, lat_3dim6[YZT]);
        give_value(f_x_recv_vec, zero, lat_3dim6[YZT]);
        give_value(b_y_recv_vec, zero, lat_3dim6[XZT]);
        give_value(f_y_recv_vec, zero, lat_3dim6[XZT]);
        give_value(b_z_recv_vec, zero, lat_3dim6[XYT]);
        give_value(f_z_recv_vec, zero, lat_3dim6[XYT]);
        give_value(b_t_recv_vec, zero, lat_3dim6[XYZ]);
        give_value(f_t_recv_vec, zero, lat_3dim6[XYZ]);
      }
      // mpi_wilson_dslash
      {
        // clean
        wilson_dslash_clear_dest<<<gridDim, blockDim>>>(cg_out, lat_1dim[X], lat_1dim[Y],
                                                        lat_1dim[Z]);
        // send x
        wilson_dslash_x_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_1dim[X],
                                                    lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity,
                                                    b_x_send_vec, f_x_send_vec);
        if (grid_1dim[X] != 1)
        {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_1dim[X], grid_1dim[X]);
          move_forward(move_f, grid_index_1dim[X], grid_1dim[X]);
          move_b = node_rank + move_b * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];
          move_f = node_rank + move_f * grid_1dim[Y] * grid_1dim[Z] * grid_1dim[T];
          MPI_Irecv(b_x_recv_vec, lat_3dim12[YZT], MPI_DOUBLE, move_b, 1,
                    MPI_COMM_WORLD, &b_x_recv_request);
          MPI_Irecv(f_x_recv_vec, lat_3dim12[YZT], MPI_DOUBLE, move_f, 0,
                    MPI_COMM_WORLD, &f_x_recv_request);
          MPI_Isend(b_x_send_vec, lat_3dim12[YZT], MPI_DOUBLE, move_b, 0,
                    MPI_COMM_WORLD, &b_x_send_request);
          MPI_Isend(f_x_send_vec, lat_3dim12[YZT], MPI_DOUBLE, move_f, 1,
                    MPI_COMM_WORLD, &f_x_send_request);
        }
        // send y
        wilson_dslash_y_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_1dim[X],
                                                    lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity,
                                                    b_y_send_vec, f_y_send_vec);
        if (grid_1dim[Y] != 1)
        {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_1dim[Y], grid_1dim[Y]);
          move_forward(move_f, grid_index_1dim[Y], grid_1dim[Y]);
          move_b = node_rank + move_b * grid_1dim[Z] * grid_1dim[T];
          move_f = node_rank + move_f * grid_1dim[Z] * grid_1dim[T];
          MPI_Irecv(b_y_recv_vec, lat_3dim12[XZT], MPI_DOUBLE, move_b, 3,
                    MPI_COMM_WORLD, &b_y_recv_request);
          MPI_Irecv(f_y_recv_vec, lat_3dim12[XZT], MPI_DOUBLE, move_f, 2,
                    MPI_COMM_WORLD, &f_y_recv_request);
          MPI_Isend(b_y_send_vec, lat_3dim12[XZT], MPI_DOUBLE, move_b, 2,
                    MPI_COMM_WORLD, &b_y_send_request);
          MPI_Isend(f_y_send_vec, lat_3dim12[XZT], MPI_DOUBLE, move_f, 3,
                    MPI_COMM_WORLD, &f_y_send_request);
        }
        // send z
        wilson_dslash_z_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_1dim[X],
                                                    lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity,
                                                    b_z_send_vec, f_z_send_vec);
        if (grid_1dim[Z] != 1)
        {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_1dim[Z], grid_1dim[Z]);
          move_forward(move_f, grid_index_1dim[Z], grid_1dim[Z]);
          move_b = node_rank + move_b * grid_1dim[T];
          move_f = node_rank + move_f * grid_1dim[T];
          MPI_Irecv(b_z_recv_vec, lat_3dim12[XYT], MPI_DOUBLE, move_b, 5,
                    MPI_COMM_WORLD, &b_z_recv_request);
          MPI_Irecv(f_z_recv_vec, lat_3dim12[XYT], MPI_DOUBLE, move_f, 4,
                    MPI_COMM_WORLD, &f_z_recv_request);
          MPI_Isend(b_z_send_vec, lat_3dim12[XYT], MPI_DOUBLE, move_b, 4,
                    MPI_COMM_WORLD, &b_z_send_request);
          MPI_Isend(f_z_send_vec, lat_3dim12[XYT], MPI_DOUBLE, move_f, 5,
                    MPI_COMM_WORLD, &f_z_send_request);
        }
        // send t
        wilson_dslash_t_send<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_1dim[X],
                                                    lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity,
                                                    b_t_send_vec, f_t_send_vec);
        if (grid_1dim[T] != 1)
        {
          checkCudaErrors(cudaDeviceSynchronize());
          move_backward(move_b, grid_index_1dim[T], grid_1dim[T]);
          move_forward(move_f, grid_index_1dim[T], grid_1dim[T]);
          move_b = node_rank + move_b;
          move_f = node_rank + move_f;
          MPI_Irecv(b_t_recv_vec, lat_3dim12[XYZ], MPI_DOUBLE, move_b, 7,
                    MPI_COMM_WORLD, &b_t_recv_request);
          MPI_Irecv(f_t_recv_vec, lat_3dim12[XYZ], MPI_DOUBLE, move_f, 6,
                    MPI_COMM_WORLD, &f_t_recv_request);
          MPI_Isend(b_t_send_vec, lat_3dim12[XYZ], MPI_DOUBLE, move_b, 6,
                    MPI_COMM_WORLD, &b_t_send_request);
          MPI_Isend(f_t_send_vec, lat_3dim12[XYZ], MPI_DOUBLE, move_f, 7,
                    MPI_COMM_WORLD, &f_t_send_request);
        }
        // recv x
        if (grid_1dim[X] != 1)
        {
          MPI_Wait(&b_x_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_x_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_x_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, b_x_recv_vec,
              f_x_recv_vec);
        }
        else
        {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_x_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, f_x_send_vec,
              b_x_send_vec);
        }
        // recv y
        if (grid_1dim[Y] != 1)
        {
          MPI_Wait(&b_y_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_y_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_y_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, b_y_recv_vec,
              f_y_recv_vec);
        }
        else
        {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_y_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, f_y_send_vec,
              b_y_send_vec);
        }
        // recv z
        if (grid_1dim[Z] != 1)
        {
          MPI_Wait(&b_z_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_z_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_z_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, b_z_recv_vec,
              f_z_recv_vec);
        }
        else
        {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_z_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, f_z_send_vec,
              b_z_send_vec);
        }
        // recv t
        if (grid_1dim[T] != 1)
        {
          MPI_Wait(&b_t_recv_request, MPI_STATUS_IGNORE);
          MPI_Wait(&f_t_recv_request, MPI_STATUS_IGNORE);
          wilson_dslash_t_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, b_t_recv_vec,
              f_t_recv_vec);
        }
        else
        {
          checkCudaErrors(cudaDeviceSynchronize());
          wilson_dslash_t_recv<<<gridDim, blockDim>>>(
              gauge, cg_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z], lat_1dim[T], parity, f_t_send_vec,
              b_t_send_vec);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
      // kappa
      {
        for (int i = 0; i < lat_4dim12; i++)
        {
          cg_out[i] = cg_in[i] - cg_out[i] * Kappa;
        }
      }
#else
#ifdef TEST_MPI_WILSON_CG_USE_WILSON_DSLASH
      // wilson_dslash
      wilson_dslash<<<gridDim, blockDim>>>(gauge, cg_in, cg_out, lat_1dim[X], lat_1dim[Y],
                                           lat_1dim[Z], lat_1dim[T], parity);
      // kappa
      {
        for (int i = 0; i < lat_4dim12; i++)
        {
          cg_out[i] = cg_in[i] - cg_out[i] * Kappa;
        }
      }
#else
      for (int i = 0; i < lat_4dim12; i++)
      {
        cg_out[i] = cg_in[i] * 2 + one;
      }
#endif
#endif
      {
        local_result = zero;
        for (int i = 0; i < lat_4dim12; i++)
        {
          local_result += t[i].conj() * s[i];
        }
        MPI_Allreduce(&local_result, &tmp0, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      {
        local_result = zero;
        for (int i = 0; i < lat_4dim12; i++)
        {
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
      for (int i = 0; i < lat_4dim12; i++)
      {
        x[i] = x[i] + p[i] * alpha + s[i] * omega;
      }
      for (int i = 0; i < lat_4dim12; i++)
      {
        r[i] = s[i] - t[i] * omega;
      }
      {
        local_result = zero;
        for (int i = 0; i < lat_4dim12; i++)
        {
          local_result += r[i].conj() * r[i];
        }
        MPI_Allreduce(&local_result, &r_norm2, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      std::cout << "##RANK:" << node_rank << "##LOOP:" << loop
                << "##Residual:" << r_norm2.real << std::endl;
      // break;
      if (r_norm2.real < TOL || loop == MAX_ITER - 1)
      {
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