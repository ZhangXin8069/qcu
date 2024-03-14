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
    MPI_Request send_request[B_X], b_x_recv_request;
    MPI_Request f_x_send_request, f_x_recv_request;
    MPI_Request send_request[B_Y], b_y_recv_request;
    MPI_Request f_y_send_request, f_y_recv_request;
    MPI_Request send_request[B_Z], b_z_recv_request;
    MPI_Request f_z_send_request, f_z_recv_request;
    MPI_Request send_request[B_T], b_t_recv_request;
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
                  MPI_COMM_WORLD, &send_request[B_X]);
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
                  MPI_COMM_WORLD, &send_request[B_Y]);
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
                  MPI_COMM_WORLD, &send_request[B_Z]);
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
                  MPI_COMM_WORLD, &send_request[B_T]);
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