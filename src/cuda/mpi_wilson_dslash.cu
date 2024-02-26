#include <iostream>
#pragma optimize(5)
#include "../../include/qcu.h"

#ifdef MPI_WILSON_DSLASH
void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                  QcuParam *param, int parity, QcuParam *grid) {
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
    // mpi wilson dslash
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
    // above define for mpi_wilson_dslash
    auto start = std::chrono::high_resolution_clock::now();
    // clean
    wilson_dslash_clear_dest<<<gridDim, blockDim>>>(fermion_out, lat_1dim[X],
                                                    lat_1dim[Y], lat_1dim[Z]);
    // send x
    wilson_dslash_x_send<<<gridDim, blockDim>>>(
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
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
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
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
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
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
        gauge, fermion_in, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
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
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
          lat_1dim[T], parity, recv_vec[B_X], recv_vec[F_X]);
    } else {
      checkCudaErrors(cudaDeviceSynchronize());
      wilson_dslash_x_recv<<<gridDim, blockDim>>>(
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
          lat_1dim[T], parity, send_vec[F_X], send_vec[B_X]);
    }
    // recv y
    if (grid_1dim[Y] != 1) {
      MPI_Wait(&recv_request[B_Y], MPI_STATUS_IGNORE);
      MPI_Wait(&recv_request[F_Y], MPI_STATUS_IGNORE);
      wilson_dslash_y_recv<<<gridDim, blockDim>>>(
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
          lat_1dim[T], parity, recv_vec[B_Y], recv_vec[F_Y]);
    } else {
      checkCudaErrors(cudaDeviceSynchronize());
      wilson_dslash_y_recv<<<gridDim, blockDim>>>(
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
          lat_1dim[T], parity, send_vec[F_Y], send_vec[B_Y]);
    }
    // recv z
    if (grid_1dim[Z] != 1) {
      MPI_Wait(&recv_request[B_Z], MPI_STATUS_IGNORE);
      MPI_Wait(&recv_request[F_Z], MPI_STATUS_IGNORE);
      wilson_dslash_z_recv<<<gridDim, blockDim>>>(
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
          lat_1dim[T], parity, recv_vec[B_Z], recv_vec[F_Z]);
    } else {
      checkCudaErrors(cudaDeviceSynchronize());
      wilson_dslash_z_recv<<<gridDim, blockDim>>>(
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
          lat_1dim[T], parity, send_vec[F_Z], send_vec[B_Z]);
    }
    // recv t
    if (grid_1dim[T] != 1) {
      MPI_Wait(&recv_request[B_T], MPI_STATUS_IGNORE);
      MPI_Wait(&recv_request[F_T], MPI_STATUS_IGNORE);
      wilson_dslash_t_recv<<<gridDim, blockDim>>>(
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
          lat_1dim[T], parity, recv_vec[B_T], recv_vec[F_T]);
    } else {
      checkCudaErrors(cudaDeviceSynchronize());
      wilson_dslash_t_recv<<<gridDim, blockDim>>>(
          gauge, fermion_out, lat_1dim[X], lat_1dim[Y], lat_1dim[Z],
          lat_1dim[T], parity, send_vec[F_T], send_vec[B_T]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    err = cudaGetLastError();
    checkCudaErrors(err);
    printf("mpi wilson dslash total time: (without malloc free memcpy) :%.9lf "
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
    }
  }
}
#endif