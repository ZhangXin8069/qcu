#include "../../include/qcu.h"
#include "../../include/qcu_cuda.h"
#include <chrono>

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge,
               QcuParam *param, int parity) {
  const int lat_x = param->lattice_size[0] >> 1;
  const int lat_y = param->lattice_size[1];
  const int lat_z = param->lattice_size[2];
  const int lat_t = param->lattice_size[3];
  void *clover;
  checkCudaErrors(cudaMalloc(&clover, (lat_t * lat_z * lat_y * lat_x * 144) *
                                          sizeof(LatticeComplex)));
  cudaError_t err;
  dim3 gridDim(lat_x * lat_y * lat_z * lat_t / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  {
    // wilson dslash
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    wilson_dslash<<<gridDim, blockDim>>>(gauge, fermion_in, fermion_out, lat_x,
                                         lat_y, lat_z, lat_t, parity);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf(
        "wilson dslash total time: (without malloc free memcpy) : %.9lf sec\n",
        double(duration) / 1e9);
  }
  {
    // make clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    make_clover<<<gridDim, blockDim>>>(gauge, clover, lat_x, lat_y, lat_z,
                                       lat_t, parity);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("make clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  {
    // inverse clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    inverse_clover<<<gridDim, blockDim>>>(clover, lat_x, lat_y, lat_z);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf(
        "inverse clover total time: (without malloc free memcpy) :%.9lf sec\n ",
        double(duration) / 1e9);
  }
  {
    // give clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    give_clover<<<gridDim, blockDim>>>(clover, fermion_out, lat_x, lat_y,
                                       lat_z);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("give clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  {
    // free
    checkCudaErrors(cudaFree(clover));
  }
}

void mpiDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                  QcuParam *param, int parity, QcuParam *grid) {
  const int lat_x = param->lattice_size[0] >> 1;
  const int lat_y = param->lattice_size[1];
  const int lat_z = param->lattice_size[2];
  const int lat_t = param->lattice_size[3];
  cudaError_t err;
  dim3 gridDim(lat_x * lat_y * lat_z * lat_t / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  {
    // mpi wilson dslash
    int node_size, node_rank, move, tmp_rank;
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
    printf("node_size: %d \n", node_size);
    printf("node_rank: %d \n", node_rank);
    printf("grid:x-%d, y-%d, z-%d, t-%d \n", grid_x, grid_y, grid_z, grid_t);
    MPI_Request *b_x_send_request;
    MPI_Request *f_x_send_request;
    MPI_Request *b_y_send_request;
    MPI_Request *f_y_send_request;
    MPI_Request *b_z_send_request;
    MPI_Request *f_z_send_request;
    MPI_Request *b_t_send_request;
    MPI_Request *f_t_send_request;
    void *b_x_send_vec;
    void *f_x_send_vec;
    void *b_y_send_vec;
    void *f_y_send_vec;
    void *b_z_send_vec;
    void *f_z_send_vec;
    void *b_t_send_vec;
    void *f_t_send_vec;
    checkCudaErrors(Malloc(&b_x_send_vec, (lat_t * lat_z * lat_y * 6) *
                                              sizeof(LatticeComplex)));
    checkCudaErrors(Malloc(&f_x_send_vec, (lat_t * lat_z * lat_y * 6) *
                                              sizeof(LatticeComplex)));
    checkCudaErrors(Malloc(&b_y_send_vec, (lat_t * lat_z * lat_x * 6) *
                                              sizeof(LatticeComplex)));
    checkCudaErrors(Malloc(&f_y_send_vec, (lat_t * lat_z * lat_x * 6) *
                                              sizeof(LatticeComplex)));
    checkCudaErrors(Malloc(&b_z_send_vec, (lat_t * lat_y * lat_x * 6) *
                                              sizeof(LatticeComplex)));
    checkCudaErrors(Malloc(&f_z_send_vec, (lat_t * lat_y * lat_x * 6) *
                                              sizeof(LatticeComplex)));
    checkCudaErrors(Malloc(&b_t_send_vec, (lat_z * lat_y * lat_x * 6) *
                                              sizeof(LatticeComplex)));
    checkCudaErrors(Malloc(&f_t_send_vec, (lat_z * lat_y * lat_x * 6) *
                                              sizeof(LatticeComplex)));
    MPI_Request *b_x_recv_request;
    MPI_Request *f_x_recv_request;
    MPI_Request *b_y_recv_request;
    MPI_Request *f_y_recv_request;
    MPI_Request *b_z_recv_request;
    MPI_Request *f_z_recv_request;
    MPI_Request *b_t_recv_request;
    MPI_Request *f_t_recv_request;
    void *b_x_recv_vec;
    void *f_x_recv_vec;
    void *b_y_recv_vec;
    void *f_y_recv_vec;
    void *b_z_recv_vec;
    void *f_z_recv_vec;
    void *b_t_recv_vec;
    void *f_t_recv_vec;
    checkCudaErrors(Malloc(&b_x_recv_vec, (lat_t * lat_z * lat_y * 6) *
                                              sizeof(LatticeComplex)));
    checkCudaErrors(Malloc(&f_x_recv_vec, (lat_t * lat_z * lat_y * 6) *
                                              sizeof(LatticeComplex)));
    checkCudaErrors(Malloc(&b_y_recv_vec, (lat_t * lat_z * lat_x * 6) *
                                              sizeof(LatticeComplex)));
    checkCudaErrors(Malloc(&f_y_recv_vec, (lat_t * lat_z * lat_x * 6) *
                                              sizeof(LatticeComplex)));
    checkCudaErrors(Malloc(&b_z_recv_vec, (lat_t * lat_y * lat_x * 6) *
                                              sizeof(LatticeComplex)));
    checkCudaErrors(Malloc(&f_z_recv_vec, (lat_t * lat_y * lat_x * 6) *
                                              sizeof(LatticeComplex)));
    checkCudaErrors(Malloc(&b_t_recv_vec, (lat_z * lat_y * lat_x * 6) *
                                              sizeof(LatticeComplex)));
    checkCudaErrors(Malloc(&f_t_recv_vec, (lat_z * lat_y * lat_x * 6) *
                                              sizeof(LatticeComplex)));
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    wilson_dslash_clear_dest<<<gridDim, blockDim>>>(fermion_out, lat_x, lat_y,
                                                    lat_z);
    // send
    {
      wilson_dslash_x_send<<<gridDim, blockDim>>>(
          gauge, fermion_in, fermion_out, lat_x, lat_y, lat_z, lat_t, parity,
          grid_x, b_x_send_vec, f_x_send_vec);
      move_backward(move, grid_index_x, grid_x);
      move = node_rank + move * grid_y * grid_z * grid_t;
      MPI_Isend(b_x_send_vec, 12, MPI_DOUBLE, move, move, MPI_COMM_WORLD,
                &b_x_send_request);
      move_forward(move, grid_index_x, grid_x);
      move = node_rank + move * grid_y * grid_z * grid_t;
      MPI_Isend(f_x_send_vec, 12, MPI_DOUBLE, move, node_rank, MPI_COMM_WORLD,
                &f_x_send_request);
    }
    {
      wilson_dslash_y_send<<<gridDim, blockDim>>>(
          gauge, fermion_in, fermion_out, lat_x, lat_y, lat_z, lat_t, parity,
          grid_y, b_y_send_vec, f_y_send_vec);
      move_backward(move, grid_index_y, grid_y);
      move = node_rank + move * grid_z * grid_t;
      MPI_Isend(b_y_send_vec, 12, MPI_DOUBLE, move, move, MPI_COMM_WORLD,
                &b_y_send_request);
      move_forward(move, grid_index_y, grid_y);
      move = node_rank + move * grid_z * grid_t;
      MPI_Isend(f_y_send_vec, 12, MPI_DOUBLE, move, node_rank, MPI_COMM_WORLD,
                &f_y_send_request);
    }
    {
      wilson_dslash_z_send<<<gridDim, blockDim>>>(
          gauge, fermion_in, fermion_out, lat_x, lat_y, lat_z, lat_t, parity,
          grid_z, b_z_send_vec, f_z_send_vec);
      move_backward(move, grid_index_z, grid_z);
      move = node_rank + move * grid_t;
      MPI_Isend(b_z_send_vec, 12, MPI_DOUBLE, move, move, MPI_COMM_WORLD,
                &b_z_send_request);
      move_forward(move, grid_index_z, grid_z);
      move = node_rank + move * grid_t;
      MPI_Isend(f_z_send_vec, 12, MPI_DOUBLE, move, node_rank, MPI_COMM_WORLD,
                &f_z_send_request);
    }
    {
      wilson_dslash_t_send<<<gridDim, blockDim>>>(
          gauge, fermion_in, fermion_out, lat_x, lat_y, lat_z, lat_t, parity,
          grid_t, b_t_send_vec, f_t_send_vec);
      move_backward(move, grid_index_t, grid_t);
      move = node_rank + move;
      MPI_Isend(b_t_send_vec, 12, MPI_DOUBLE, move, move, MPI_COMM_WORLD,
                &b_t_send_request);
      move_forward(move, grid_index_t, grid_t);
      move = node_rank + move;
      MPI_Isend(f_t_send_vec, 12, MPI_DOUBLE, move, node_rank, MPI_COMM_WORLD,
                &f_t_send_request);
    }
    // recv
    {
      move_backward(move, grid_index_x, grid_x);
      move = node_rank + move * grid_y * grid_z * grid_t;
      MPI_Wait(&b_x_recv_request, MPI_STATUS_IGNORE);
      MPI_Irecv(b_x_recv_vec, 12, MPI_DOUBLE, move, move, MPI_COMM_WORLD,
                &b_x_recv_request);
      move_forward(move, grid_index_x, grid_x);
      move = node_rank + move * grid_y * grid_z * grid_t;
      MPI_Wait(&f_x_recv_request, MPI_STATUS_IGNORE);
      MPI_Irecv(f_x_recv_vec, 12, MPI_DOUBLE, move, node_rank, MPI_COMM_WORLD,
                &f_x_recv_request);
      wilson_dslash_x_recv<<<gridDim, blockDim>>>(
          gauge, fermion_out, lat_x, lat_y, lat_z, lat_t, parity, grid_x,
          b_x_recv_vec, f_x_recv_vec);
    }
    {
      move_backward(move, grid_index_y, grid_y);
      move = node_rank + move * grid_z * grid_t;
      MPI_Wait(&b_y_recv_request, MPI_STATUS_IGNORE);
      MPI_Irecv(b_y_recv_vec, 12, MPI_DOUBLE, move, move, MPI_COMM_WORLD,
                &b_y_recv_request);
      move_forward(move, grid_index_y, grid_y);
      move = node_rank + move * grid_z * grid_t;
      MPI_Wait(&f_y_recv_request, MPI_STATUS_IGNORE);
      MPI_Irecv(f_y_recv_vec, 12, MPI_DOUBLE, move, node_rank, MPI_COMM_WORLD,
                &f_y_recv_request);
      wilson_dslash_y_recv<<<gridDim, blockDim>>>(
          gauge, fermion_out, lat_x, lat_y, lat_z, lat_t, parity, grid_y,
          b_y_recv_vec, f_y_recv_vec);
    }
    {
      move_backward(move, grid_index_z, grid_z);
      move = node_rank + move * grid_t;
      MPI_Wait(&b_z_recv_request, MPI_STATUS_IGNORE);
      MPI_Irecv(b_z_recv_vec, 12, MPI_DOUBLE, move, move, MPI_COMM_WORLD,
                &b_z_recv_request);
      move_forward(move, grid_index_z, grid_z);
      move = node_rank + move * grid_t;
      MPI_Wait(&f_z_recv_request, MPI_STATUS_IGNORE);
      MPI_Irecv(f_z_recv_vec, 12, MPI_DOUBLE, move, node_rank, MPI_COMM_WORLD,
                &f_z_recv_request);
      wilson_dslash_z_recv<<<gridDim, blockDim>>>(
          gauge, fermion_out, lat_x, lat_y, lat_z, lat_t, parity, grid_z,
          b_z_recv_vec, f_z_recv_vec);
    }
    {
      move_backward(move, grid_index_t, grid_t);
      move = node_rank + move;
      MPI_Wait(&b_t_recv_request, MPI_STATUS_IGNORE);
      MPI_Irecv(b_t_recv_vec, 12, MPI_DOUBLE, move, move, MPI_COMM_WORLD,
                &b_t_recv_request);
      move_forward(move, grid_index_t, grid_t);
      move = node_rank + move;
      MPI_Wait(&f_t_recv_request, MPI_STATUS_IGNORE);
      MPI_Irecv(f_t_recv_vec, 12, MPI_DOUBLE, move, node_rank, MPI_COMM_WORLD,
                &f_t_recv_request);
      wilson_dslash_t_recv<<<gridDim, blockDim>>>(
          gauge, fermion_out, lat_x, lat_y, lat_z, lat_t, parity, grid_t,
          b_t_recv_vec, f_t_recv_vec);
    }
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("mpi wilson dslash total time: (without malloc free memcpy) : %.9lf "
           "sec\n",
           double(duration) / 1e9);
    {
      // free
      checkCudaErrors(free(b_x_send_vec));
      checkCudaErrors(free(f_x_send_vec));
      checkCudaErrors(free(b_y_send_vec));
      checkCudaErrors(free(f_y_send_vec));
      checkCudaErrors(free(b_z_send_vec));
      checkCudaErrors(free(f_z_send_vec));
      checkCudaErrors(free(b_t_send_vec));
      checkCudaErrors(free(f_t_send_vec));
      checkCudaErrors(free(b_x_recv_vec));
      checkCudaErrors(free(f_x_recv_vec));
      checkCudaErrors(free(b_y_recv_vec));
      checkCudaErrors(free(f_y_recv_vec));
      checkCudaErrors(free(b_z_recv_vec));
      checkCudaErrors(free(f_z_recv_vec));
      checkCudaErrors(free(b_t_recv_vec));
      checkCudaErrors(free(f_t_recv_vec));
    }
  }
}
