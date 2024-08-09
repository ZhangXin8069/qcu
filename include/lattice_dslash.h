#ifndef _LATTICE_DSLASH_H
#define _LATTICE_DSLASH_H
#include "./dslash.h"
#include "./lattice_set.h"
#include "define.h"
#include "wilson_dslash.h"
struct LatticeWilsonDslash {
  LatticeSet *set_ptr;
  cudaError_t err;
  void give(LatticeSet *_set_ptr) { set_ptr = _set_ptr; }
  void run_nccl(void *fermion_out, void *fermion_in, void *gauge, int parity) {
    { // edge send part
      wilson_dslash_x_send<<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                             set_ptr->stream_dims[_X_]>>>(
          gauge, fermion_in, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_X_], set_ptr->device_send_vec[_F_X_]);
      wilson_dslash_y_send<<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                             set_ptr->stream_dims[_Y_]>>>(
          gauge, fermion_in, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_Y_], set_ptr->device_send_vec[_F_Y_]);
      wilson_dslash_z_send<<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                             set_ptr->stream_dims[_Z_]>>>(
          gauge, fermion_in, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_Z_], set_ptr->device_send_vec[_F_Z_]);
      wilson_dslash_t_send<<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                             set_ptr->stream_dims[_T_]>>>(
          gauge, fermion_in, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_T_], set_ptr->device_send_vec[_F_T_]);
    }
    { // inside compute part
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      wilson_dslash_inside<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                             set_ptr->stream>>>(gauge, fermion_in, fermion_out,
                                                set_ptr->device_xyztsc, parity);
    }
    {
      // x edge part
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
      if (set_ptr->grid_1dim[_X_] == 1) {
        // no comm
        // edge recv part
        wilson_dslash_x_recv<<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_send_vec[_F_X_], set_ptr->device_send_vec[_B_X_]);
      } else {
        // comm
        ncclGroupStart();
        ncclSend(set_ptr->device_send_vec[_B_X_], set_ptr->lat_3dim_SC[_X_],
                 ncclDouble, set_ptr->move_wards[_B_X_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_X_]);
        ncclRecv(set_ptr->device_recv_vec[_F_X_], set_ptr->lat_3dim_SC[_X_],
                 ncclDouble, set_ptr->move_wards[_F_X_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_X_]);
        ncclGroupEnd();
        ncclGroupStart();
        ncclSend(set_ptr->device_send_vec[_F_X_], set_ptr->lat_3dim_SC[_X_],
                 ncclDouble, set_ptr->move_wards[_F_X_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_X_]);
        ncclRecv(set_ptr->device_recv_vec[_B_X_], set_ptr->lat_3dim_SC[_X_],
                 ncclDouble, set_ptr->move_wards[_B_X_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_X_]);
        ncclGroupEnd();
      }
    }
    {
      // y edge part
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
      if (set_ptr->grid_1dim[_Y_] == 1) {
        // no comm
        // edge recv part
        wilson_dslash_y_recv<<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_send_vec[_F_Y_], set_ptr->device_send_vec[_B_Y_]);
      } else {
        // comm
        ncclGroupStart();
        ncclSend(set_ptr->device_send_vec[_B_Y_], set_ptr->lat_3dim_SC[_Y_],
                 ncclDouble, set_ptr->move_wards[_B_Y_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_Y_]);
        ncclRecv(set_ptr->device_recv_vec[_F_Y_], set_ptr->lat_3dim_SC[_Y_],
                 ncclDouble, set_ptr->move_wards[_F_Y_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_Y_]);
        ncclGroupEnd();
        ncclGroupStart();
        ncclSend(set_ptr->device_send_vec[_F_Y_], set_ptr->lat_3dim_SC[_Y_],
                 ncclDouble, set_ptr->move_wards[_F_Y_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_Y_]);
        ncclRecv(set_ptr->device_recv_vec[_B_Y_], set_ptr->lat_3dim_SC[_Y_],
                 ncclDouble, set_ptr->move_wards[_B_Y_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_Y_]);
        ncclGroupEnd();
      }
    }
    {
      // z edge part
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
      if (set_ptr->grid_1dim[_Z_] == 1) {
        // no comm
        // edge recv part
        wilson_dslash_z_recv<<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_send_vec[_F_Z_], set_ptr->device_send_vec[_B_Z_]);
      } else {
        // comm
        ncclGroupStart();
        ncclSend(set_ptr->device_send_vec[_B_Z_], set_ptr->lat_3dim_SC[_Z_],
                 ncclDouble, set_ptr->move_wards[_B_Z_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_Z_]);
        ncclRecv(set_ptr->device_recv_vec[_F_Z_], set_ptr->lat_3dim_SC[_Z_],
                 ncclDouble, set_ptr->move_wards[_F_Z_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_Z_]);
        ncclGroupEnd();
        ncclGroupStart();
        ncclSend(set_ptr->device_send_vec[_F_Z_], set_ptr->lat_3dim_SC[_Z_],
                 ncclDouble, set_ptr->move_wards[_F_Z_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_Z_]);
        ncclRecv(set_ptr->device_recv_vec[_B_Z_], set_ptr->lat_3dim_SC[_Z_],
                 ncclDouble, set_ptr->move_wards[_B_Z_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_Z_]);
        ncclGroupEnd();
      }
    }
    {
      // t edge part
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
      if (set_ptr->grid_1dim[_T_] == 1) {
        // no comm
        // edge recv part
        wilson_dslash_t_recv<<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_send_vec[_F_T_], set_ptr->device_send_vec[_B_T_]);
      } else {
        // comm
        ncclGroupStart();
        ncclSend(set_ptr->device_send_vec[_B_T_], set_ptr->lat_3dim_SC[_T_],
                 ncclDouble, set_ptr->move_wards[_B_T_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_T_]);
        ncclRecv(set_ptr->device_recv_vec[_F_T_], set_ptr->lat_3dim_SC[_T_],
                 ncclDouble, set_ptr->move_wards[_F_T_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_T_]);
        ncclGroupEnd();
        ncclGroupStart();
        ncclSend(set_ptr->device_send_vec[_F_T_], set_ptr->lat_3dim_SC[_T_],
                 ncclDouble, set_ptr->move_wards[_F_T_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_T_]);
        ncclRecv(set_ptr->device_recv_vec[_B_T_], set_ptr->lat_3dim_SC[_T_],
                 ncclDouble, set_ptr->move_wards[_B_T_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_T_]);
        ncclGroupEnd();
      }
    }
    {
      // edge recv part
      if (set_ptr->grid_1dim[_X_] != 1) { // x part recv
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
        wilson_dslash_x_recv<<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_recv_vec[_B_X_], set_ptr->device_recv_vec[_F_X_]);
      }
      if (set_ptr->grid_1dim[_Y_] != 1) { // y part recv
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
        wilson_dslash_y_recv<<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_recv_vec[_B_Y_], set_ptr->device_recv_vec[_F_Y_]);
      }
      if (set_ptr->grid_1dim[_Z_] != 1) { // z part recv
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
        wilson_dslash_z_recv<<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_recv_vec[_B_Z_], set_ptr->device_recv_vec[_F_Z_]);
      }
      if (set_ptr->grid_1dim[_T_] != 1) { // t part recv
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
        wilson_dslash_t_recv<<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_recv_vec[_B_T_], set_ptr->device_recv_vec[_F_T_]);
      }
    }
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
  }
  void run_mpi(void *fermion_out, void *fermion_in, void *gauge, int parity) {
    { // edge send part
      wilson_dslash_x_send<<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                             set_ptr->stream_dims[_X_]>>>(
          gauge, fermion_in, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_X_], set_ptr->device_send_vec[_F_X_]);
      if (set_ptr->grid_1dim[_X_] != 1) { // x part d2h
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_B_X_], set_ptr->device_send_vec[_B_X_],
            sizeof(double) * set_ptr->lat_3dim_SC[_X_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_X_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_F_X_], set_ptr->device_send_vec[_F_X_],
            sizeof(double) * set_ptr->lat_3dim_SC[_X_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_X_]));
      }
      wilson_dslash_y_send<<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                             set_ptr->stream_dims[_Y_]>>>(
          gauge, fermion_in, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_Y_], set_ptr->device_send_vec[_F_Y_]);
      if (set_ptr->grid_1dim[_Y_] != 1) { // y part d2h
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_B_Y_], set_ptr->device_send_vec[_B_Y_],
            sizeof(double) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_Y_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_F_Y_], set_ptr->device_send_vec[_F_Y_],
            sizeof(double) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_Y_]));
      }
      wilson_dslash_z_send<<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                             set_ptr->stream_dims[_Z_]>>>(
          gauge, fermion_in, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_Z_], set_ptr->device_send_vec[_F_Z_]);
      if (set_ptr->grid_1dim[_Z_] != 1) { // z part d2h
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_B_Z_], set_ptr->device_send_vec[_B_Z_],
            sizeof(double) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_Z_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_F_Z_], set_ptr->device_send_vec[_F_Z_],
            sizeof(double) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_Z_]));
      }
      wilson_dslash_t_send<<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                             set_ptr->stream_dims[_T_]>>>(
          gauge, fermion_in, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_T_], set_ptr->device_send_vec[_F_T_]);
      if (set_ptr->grid_1dim[_T_] != 1) { // t part d2h
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_B_T_], set_ptr->device_send_vec[_B_T_],
            sizeof(double) * set_ptr->lat_3dim_SC[_T_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_T_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_F_T_], set_ptr->device_send_vec[_F_T_],
            sizeof(double) * set_ptr->lat_3dim_SC[_T_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_T_]));
      }
    }
    { // inside compute part ans wait
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      wilson_dslash_inside<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                             set_ptr->stream>>>(gauge, fermion_in, fermion_out,
                                                set_ptr->device_xyztsc, parity);
    }
    {
      // x edge part
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
      if (set_ptr->grid_1dim[_X_] == 1) {
        // no comm
        // edge recv part
        wilson_dslash_x_recv<<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_send_vec[_F_X_], set_ptr->device_send_vec[_B_X_]);
      } else {
        // comm
        MPI_Isend(set_ptr->host_send_vec[_B_X_], set_ptr->lat_3dim_SC[_X_],
                  MPI_DOUBLE, set_ptr->move_wards[_B_X_], _B_X_, MPI_COMM_WORLD,
                  &set_ptr->send_request[_B_X_]);
        MPI_Irecv(set_ptr->host_recv_vec[_F_X_], set_ptr->lat_3dim_SC[_X_],
                  MPI_DOUBLE, set_ptr->move_wards[_F_X_], _B_X_, MPI_COMM_WORLD,
                  &set_ptr->recv_request[_B_X_]);
        MPI_Isend(set_ptr->host_send_vec[_F_X_], set_ptr->lat_3dim_SC[_X_],
                  MPI_DOUBLE, set_ptr->move_wards[_F_X_], _F_X_, MPI_COMM_WORLD,
                  &set_ptr->send_request[_F_X_]);
        MPI_Irecv(set_ptr->host_recv_vec[_B_X_], set_ptr->lat_3dim_SC[_X_],
                  MPI_DOUBLE, set_ptr->move_wards[_B_X_], _F_X_, MPI_COMM_WORLD,
                  &set_ptr->recv_request[_F_X_]);
      }
    }
    {
      // y edge part
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
      if (set_ptr->grid_1dim[_Y_] == 1) {
        // no comm
        // edge recv part
        wilson_dslash_y_recv<<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_send_vec[_F_Y_], set_ptr->device_send_vec[_B_Y_]);
      } else {
        // comm
        MPI_Isend(set_ptr->host_send_vec[_B_Y_], set_ptr->lat_3dim_SC[_Y_],
                  MPI_DOUBLE, set_ptr->move_wards[_B_Y_], _B_Y_, MPI_COMM_WORLD,
                  &set_ptr->send_request[_B_Y_]);
        MPI_Irecv(set_ptr->host_recv_vec[_F_Y_], set_ptr->lat_3dim_SC[_Y_],
                  MPI_DOUBLE, set_ptr->move_wards[_F_Y_], _B_Y_, MPI_COMM_WORLD,
                  &set_ptr->recv_request[_B_Y_]);
        MPI_Isend(set_ptr->host_send_vec[_F_Y_], set_ptr->lat_3dim_SC[_Y_],
                  MPI_DOUBLE, set_ptr->move_wards[_F_Y_], _F_Y_, MPI_COMM_WORLD,
                  &set_ptr->send_request[_F_Y_]);
        MPI_Irecv(set_ptr->host_recv_vec[_B_Y_], set_ptr->lat_3dim_SC[_Y_],
                  MPI_DOUBLE, set_ptr->move_wards[_B_Y_], _F_Y_, MPI_COMM_WORLD,
                  &set_ptr->recv_request[_F_Y_]);
      }
    }
    {
      // z edge part
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
      if (set_ptr->grid_1dim[_Z_] == 1) {
        // no comm
        // edge recv part
        wilson_dslash_z_recv<<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_send_vec[_F_Z_], set_ptr->device_send_vec[_B_Z_]);
      } else {
        // comm
        MPI_Isend(set_ptr->host_send_vec[_B_Z_], set_ptr->lat_3dim_SC[_Z_],
                  MPI_DOUBLE, set_ptr->move_wards[_B_Z_], _B_Z_, MPI_COMM_WORLD,
                  &set_ptr->send_request[_B_Z_]);
        MPI_Irecv(set_ptr->host_recv_vec[_F_Z_], set_ptr->lat_3dim_SC[_Z_],
                  MPI_DOUBLE, set_ptr->move_wards[_F_Z_], _B_Z_, MPI_COMM_WORLD,
                  &set_ptr->recv_request[_B_Z_]);
        MPI_Isend(set_ptr->host_send_vec[_F_Z_], set_ptr->lat_3dim_SC[_Z_],
                  MPI_DOUBLE, set_ptr->move_wards[_F_Z_], _F_Z_, MPI_COMM_WORLD,
                  &set_ptr->send_request[_F_Z_]);
        MPI_Irecv(set_ptr->host_recv_vec[_B_Z_], set_ptr->lat_3dim_SC[_Z_],
                  MPI_DOUBLE, set_ptr->move_wards[_B_Z_], _F_Z_, MPI_COMM_WORLD,
                  &set_ptr->recv_request[_F_Z_]);
      }
    }
    {
      // t edge part
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
      if (set_ptr->grid_1dim[_T_] == 1) {
        // no comm
        // edge recv part
        wilson_dslash_t_recv<<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_send_vec[_F_T_], set_ptr->device_send_vec[_B_T_]);
      } else {
        // comm
        MPI_Isend(set_ptr->host_send_vec[_B_T_], set_ptr->lat_3dim_SC[_T_],
                  MPI_DOUBLE, set_ptr->move_wards[_B_T_], _B_T_, MPI_COMM_WORLD,
                  &set_ptr->send_request[_B_T_]);
        MPI_Irecv(set_ptr->host_recv_vec[_F_T_], set_ptr->lat_3dim_SC[_T_],
                  MPI_DOUBLE, set_ptr->move_wards[_F_T_], _B_T_, MPI_COMM_WORLD,
                  &set_ptr->recv_request[_B_T_]);
        MPI_Isend(set_ptr->host_send_vec[_F_T_], set_ptr->lat_3dim_SC[_T_],
                  MPI_DOUBLE, set_ptr->move_wards[_F_T_], _F_T_, MPI_COMM_WORLD,
                  &set_ptr->send_request[_F_T_]);
        MPI_Irecv(set_ptr->host_recv_vec[_B_T_], set_ptr->lat_3dim_SC[_T_],
                  MPI_DOUBLE, set_ptr->move_wards[_B_T_], _F_T_, MPI_COMM_WORLD,
                  &set_ptr->recv_request[_F_T_]);
      }
    }
    if (set_ptr->grid_1dim[_X_] != 1) { // x part h2d
      MPI_Wait(&set_ptr->recv_request[_B_X_], MPI_STATUS_IGNORE);
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_F_X_], set_ptr->host_recv_vec[_F_X_],
          sizeof(double) * set_ptr->lat_3dim_SC[_X_], cudaMemcpyHostToDevice,
          set_ptr->stream_dims[_X_]));
      MPI_Wait(&set_ptr->recv_request[_F_X_], MPI_STATUS_IGNORE);
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_B_X_], set_ptr->host_recv_vec[_B_X_],
          sizeof(double) * set_ptr->lat_3dim_SC[_X_], cudaMemcpyHostToDevice,
          set_ptr->stream_dims[_X_]));
    }
    if (set_ptr->grid_1dim[_Y_] != 1) { // y part h2d
      MPI_Wait(&set_ptr->recv_request[_B_Y_], MPI_STATUS_IGNORE);
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_F_Y_], set_ptr->host_recv_vec[_F_Y_],
          sizeof(double) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyHostToDevice,
          set_ptr->stream_dims[_Y_]));
      MPI_Wait(&set_ptr->recv_request[_F_Y_], MPI_STATUS_IGNORE);
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_B_Y_], set_ptr->host_recv_vec[_B_Y_],
          sizeof(double) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyHostToDevice,
          set_ptr->stream_dims[_Y_]));
    }
    if (set_ptr->grid_1dim[_Z_] != 1) { // z part h2d
      MPI_Wait(&set_ptr->recv_request[_B_Z_], MPI_STATUS_IGNORE);
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_F_Z_], set_ptr->host_recv_vec[_F_Z_],
          sizeof(double) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyHostToDevice,
          set_ptr->stream_dims[_Z_]));
      MPI_Wait(&set_ptr->recv_request[_F_Z_], MPI_STATUS_IGNORE);
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_B_Z_], set_ptr->host_recv_vec[_B_Z_],
          sizeof(double) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyHostToDevice,
          set_ptr->stream_dims[_Z_]));
    }
    if (set_ptr->grid_1dim[_T_] != 1) { // t part h2d
      MPI_Wait(&set_ptr->recv_request[_B_T_], MPI_STATUS_IGNORE);
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_F_T_], set_ptr->host_recv_vec[_F_T_],
          sizeof(double) * set_ptr->lat_3dim_SC[_T_], cudaMemcpyHostToDevice,
          set_ptr->stream_dims[_T_]));
      MPI_Wait(&set_ptr->recv_request[_F_T_], MPI_STATUS_IGNORE);
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_B_T_], set_ptr->host_recv_vec[_B_T_],
          sizeof(double) * set_ptr->lat_3dim_SC[_T_], cudaMemcpyHostToDevice,
          set_ptr->stream_dims[_T_]));
    }
    {
      // edge recv part
      if (set_ptr->grid_1dim[_X_] != 1) { // x part recv
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
        wilson_dslash_x_recv<<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_recv_vec[_B_X_], set_ptr->device_recv_vec[_F_X_]);
      }
      if (set_ptr->grid_1dim[_Y_] != 1) { // y part recv
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
        wilson_dslash_y_recv<<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_recv_vec[_B_Y_], set_ptr->device_recv_vec[_F_Y_]);
      }
      if (set_ptr->grid_1dim[_Z_] != 1) { // z part recv
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
        wilson_dslash_z_recv<<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_recv_vec[_B_Z_], set_ptr->device_recv_vec[_F_Z_]);
      }
      if (set_ptr->grid_1dim[_T_] != 1) { // t part recv
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
        wilson_dslash_t_recv<<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                               set_ptr->stream>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_recv_vec[_B_T_], set_ptr->device_recv_vec[_F_T_]);
      }
    }
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
  }
  void run(void *fermion_out, void *fermion_in, void *gauge, int parity) {
    // run_mpi(fermion_out, fermion_in, gauge, parity);
    run_nccl(fermion_out, fermion_in, gauge, parity);
  }
  void run_eo(void *fermion_out, void *fermion_in, void *gauge) {
    run(fermion_out, fermion_in, gauge, _EVEN_);
  }
  void run_oe(void *fermion_out, void *fermion_in, void *gauge) {
    run(fermion_out, fermion_in, gauge, _ODD_);
  }
  void run_test(void *fermion_out, void *fermion_in, void *gauge, int parity) {
#ifdef PRINT_NCCL_WILSON_DSLASH
    set_ptr->_print();
#endif
    auto start = std::chrono::high_resolution_clock::now();
    run(fermion_out, fermion_in, gauge, parity);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    err = cudaGetLastError();
    checkCudaErrors(err);
    printf("nccl wilson dslash total time: (without malloc free memcpy) :%.9lf "
           "sec\n",
           double(duration) / 1e9);
  }
};
#endif
