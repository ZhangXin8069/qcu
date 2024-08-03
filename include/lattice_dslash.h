#ifndef _LATTICE_DSLASH_H
#define _LATTICE_DSLASH_H
#include "./dslash.h"
#include "./lattice_set.h"
#include "define.h"
struct LatticeWilsonDslash {
  LatticeSet *set_ptr;
  cudaError_t err;
  void give(LatticeSet *_set_ptr) { set_ptr = _set_ptr; }
  void run(void *fermion_out, void *fermion_in, void *gauge, int parity) {
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      // checkCudaErrors(cudaDeviceSynchronize()); // needed
      give_fermi_val<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                          set_ptr->stream>>>(fermion_out, 0.0, 0.0);
      wilson_dslash_x_send<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                             set_ptr->stream_wards[_B_X_]>>>(
          gauge, fermion_in, fermion_out, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_X_], set_ptr->device_send_vec[_F_X_]);
      wilson_dslash_y_send<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                             set_ptr->stream_wards[_B_Y_]>>>(
          gauge, fermion_in, fermion_out, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_Y_], set_ptr->device_send_vec[_F_Y_]);
      wilson_dslash_z_send<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                             set_ptr->stream_wards[_B_Z_]>>>(
          gauge, fermion_in, fermion_out, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_Z_], set_ptr->device_send_vec[_F_Z_]);
      wilson_dslash_t_send<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                             set_ptr->stream_wards[_B_T_]>>>(
          gauge, fermion_in, fermion_out, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_T_], set_ptr->device_send_vec[_F_T_]);
    }
    {
      ncclGroupStart();
      if (set_ptr->grid_1dim[_X_] != 1) {
        move_backward(set_ptr->move[_B_], set_ptr->grid_index_1dim[_X_],
                      set_ptr->grid_1dim[_X_]);
        move_forward(set_ptr->move[_F_], set_ptr->grid_index_1dim[_X_],
                     set_ptr->grid_1dim[_X_]);
        set_ptr->move[_B_] = set_ptr->node_rank + set_ptr->move[_B_] *
                                                      set_ptr->grid_1dim[_Y_] *
                                                      set_ptr->grid_1dim[_Z_] *
                                                      set_ptr->grid_1dim[_T_];
        set_ptr->move[_F_] = set_ptr->node_rank + set_ptr->move[_F_] *
                                                      set_ptr->grid_1dim[_Y_] *
                                                      set_ptr->grid_1dim[_Z_] *
                                                      set_ptr->grid_1dim[_T_];
        ncclSend(set_ptr->device_send_vec[_B_X_], set_ptr->lat_3dim_SC[_YZT_],
                 ncclDouble, set_ptr->move[_B_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_B_X_]);
        ncclRecv(set_ptr->device_recv_vec[_F_X_], set_ptr->lat_3dim_SC[_YZT_],
                 ncclDouble, set_ptr->move[_F_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_B_X_]);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_wards[_B_X_]));
        ncclSend(set_ptr->device_send_vec[_F_X_], set_ptr->lat_3dim_SC[_YZT_],
                 ncclDouble, set_ptr->move[_F_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_F_X_]);
        ncclRecv(set_ptr->device_recv_vec[_B_X_], set_ptr->lat_3dim_SC[_YZT_],
                 ncclDouble, set_ptr->move[_B_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_F_X_]);
      }
      if (set_ptr->grid_1dim[_Y_] != 1) {
        move_backward(set_ptr->move[_B_], set_ptr->grid_index_1dim[_Y_],
                      set_ptr->grid_1dim[_Y_]);
        move_forward(set_ptr->move[_F_], set_ptr->grid_index_1dim[_Y_],
                     set_ptr->grid_1dim[_Y_]);
        set_ptr->move[_B_] = set_ptr->node_rank + set_ptr->move[_B_] *
                                                      set_ptr->grid_1dim[_Z_] *
                                                      set_ptr->grid_1dim[_T_];
        set_ptr->move[_F_] = set_ptr->node_rank + set_ptr->move[_F_] *
                                                      set_ptr->grid_1dim[_Z_] *
                                                      set_ptr->grid_1dim[_T_];
        ncclSend(set_ptr->device_send_vec[_B_Y_], set_ptr->lat_3dim_SC[_XZT_],
                 ncclDouble, set_ptr->move[_B_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_B_Y_]);
        ncclRecv(set_ptr->device_recv_vec[_F_Y_], set_ptr->lat_3dim_SC[_XZT_],
                 ncclDouble, set_ptr->move[_F_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_B_Y_]);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_wards[_B_Y_]));
        ncclSend(set_ptr->device_send_vec[_F_Y_], set_ptr->lat_3dim_SC[_XZT_],
                 ncclDouble, set_ptr->move[_F_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_F_Y_]);
        ncclRecv(set_ptr->device_recv_vec[_B_Y_], set_ptr->lat_3dim_SC[_XZT_],
                 ncclDouble, set_ptr->move[_B_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_F_Y_]);
      }
      if (set_ptr->grid_1dim[_Z_] != 1) {
        move_backward(set_ptr->move[_B_], set_ptr->grid_index_1dim[_Z_],
                      set_ptr->grid_1dim[_Z_]);
        move_forward(set_ptr->move[_F_], set_ptr->grid_index_1dim[_Z_],
                     set_ptr->grid_1dim[_Z_]);
        set_ptr->move[_B_] =
            set_ptr->node_rank + set_ptr->move[_B_] * set_ptr->grid_1dim[_T_];
        set_ptr->move[_F_] =
            set_ptr->node_rank + set_ptr->move[_F_] * set_ptr->grid_1dim[_T_];
        ncclSend(set_ptr->device_send_vec[_B_Z_], set_ptr->lat_3dim_SC[_XYT_],
                 ncclDouble, set_ptr->move[_B_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_B_Z_]);
        ncclRecv(set_ptr->device_recv_vec[_F_Z_], set_ptr->lat_3dim_SC[_XYT_],
                 ncclDouble, set_ptr->move[_F_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_B_Z_]);
        cudaStreamSynchronize(set_ptr->stream_wards[_B_Z_]);
        ncclSend(set_ptr->device_send_vec[_F_Z_], set_ptr->lat_3dim_SC[_XYT_],
                 ncclDouble, set_ptr->move[_F_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_F_Z_]);
        ncclRecv(set_ptr->device_recv_vec[_B_Z_], set_ptr->lat_3dim_SC[_XYT_],
                 ncclDouble, set_ptr->move[_B_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_F_Z_]);
      }
      if (set_ptr->grid_1dim[_T_] != 1) {
        move_backward(set_ptr->move[_B_], set_ptr->grid_index_1dim[_T_],
                      set_ptr->grid_1dim[_T_]);
        move_forward(set_ptr->move[_F_], set_ptr->grid_index_1dim[_T_],
                     set_ptr->grid_1dim[_T_]);
        set_ptr->move[_B_] = set_ptr->node_rank + set_ptr->move[_B_];
        set_ptr->move[_F_] = set_ptr->node_rank + set_ptr->move[_F_];
        ncclSend(set_ptr->device_send_vec[_B_T_], set_ptr->lat_3dim_SC[_XYZ_],
                 ncclDouble, set_ptr->move[_B_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_B_T_]);
        ncclRecv(set_ptr->device_recv_vec[_F_T_], set_ptr->lat_3dim_SC[_XYZ_],
                 ncclDouble, set_ptr->move[_F_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_B_T_]);
        cudaStreamSynchronize(set_ptr->stream_wards[_B_T_]);
        ncclSend(set_ptr->device_send_vec[_F_T_], set_ptr->lat_3dim_SC[_XYZ_],
                 ncclDouble, set_ptr->move[_F_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_F_T_]);
        ncclRecv(set_ptr->device_recv_vec[_B_T_], set_ptr->lat_3dim_SC[_XYZ_],
                 ncclDouble, set_ptr->move[_B_], set_ptr->nccl_comm,
                 set_ptr->stream_wards[_F_T_]);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      ncclGroupEnd();
    }
    {
      if (set_ptr->grid_1dim[_X_] != 1) {
        wilson_dslash_x_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                               set_ptr->stream_wards[_F_X_]>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_recv_vec[_B_X_], set_ptr->device_recv_vec[_F_X_]);
      } else {
        wilson_dslash_x_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                               set_ptr->stream_wards[_B_X_]>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_send_vec[_F_X_], set_ptr->device_send_vec[_B_X_]);
      }
      if (set_ptr->grid_1dim[_Y_] != 1) {
        wilson_dslash_y_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                               set_ptr->stream_wards[_F_Y_]>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_recv_vec[_B_Y_], set_ptr->device_recv_vec[_F_Y_]);
      } else {
        wilson_dslash_y_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                               set_ptr->stream_wards[_B_Y_]>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_send_vec[_F_Y_], set_ptr->device_send_vec[_B_Y_]);
      }
      if (set_ptr->grid_1dim[_Z_] != 1) {
        wilson_dslash_z_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                               set_ptr->stream_wards[_F_Z_]>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_recv_vec[_B_Z_], set_ptr->device_recv_vec[_F_Z_]);
      } else {
        wilson_dslash_z_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                               set_ptr->stream_wards[_B_Z_]>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_send_vec[_F_Z_], set_ptr->device_send_vec[_B_Z_]);
      }
      if (set_ptr->grid_1dim[_T_] != 1) {
        wilson_dslash_t_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                               set_ptr->stream_wards[_F_T_]>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_recv_vec[_B_T_], set_ptr->device_recv_vec[_F_T_]);
      } else {
        wilson_dslash_t_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                               set_ptr->stream_wards[_B_T_]>>>(
            gauge, fermion_out, set_ptr->device_xyztsc, parity,
            set_ptr->device_send_vec[_F_T_], set_ptr->device_send_vec[_B_T_]);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_wards[_F_X_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_wards[_F_Y_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_wards[_F_Z_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_wards[_F_T_]));
    }
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