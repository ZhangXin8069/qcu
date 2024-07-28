#ifndef _LATTICE_DSLASH_H
#define _LATTICE_DSLASH_H
#include "./lattice_set.h"
__global__ void wilson_dslash(void *device_U, void *device_src,
                              void *device_dest, int device_lat_x,
                              const int device_lat_y, const int device_lat_z,
                              const int device_lat_t, const int device_parity);

__global__ void wilson_dslash_clear_dest(void *device_dest, int device_lat_x,
                                         const int device_lat_y,
                                         const int device_lat_z);

__global__ void
wilson_dslash_x_send(void *device_U, void *device_src, void *device_dest,
                     int device_lat_x, const int device_lat_y,
                     const int device_lat_z, const int device_lat_t,
                     const int device_parity, void *device_b_x_send_vec,
                     void *device_f_x_send_vec);

__global__ void
wilson_dslash_x_recv(void *device_U, void *device_dest, int device_lat_x,
                     const int device_lat_y, const int device_lat_z,
                     const int device_lat_t, const int device_parity,
                     void *device_b_x_recv_vec, void *device_f_x_recv_vec);

__global__ void
wilson_dslash_y_send(void *device_U, void *device_src, void *device_dest,
                     int device_lat_x, const int device_lat_y,
                     const int device_lat_z, const int device_lat_t,
                     const int device_parity, void *device_b_y_send_vec,
                     void *device_f_y_send_vec);

__global__ void
wilson_dslash_y_recv(void *device_U, void *device_dest, int device_lat_x,
                     const int device_lat_y, const int device_lat_z,
                     const int device_lat_t, const int device_parity,
                     void *device_b_y_recv_vec, void *device_f_y_recv_vec);

__global__ void
wilson_dslash_z_send(void *device_U, void *device_src, void *device_dest,
                     int device_lat_x, const int device_lat_y,
                     const int device_lat_z, const int device_lat_t,
                     const int device_parity, void *device_b_z_send_vec,
                     void *device_f_z_send_vec);

__global__ void
wilson_dslash_z_recv(void *device_U, void *device_dest, int device_lat_x,
                     const int device_lat_y, const int device_lat_z,
                     const int device_lat_t, const int device_parity,
                     void *device_b_z_recv_vec, void *device_f_z_recv_vec);

__global__ void
wilson_dslash_t_send(void *device_U, void *device_src, void *device_dest,
                     int device_lat_x, const int device_lat_y,
                     const int device_lat_z, const int device_lat_t,
                     const int device_parity, void *device_b_t_send_vec,
                     void *device_f_t_send_vec);

__global__ void
wilson_dslash_t_recv(void *device_U, void *device_dest, int device_lat_x,
                     const int device_lat_y, const int device_lat_z,
                     const int device_lat_t, const int device_parity,
                     void *device_b_t_recv_vec, void *device_f_t_recv_vec);

__global__ void test_wilson_dslash(void *device_U, void *device_src,
                                   void *device_dest, int device_lat_x,
                                   const int device_lat_y,
                                   const int device_lat_z,
                                   const int device_lat_t,
                                   const int device_parity);

__global__ void make_clover(void *device_U, void *device_clover,
                            int device_lat_x, const int device_lat_y,
                            const int device_lat_z, const int device_lat_t,
                            const int device_parity);

__global__ void inverse_clover(void *device_clover, int device_lat_x,
                               const int device_lat_y, const int device_lat_z);

__global__ void give_clover(void *device_clover, void *device_dest,
                            int device_lat_x, const int device_lat_y,
                            const int device_lat_z);

struct LatticeWilsonDslash {
  LatticeSet *set_ptr;
  void give(LatticeSet *_set_ptr) { set_ptr = _set_ptr; }
  void run(void *fermion_out, void *fermion_in, void *gauge, int parity) {
    checkCudaErrors(cudaDeviceSynchronize());
    ncclGroupStart();
    wilson_dslash_clear_dest<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                               set_ptr->qcu_stream>>>(
        fermion_out, set_ptr->lat_1dim[_X_], set_ptr->lat_1dim[_Y_],
        set_ptr->lat_1dim[_Z_]);
    cudaStreamSynchronize(set_ptr->qcu_stream);
    wilson_dslash_x_send<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                           set_ptr->qcu_stream>>>(
        gauge, fermion_in, fermion_out, set_ptr->lat_1dim[_X_],
        set_ptr->lat_1dim[_Y_], set_ptr->lat_1dim[_Z_], set_ptr->lat_1dim[_T_],
        parity, set_ptr->device_send_vec[_B_X_],
        set_ptr->device_send_vec[_F_X_]);
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
      cudaStreamSynchronize(set_ptr->qcu_stream);
      ncclSend(set_ptr->device_send_vec[_B_X_], set_ptr->lat_3dim12[_YZT_],
               ncclDouble, set_ptr->move[_B_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
      ncclRecv(set_ptr->device_recv_vec[_F_X_], set_ptr->lat_3dim12[_YZT_],
               ncclDouble, set_ptr->move[_F_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
      ncclSend(set_ptr->device_send_vec[_F_X_], set_ptr->lat_3dim12[_YZT_],
               ncclDouble, set_ptr->move[_F_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
      ncclRecv(set_ptr->device_recv_vec[_B_X_], set_ptr->lat_3dim12[_YZT_],
               ncclDouble, set_ptr->move[_B_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
    }
    wilson_dslash_y_send<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                           set_ptr->qcu_stream>>>(
        gauge, fermion_in, fermion_out, set_ptr->lat_1dim[_X_],
        set_ptr->lat_1dim[_Y_], set_ptr->lat_1dim[_Z_], set_ptr->lat_1dim[_T_],
        parity, set_ptr->device_send_vec[_B_Y_],
        set_ptr->device_send_vec[_F_Y_]);
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
      cudaStreamSynchronize(set_ptr->qcu_stream);
      ncclSend(set_ptr->device_send_vec[_B_Y_], set_ptr->lat_3dim12[_XZT_],
               ncclDouble, set_ptr->move[_B_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
      ncclRecv(set_ptr->device_recv_vec[_F_Y_], set_ptr->lat_3dim12[_XZT_],
               ncclDouble, set_ptr->move[_F_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
      ncclSend(set_ptr->device_send_vec[_F_Y_], set_ptr->lat_3dim12[_XZT_],
               ncclDouble, set_ptr->move[_F_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
      ncclRecv(set_ptr->device_recv_vec[_B_Y_], set_ptr->lat_3dim12[_XZT_],
               ncclDouble, set_ptr->move[_B_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
    }
    wilson_dslash_z_send<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                           set_ptr->qcu_stream>>>(
        gauge, fermion_in, fermion_out, set_ptr->lat_1dim[_X_],
        set_ptr->lat_1dim[_Y_], set_ptr->lat_1dim[_Z_], set_ptr->lat_1dim[_T_],
        parity, set_ptr->device_send_vec[_B_Z_],
        set_ptr->device_send_vec[_F_Z_]);
    if (set_ptr->grid_1dim[_Z_] != 1) {
      move_backward(set_ptr->move[_B_], set_ptr->grid_index_1dim[_Z_],
                    set_ptr->grid_1dim[_Z_]);
      move_forward(set_ptr->move[_F_], set_ptr->grid_index_1dim[_Z_],
                   set_ptr->grid_1dim[_Z_]);
      set_ptr->move[_B_] =
          set_ptr->node_rank + set_ptr->move[_B_] * set_ptr->grid_1dim[_T_];
      set_ptr->move[_F_] =
          set_ptr->node_rank + set_ptr->move[_F_] * set_ptr->grid_1dim[_T_];
      cudaStreamSynchronize(set_ptr->qcu_stream);
      ncclSend(set_ptr->device_send_vec[_B_Z_], set_ptr->lat_3dim12[_XYT_],
               ncclDouble, set_ptr->move[_B_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
      ncclRecv(set_ptr->device_recv_vec[_F_Z_], set_ptr->lat_3dim12[_XYT_],
               ncclDouble, set_ptr->move[_F_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
      ncclSend(set_ptr->device_send_vec[_F_Z_], set_ptr->lat_3dim12[_XYT_],
               ncclDouble, set_ptr->move[_F_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
      ncclRecv(set_ptr->device_recv_vec[_B_Z_], set_ptr->lat_3dim12[_XYT_],
               ncclDouble, set_ptr->move[_B_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
    }
    wilson_dslash_t_send<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                           set_ptr->qcu_stream>>>(
        gauge, fermion_in, fermion_out, set_ptr->lat_1dim[_X_],
        set_ptr->lat_1dim[_Y_], set_ptr->lat_1dim[_Z_], set_ptr->lat_1dim[_T_],
        parity, set_ptr->device_send_vec[_B_T_],
        set_ptr->device_send_vec[_F_T_]);
    if (set_ptr->grid_1dim[_T_] != 1) {
      move_backward(set_ptr->move[_B_], set_ptr->grid_index_1dim[_T_],
                    set_ptr->grid_1dim[_T_]);
      move_forward(set_ptr->move[_F_], set_ptr->grid_index_1dim[_T_],
                   set_ptr->grid_1dim[_T_]);
      set_ptr->move[_B_] = set_ptr->node_rank + set_ptr->move[_B_];
      set_ptr->move[_F_] = set_ptr->node_rank + set_ptr->move[_F_];
      cudaStreamSynchronize(set_ptr->qcu_stream);
      ncclSend(set_ptr->device_send_vec[_B_T_], set_ptr->lat_3dim12[_XYZ_],
               ncclDouble, set_ptr->move[_B_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
      ncclRecv(set_ptr->device_recv_vec[_F_T_], set_ptr->lat_3dim12[_XYZ_],
               ncclDouble, set_ptr->move[_F_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
      ncclSend(set_ptr->device_send_vec[_F_T_], set_ptr->lat_3dim12[_XYZ_],
               ncclDouble, set_ptr->move[_F_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
      ncclRecv(set_ptr->device_recv_vec[_B_T_], set_ptr->lat_3dim12[_XYZ_],
               ncclDouble, set_ptr->move[_B_], set_ptr->qcu_nccl_comm,
               set_ptr->qcu_stream);
    }
    checkCudaErrors(cudaStreamSynchronize(set_ptr->qcu_stream));
    ncclGroupEnd();
    if (set_ptr->grid_1dim[_X_] != 1) {
      wilson_dslash_x_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                             set_ptr->qcu_stream>>>(
          gauge, fermion_out, set_ptr->lat_1dim[_X_], set_ptr->lat_1dim[_Y_],
          set_ptr->lat_1dim[_Z_], set_ptr->lat_1dim[_T_], parity,
          set_ptr->device_recv_vec[_B_X_], set_ptr->device_recv_vec[_F_X_]);
    } else {
      wilson_dslash_x_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                             set_ptr->qcu_stream>>>(
          gauge, fermion_out, set_ptr->lat_1dim[_X_], set_ptr->lat_1dim[_Y_],
          set_ptr->lat_1dim[_Z_], set_ptr->lat_1dim[_T_], parity,
          set_ptr->device_send_vec[_F_X_], set_ptr->device_send_vec[_B_X_]);
    }
    if (set_ptr->grid_1dim[_Y_] != 1) {
      wilson_dslash_y_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                             set_ptr->qcu_stream>>>(
          gauge, fermion_out, set_ptr->lat_1dim[_X_], set_ptr->lat_1dim[_Y_],
          set_ptr->lat_1dim[_Z_], set_ptr->lat_1dim[_T_], parity,
          set_ptr->device_recv_vec[_B_Y_], set_ptr->device_recv_vec[_F_Y_]);
    } else {
      wilson_dslash_y_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                             set_ptr->qcu_stream>>>(
          gauge, fermion_out, set_ptr->lat_1dim[_X_], set_ptr->lat_1dim[_Y_],
          set_ptr->lat_1dim[_Z_], set_ptr->lat_1dim[_T_], parity,
          set_ptr->device_send_vec[_F_Y_], set_ptr->device_send_vec[_B_Y_]);
    }
    if (set_ptr->grid_1dim[_Z_] != 1) {
      wilson_dslash_z_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                             set_ptr->qcu_stream>>>(
          gauge, fermion_out, set_ptr->lat_1dim[_X_], set_ptr->lat_1dim[_Y_],
          set_ptr->lat_1dim[_Z_], set_ptr->lat_1dim[_T_], parity,
          set_ptr->device_recv_vec[_B_Z_], set_ptr->device_recv_vec[_F_Z_]);
    } else {
      wilson_dslash_z_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                             set_ptr->qcu_stream>>>(
          gauge, fermion_out, set_ptr->lat_1dim[_X_], set_ptr->lat_1dim[_Y_],
          set_ptr->lat_1dim[_Z_], set_ptr->lat_1dim[_T_], parity,
          set_ptr->device_send_vec[_F_Z_], set_ptr->device_send_vec[_B_Z_]);
    }
    if (set_ptr->grid_1dim[_T_] != 1) {
      wilson_dslash_t_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                             set_ptr->qcu_stream>>>(
          gauge, fermion_out, set_ptr->lat_1dim[_X_], set_ptr->lat_1dim[_Y_],
          set_ptr->lat_1dim[_Z_], set_ptr->lat_1dim[_T_], parity,
          set_ptr->device_recv_vec[_B_T_], set_ptr->device_recv_vec[_F_T_]);
    } else {
      wilson_dslash_t_recv<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                             set_ptr->qcu_stream>>>(
          gauge, fermion_out, set_ptr->lat_1dim[_X_], set_ptr->lat_1dim[_Y_],
          set_ptr->lat_1dim[_Z_], set_ptr->lat_1dim[_T_], parity,
          set_ptr->device_send_vec[_F_T_], set_ptr->device_send_vec[_B_T_]);
    }
    checkCudaErrors(cudaStreamSynchronize(set_ptr->qcu_stream));
    checkCudaErrors(cudaDeviceSynchronize());
  }

  void run_eo(void *fermion_out, void *fermion_in, void *gauge) {
    run(fermion_out, fermion_in, gauge, _EVEN_);
  }

  void run_oe(void *fermion_out, void *fermion_in, void *gauge) {
    run(fermion_out, fermion_in, gauge, _ODD_);
  }
};

#endif