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
  void run(void *fermion_out, void *fermion_in, void *gauge, int parity) {
    {
      // x comm
      wilson_dslash_x_send<<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                             set_ptr->stream_dims[_X_]>>>(
          gauge, fermion_in, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_X_], set_ptr->device_send_vec[_F_X_]);
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
    {
      // y comm
      wilson_dslash_y_send<<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                             set_ptr->stream_dims[_Y_]>>>(
          gauge, fermion_in, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_Y_], set_ptr->device_send_vec[_F_Y_]);
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
    {
      // z comm
      wilson_dslash_z_send<<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                             set_ptr->stream_dims[_Z_]>>>(
          gauge, fermion_in, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_Z_], set_ptr->device_send_vec[_F_Z_]);
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
    {
      // t comm
      wilson_dslash_t_send<<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                             set_ptr->stream_dims[_T_]>>>(
          gauge, fermion_in, set_ptr->device_xyztsc, parity,
          set_ptr->device_send_vec[_B_T_], set_ptr->device_send_vec[_F_T_]);
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
    { // inside compute part
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      wilson_dslash_inside<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                             set_ptr->stream>>>(gauge, fermion_in, fermion_out,
                                                set_ptr->device_xyztsc, parity);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
    }
    {
      // edge send part
      wilson_dslash_x_recv<<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                             set_ptr->stream>>>(
          gauge, fermion_out, set_ptr->device_xyztsc, parity,
          set_ptr->device_recv_vec[_B_X_], set_ptr->device_recv_vec[_F_X_]);
      wilson_dslash_y_recv<<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                             set_ptr->stream>>>(
          gauge, fermion_out, set_ptr->device_xyztsc, parity,
          set_ptr->device_recv_vec[_B_Y_], set_ptr->device_recv_vec[_F_Y_]);
      wilson_dslash_z_recv<<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                             set_ptr->stream>>>(
          gauge, fermion_out, set_ptr->device_xyztsc, parity,
          set_ptr->device_recv_vec[_B_Z_], set_ptr->device_recv_vec[_F_Z_]);
      wilson_dslash_t_recv<<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                             set_ptr->stream>>>(
          gauge, fermion_out, set_ptr->device_xyztsc, parity,
          set_ptr->device_recv_vec[_B_T_], set_ptr->device_recv_vec[_F_T_]);
    }
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
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