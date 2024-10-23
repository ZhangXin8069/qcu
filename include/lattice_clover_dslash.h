#ifndef _LATTICE_CLOVER_DSLASH_H
#define _LATTICE_CLOVER_DSLASH_H
#include "./clover_dslash.h"
#include "./define.h"
#include "./lattice_set.h"
struct LatticeCloverDslash {
  LatticeSet *set_ptr;
  cudaError_t err;
  void *clover;
  void give(LatticeSet *_set_ptr) { set_ptr = _set_ptr; }
  void init() {
    checkCudaErrors(cudaMallocAsync(
        &clover, (set_ptr->lat_4dim * _QCU_LAT_SCSC_) * sizeof(LatticeComplex),
        set_ptr->stream));
  }
  void _make(void *gauge) {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_X_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_Y_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_Z_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_T_]));
    // edge send part
    {
      // u_1dim_send
      pick_up_u_x<<<set_ptr->gridDim_3dim[_QCU_X_], set_ptr->blockDim, 0,
                    set_ptr->stream_dims[_QCU_X_]>>>(
          gauge, set_ptr->device_params, set_ptr->device_u_1dim_send_vec[_QCU_B_X_],
          set_ptr->device_u_1dim_send_vec[_QCU_F_X_]);
      pick_up_u_y<<<set_ptr->gridDim_3dim[_QCU_Y_], set_ptr->blockDim, 0,
                    set_ptr->stream_dims[_QCU_Y_]>>>(
          gauge, set_ptr->device_params, set_ptr->device_u_1dim_send_vec[_QCU_B_Y_],
          set_ptr->device_u_1dim_send_vec[_QCU_F_Y_]);
      pick_up_u_z<<<set_ptr->gridDim_3dim[_QCU_Z_], set_ptr->blockDim, 0,
                    set_ptr->stream_dims[_QCU_Z_]>>>(
          gauge, set_ptr->device_params, set_ptr->device_u_1dim_send_vec[_QCU_B_Z_],
          set_ptr->device_u_1dim_send_vec[_QCU_F_Z_]);
      pick_up_u_t<<<set_ptr->gridDim_3dim[_QCU_T_], set_ptr->blockDim, 0,
                    set_ptr->stream_dims[_QCU_T_]>>>(
          gauge, set_ptr->device_params, set_ptr->device_u_1dim_send_vec[_QCU_B_T_],
          set_ptr->device_u_1dim_send_vec[_QCU_F_T_]);
    }
    {
      // u_2dim_send
      pick_up_u_xy<<<set_ptr->gridDim_2dim[_QCU_2DIM_ - 1 - _QCU_XY_],
                     set_ptr->blockDim, 0, set_ptr->stream>>>(
          gauge, set_ptr->device_params,
          set_ptr->device_u_2dim_send_vec[_QCU_B_X_B_Y_],
          set_ptr->device_u_2dim_send_vec[_QCU_F_X_B_Y_],
          set_ptr->device_u_2dim_send_vec[_QCU_B_X_F_Y_],
          set_ptr->device_u_2dim_send_vec[_QCU_F_X_F_Y_]);
      pick_up_u_xz<<<set_ptr->gridDim_2dim[_QCU_2DIM_ - 1 - _QCU_XZ_],
                     set_ptr->blockDim, 0, set_ptr->stream>>>(
          gauge, set_ptr->device_params,
          set_ptr->device_u_2dim_send_vec[_QCU_B_X_B_Z_],
          set_ptr->device_u_2dim_send_vec[_QCU_F_X_B_Z_],
          set_ptr->device_u_2dim_send_vec[_QCU_B_X_F_Z_],
          set_ptr->device_u_2dim_send_vec[_QCU_F_X_F_Z_]);
      pick_up_u_xt<<<set_ptr->gridDim_2dim[_QCU_2DIM_ - 1 - _QCU_XT_],
                     set_ptr->blockDim, 0, set_ptr->stream>>>(
          gauge, set_ptr->device_params,
          set_ptr->device_u_2dim_send_vec[_QCU_B_X_B_T_],
          set_ptr->device_u_2dim_send_vec[_QCU_F_X_B_T_],
          set_ptr->device_u_2dim_send_vec[_QCU_B_X_F_T_],
          set_ptr->device_u_2dim_send_vec[_QCU_F_X_F_T_]);
      pick_up_u_yz<<<set_ptr->gridDim_2dim[_QCU_2DIM_ - 1 - _QCU_YZ_],
                     set_ptr->blockDim, 0, set_ptr->stream>>>(
          gauge, set_ptr->device_params,
          set_ptr->device_u_2dim_send_vec[_QCU_B_Y_B_Z_],
          set_ptr->device_u_2dim_send_vec[_QCU_F_Y_B_Z_],
          set_ptr->device_u_2dim_send_vec[_QCU_B_Y_F_Z_],
          set_ptr->device_u_2dim_send_vec[_QCU_F_Y_F_Z_]);
      pick_up_u_yt<<<set_ptr->gridDim_2dim[_QCU_2DIM_ - 1 - _QCU_YT_],
                     set_ptr->blockDim, 0, set_ptr->stream>>>(
          gauge, set_ptr->device_params,
          set_ptr->device_u_2dim_send_vec[_QCU_B_Y_B_T_],
          set_ptr->device_u_2dim_send_vec[_QCU_F_Y_B_T_],
          set_ptr->device_u_2dim_send_vec[_QCU_B_Y_F_T_],
          set_ptr->device_u_2dim_send_vec[_QCU_F_Y_F_T_]);
      pick_up_u_zt<<<set_ptr->gridDim_2dim[_QCU_2DIM_ - 1 - _QCU_ZT_],
                     set_ptr->blockDim, 0, set_ptr->stream>>>(
          gauge, set_ptr->device_params,
          set_ptr->device_u_2dim_send_vec[_QCU_B_Z_B_T_],
          set_ptr->device_u_2dim_send_vec[_QCU_F_Z_B_T_],
          set_ptr->device_u_2dim_send_vec[_QCU_B_Z_F_T_],
          set_ptr->device_u_2dim_send_vec[_QCU_F_Z_F_T_]);
    }
    // edge comm part
    {
      // u_1dim_comm
      {
        // x edge part comm
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_X_]));
        ncclGroupStart();
        ncclSend(set_ptr->device_u_1dim_send_vec[_QCU_B_X_],
                 set_ptr->lat_3dim[_QCU_X_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_B_X_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_X_]);
        ncclRecv(set_ptr->device_u_1dim_recv_vec[_QCU_F_X_],
                 set_ptr->lat_3dim[_QCU_X_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_F_X_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_X_]);
        ncclGroupEnd();
        ncclGroupStart();
        ncclSend(set_ptr->device_u_1dim_send_vec[_QCU_F_X_],
                 set_ptr->lat_3dim[_QCU_X_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_F_X_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_X_]);
        ncclRecv(set_ptr->device_u_1dim_recv_vec[_QCU_B_X_],
                 set_ptr->lat_3dim[_QCU_X_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_B_X_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_X_]);
        ncclGroupEnd();
      }
      {
        // y edge part comm
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_Y_]));
        ncclGroupStart();
        ncclSend(set_ptr->device_u_1dim_send_vec[_QCU_B_Y_],
                 set_ptr->lat_3dim[_QCU_Y_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_B_Y_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_Y_]);
        ncclRecv(set_ptr->device_u_1dim_recv_vec[_QCU_F_Y_],
                 set_ptr->lat_3dim[_QCU_Y_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_F_Y_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_Y_]);
        ncclGroupEnd();
        ncclGroupStart();
        ncclSend(set_ptr->device_u_1dim_send_vec[_QCU_F_Y_],
                 set_ptr->lat_3dim[_QCU_Y_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_F_Y_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_Y_]);
        ncclRecv(set_ptr->device_u_1dim_recv_vec[_QCU_B_Y_],
                 set_ptr->lat_3dim[_QCU_Y_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_B_Y_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_Y_]);
        ncclGroupEnd();
      }
      {
        // z edge part comm
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_Z_]));
        ncclGroupStart();
        ncclSend(set_ptr->device_u_1dim_send_vec[_QCU_B_Z_],
                 set_ptr->lat_3dim[_QCU_Z_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_B_Z_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_Z_]);
        ncclRecv(set_ptr->device_u_1dim_recv_vec[_QCU_F_Z_],
                 set_ptr->lat_3dim[_QCU_Z_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_F_Z_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_Z_]);
        ncclGroupEnd();
        ncclGroupStart();
        ncclSend(set_ptr->device_u_1dim_send_vec[_QCU_F_Z_],
                 set_ptr->lat_3dim[_QCU_Z_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_F_Z_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_Z_]);
        ncclRecv(set_ptr->device_u_1dim_recv_vec[_QCU_B_Z_],
                 set_ptr->lat_3dim[_QCU_Z_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_B_Z_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_Z_]);
        ncclGroupEnd();
      }
      {
        // t edge part comm
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_T_]));
        ncclGroupStart();
        ncclSend(set_ptr->device_u_1dim_send_vec[_QCU_B_T_],
                 set_ptr->lat_3dim[_QCU_T_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_B_T_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_T_]);
        ncclRecv(set_ptr->device_u_1dim_recv_vec[_QCU_F_T_],
                 set_ptr->lat_3dim[_QCU_T_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_F_T_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_T_]);
        ncclGroupEnd();
        ncclGroupStart();
        ncclSend(set_ptr->device_u_1dim_send_vec[_QCU_F_T_],
                 set_ptr->lat_3dim[_QCU_T_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_F_T_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_T_]);
        ncclRecv(set_ptr->device_u_1dim_recv_vec[_QCU_B_T_],
                 set_ptr->lat_3dim[_QCU_T_] * _QCU_LAT_PDCC_ * _QCU_REAL_IMAG_, ncclDouble,
                 set_ptr->move_wards[_QCU_B_T_], set_ptr->nccl_comm,
                 set_ptr->stream_dims[_QCU_T_]);
        ncclGroupEnd();
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      {
        // u_2dim_comm
        {
          // xy edge part comm
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_B_X_B_Y_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XY_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BX_BY_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_a_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_F_X_F_Y_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XY_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FX_FY_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_a_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_F_X_B_Y_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XY_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FX_BY_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_b_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_B_X_F_Y_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XY_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BX_FY_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_b_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_B_X_F_Y_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XY_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BX_FY_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_c_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_F_X_B_Y_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XY_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FX_BY_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_c_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_F_X_F_Y_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XY_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FX_FY_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_d_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_B_X_B_Y_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XY_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BX_BY_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_d_]);
          ncclGroupEnd();
        }
        {
          // xz edge part comm
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_B_X_B_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BX_BZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_a_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_F_X_F_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FX_FZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_a_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_F_X_B_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FX_BZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_b_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_B_X_F_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BX_FZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_b_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_B_X_F_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BX_FZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_c_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_F_X_B_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FX_BZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_c_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_F_X_F_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FX_FZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_d_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_B_X_B_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BX_BZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_d_]);
          ncclGroupEnd();
        }
        {
          // xt edge part comm
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_B_X_B_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BX_BT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_a_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_F_X_F_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FX_FT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_a_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_F_X_B_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FX_BT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_b_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_B_X_F_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BX_FT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_b_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_B_X_F_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BX_FT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_c_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_F_X_B_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FX_BT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_c_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_F_X_F_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FX_FT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_d_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_B_X_B_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_XT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BX_BT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_d_]);
          ncclGroupEnd();
        }
        {
          // yz edge part comm
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_B_Y_B_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BY_BZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_a_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_F_Y_F_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FY_FZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_a_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_F_Y_B_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FY_BZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_b_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_B_Y_F_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BY_FZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_b_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_B_Y_F_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BY_FZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_c_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_F_Y_B_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FY_BZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_c_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_F_Y_F_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FY_FZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_d_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_B_Y_B_Z_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YZ_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BY_BZ_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_d_]);
          ncclGroupEnd();
        }
        {
          // yt edge part comm
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_B_Y_B_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BY_BT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_a_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_F_Y_F_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FY_FT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_a_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_F_Y_B_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FY_BT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_b_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_B_Y_F_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BY_FT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_b_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_B_Y_F_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BY_FT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_c_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_F_Y_B_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FY_BT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_c_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_F_Y_F_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FY_FT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_d_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_B_Y_B_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_YT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BY_BT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_d_]);
          ncclGroupEnd();
        }
        {
          // zt edge part comm
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_B_Z_B_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_ZT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BZ_BT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_a_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_F_Z_F_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_ZT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FZ_FT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_a_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_F_Z_B_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_ZT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FZ_BT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_b_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_B_Z_F_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_ZT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BZ_FT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_b_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_B_Z_F_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_ZT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BZ_FT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_c_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_F_Z_B_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_ZT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FZ_BT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_c_]);
          ncclGroupEnd();
          ncclGroupStart();
          ncclSend(set_ptr->device_u_2dim_send_vec[_QCU_F_Z_F_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_ZT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_FZ_FT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_d_]);
          ncclRecv(set_ptr->device_u_2dim_recv_vec[_QCU_B_Z_B_T_],
                   set_ptr->lat_2dim[_QCU_2DIM_ - 1 - _QCU_ZT_] * _QCU_LAT_PDCC_ *
                       _QCU_REAL_IMAG_,
                   ncclDouble, set_ptr->move_wards[_QCU_BZ_BT_], set_ptr->nccl_comm,
                   set_ptr->stream_dims[_qcu_d_]);
          ncclGroupEnd();
        }
      }
    }
    // edge recv part
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_X_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_Y_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_Z_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_T_]));
    make_clover_all<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                      set_ptr->stream>>>(
        gauge, clover, set_ptr->device_params,
        set_ptr->device_u_1dim_recv_vec[_QCU_B_X_],
        set_ptr->device_u_1dim_recv_vec[_QCU_F_X_],
        set_ptr->device_u_1dim_recv_vec[_QCU_B_Y_],
        set_ptr->device_u_1dim_recv_vec[_QCU_F_Y_],
        set_ptr->device_u_1dim_recv_vec[_QCU_B_Z_],
        set_ptr->device_u_1dim_recv_vec[_QCU_F_Z_],
        set_ptr->device_u_1dim_recv_vec[_QCU_B_T_],
        set_ptr->device_u_1dim_recv_vec[_QCU_F_T_],
        set_ptr->device_u_2dim_recv_vec[_QCU_B_X_B_Y_],
        set_ptr->device_u_2dim_recv_vec[_QCU_F_X_B_Y_],
        set_ptr->device_u_2dim_recv_vec[_QCU_B_X_F_Y_],
        set_ptr->device_u_2dim_recv_vec[_QCU_F_X_F_Y_],
        set_ptr->device_u_2dim_recv_vec[_QCU_B_X_B_Z_],
        set_ptr->device_u_2dim_recv_vec[_QCU_F_X_B_Z_],
        set_ptr->device_u_2dim_recv_vec[_QCU_B_X_F_Z_],
        set_ptr->device_u_2dim_recv_vec[_QCU_F_X_F_Z_],
        set_ptr->device_u_2dim_recv_vec[_QCU_B_X_B_T_],
        set_ptr->device_u_2dim_recv_vec[_QCU_F_X_B_T_],
        set_ptr->device_u_2dim_recv_vec[_QCU_B_X_F_T_],
        set_ptr->device_u_2dim_recv_vec[_QCU_F_X_F_T_],
        set_ptr->device_u_2dim_recv_vec[_QCU_B_Y_B_Z_],
        set_ptr->device_u_2dim_recv_vec[_QCU_F_Y_B_Z_],
        set_ptr->device_u_2dim_recv_vec[_QCU_B_Y_F_Z_],
        set_ptr->device_u_2dim_recv_vec[_QCU_F_Y_F_Z_],
        set_ptr->device_u_2dim_recv_vec[_QCU_B_Y_B_T_],
        set_ptr->device_u_2dim_recv_vec[_QCU_F_Y_B_T_],
        set_ptr->device_u_2dim_recv_vec[_QCU_B_Y_F_T_],
        set_ptr->device_u_2dim_recv_vec[_QCU_F_Y_F_T_],
        set_ptr->device_u_2dim_recv_vec[_QCU_B_Z_B_T_],
        set_ptr->device_u_2dim_recv_vec[_QCU_F_Z_B_T_],
        set_ptr->device_u_2dim_recv_vec[_QCU_B_Z_F_T_],
        set_ptr->device_u_2dim_recv_vec[_QCU_F_Z_F_T_]);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_X_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_Y_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_Z_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_QCU_T_]));
  }
  void make(void *gauge) {
    // make clover
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    auto start = std::chrono::high_resolution_clock::now();
    _make(gauge);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    err = cudaGetLastError();
    checkCudaErrors(err);
    printf("make clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  void inverse() {
    // inverse clover
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    auto start = std::chrono::high_resolution_clock::now();
    inverse_clover<<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        clover, set_ptr->device_params);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    err = cudaGetLastError();
    checkCudaErrors(err);
    printf(
        "inverse clover total time: (without malloc free memcpy) :%.9lf sec\n ",
        double(duration) / 1e9);
  }
  void give(void *fermion_out) {
    // give clover
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    auto start = std::chrono::high_resolution_clock::now();
    give_clover<<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        clover, fermion_out, set_ptr->device_params);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    err = cudaGetLastError();
    checkCudaErrors(err);
    printf("give clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  void end() {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaFreeAsync(clover, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};
#endif