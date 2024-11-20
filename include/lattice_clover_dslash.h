#ifndef _LATTICE_CLOVER_DSLASH_H
#define _LATTICE_CLOVER_DSLASH_H
#include "./clover_dslash.h"
#include "./define.h"
#include "./lattice_set.h"
namespace qcu
{
    struct LatticeCloverDslash
    {
        LatticeSet *set_ptr;
        cudaError_t err;
        void *clover;
        void give(LatticeSet *_set_ptr) { set_ptr = _set_ptr; }
        void init()
        {
            checkCudaErrors(cudaMallocAsync(
                &clover, (set_ptr->lat_4dim * _LAT_SCSC_) * sizeof(LatticeComplex),
                set_ptr->stream));
        }
        void _make_nccl(void *gauge)
        {
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
            // edge send part
            {
                // u_1dim_send
                pick_up_u_x<<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                              set_ptr->stream_dims[_X_]>>>(
                    gauge, set_ptr->device_params, set_ptr->device_u_1dim_send_vec[_B_X_],
                    set_ptr->device_u_1dim_send_vec[_F_X_]);
                pick_up_u_y<<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                              set_ptr->stream_dims[_Y_]>>>(
                    gauge, set_ptr->device_params, set_ptr->device_u_1dim_send_vec[_B_Y_],
                    set_ptr->device_u_1dim_send_vec[_F_Y_]);
                pick_up_u_z<<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                              set_ptr->stream_dims[_Z_]>>>(
                    gauge, set_ptr->device_params, set_ptr->device_u_1dim_send_vec[_B_Z_],
                    set_ptr->device_u_1dim_send_vec[_F_Z_]);
                pick_up_u_t<<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                              set_ptr->stream_dims[_T_]>>>(
                    gauge, set_ptr->device_params, set_ptr->device_u_1dim_send_vec[_B_T_],
                    set_ptr->device_u_1dim_send_vec[_F_T_]);
            }
            {
                // u_2dim_send
                pick_up_u_xy<<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _XY_],
                               set_ptr->blockDim, 0, set_ptr->stream>>>(
                    gauge, set_ptr->device_params,
                    set_ptr->device_u_2dim_send_vec[_B_X_B_Y_],
                    set_ptr->device_u_2dim_send_vec[_F_X_B_Y_],
                    set_ptr->device_u_2dim_send_vec[_B_X_F_Y_],
                    set_ptr->device_u_2dim_send_vec[_F_X_F_Y_]);
                pick_up_u_xz<<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _XZ_],
                               set_ptr->blockDim, 0, set_ptr->stream>>>(
                    gauge, set_ptr->device_params,
                    set_ptr->device_u_2dim_send_vec[_B_X_B_Z_],
                    set_ptr->device_u_2dim_send_vec[_F_X_B_Z_],
                    set_ptr->device_u_2dim_send_vec[_B_X_F_Z_],
                    set_ptr->device_u_2dim_send_vec[_F_X_F_Z_]);
                pick_up_u_xt<<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _XT_],
                               set_ptr->blockDim, 0, set_ptr->stream>>>(
                    gauge, set_ptr->device_params,
                    set_ptr->device_u_2dim_send_vec[_B_X_B_T_],
                    set_ptr->device_u_2dim_send_vec[_F_X_B_T_],
                    set_ptr->device_u_2dim_send_vec[_B_X_F_T_],
                    set_ptr->device_u_2dim_send_vec[_F_X_F_T_]);
                pick_up_u_yz<<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _YZ_],
                               set_ptr->blockDim, 0, set_ptr->stream>>>(
                    gauge, set_ptr->device_params,
                    set_ptr->device_u_2dim_send_vec[_B_Y_B_Z_],
                    set_ptr->device_u_2dim_send_vec[_F_Y_B_Z_],
                    set_ptr->device_u_2dim_send_vec[_B_Y_F_Z_],
                    set_ptr->device_u_2dim_send_vec[_F_Y_F_Z_]);
                pick_up_u_yt<<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _YT_],
                               set_ptr->blockDim, 0, set_ptr->stream>>>(
                    gauge, set_ptr->device_params,
                    set_ptr->device_u_2dim_send_vec[_B_Y_B_T_],
                    set_ptr->device_u_2dim_send_vec[_F_Y_B_T_],
                    set_ptr->device_u_2dim_send_vec[_B_Y_F_T_],
                    set_ptr->device_u_2dim_send_vec[_F_Y_F_T_]);
                pick_up_u_zt<<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _ZT_],
                               set_ptr->blockDim, 0, set_ptr->stream>>>(
                    gauge, set_ptr->device_params,
                    set_ptr->device_u_2dim_send_vec[_B_Z_B_T_],
                    set_ptr->device_u_2dim_send_vec[_F_Z_B_T_],
                    set_ptr->device_u_2dim_send_vec[_B_Z_F_T_],
                    set_ptr->device_u_2dim_send_vec[_F_Z_F_T_]);
            }
            // edge comm part
            {
                // u_1dim_comm
                {
                    // x edge part comm
                    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
                    ncclGroupStart();
                    ncclSend(set_ptr->device_u_1dim_send_vec[_B_X_],
                             set_ptr->lat_3dim[_X_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_B_X_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_X_]);
                    ncclRecv(set_ptr->device_u_1dim_recv_vec[_F_X_],
                             set_ptr->lat_3dim[_X_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_F_X_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_X_]);
                    ncclGroupEnd();
                    ncclGroupStart();
                    ncclSend(set_ptr->device_u_1dim_send_vec[_F_X_],
                             set_ptr->lat_3dim[_X_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_F_X_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_X_]);
                    ncclRecv(set_ptr->device_u_1dim_recv_vec[_B_X_],
                             set_ptr->lat_3dim[_X_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_B_X_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_X_]);
                    ncclGroupEnd();
                }
                {
                    // y edge part comm
                    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
                    ncclGroupStart();
                    ncclSend(set_ptr->device_u_1dim_send_vec[_B_Y_],
                             set_ptr->lat_3dim[_Y_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_B_Y_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_Y_]);
                    ncclRecv(set_ptr->device_u_1dim_recv_vec[_F_Y_],
                             set_ptr->lat_3dim[_Y_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_F_Y_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_Y_]);
                    ncclGroupEnd();
                    ncclGroupStart();
                    ncclSend(set_ptr->device_u_1dim_send_vec[_F_Y_],
                             set_ptr->lat_3dim[_Y_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_F_Y_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_Y_]);
                    ncclRecv(set_ptr->device_u_1dim_recv_vec[_B_Y_],
                             set_ptr->lat_3dim[_Y_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_B_Y_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_Y_]);
                    ncclGroupEnd();
                }
                {
                    // z edge part comm
                    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
                    ncclGroupStart();
                    ncclSend(set_ptr->device_u_1dim_send_vec[_B_Z_],
                             set_ptr->lat_3dim[_Z_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_B_Z_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_Z_]);
                    ncclRecv(set_ptr->device_u_1dim_recv_vec[_F_Z_],
                             set_ptr->lat_3dim[_Z_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_F_Z_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_Z_]);
                    ncclGroupEnd();
                    ncclGroupStart();
                    ncclSend(set_ptr->device_u_1dim_send_vec[_F_Z_],
                             set_ptr->lat_3dim[_Z_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_F_Z_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_Z_]);
                    ncclRecv(set_ptr->device_u_1dim_recv_vec[_B_Z_],
                             set_ptr->lat_3dim[_Z_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_B_Z_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_Z_]);
                    ncclGroupEnd();
                }
                {
                    // t edge part comm
                    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
                    ncclGroupStart();
                    ncclSend(set_ptr->device_u_1dim_send_vec[_B_T_],
                             set_ptr->lat_3dim[_T_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_B_T_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_T_]);
                    ncclRecv(set_ptr->device_u_1dim_recv_vec[_F_T_],
                             set_ptr->lat_3dim[_T_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_F_T_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_T_]);
                    ncclGroupEnd();
                    ncclGroupStart();
                    ncclSend(set_ptr->device_u_1dim_send_vec[_F_T_],
                             set_ptr->lat_3dim[_T_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_F_T_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_T_]);
                    ncclRecv(set_ptr->device_u_1dim_recv_vec[_B_T_],
                             set_ptr->lat_3dim[_T_] * _LAT_PDCC_ * _REAL_IMAG_, ncclDouble,
                             set_ptr->move_wards[_B_T_], set_ptr->nccl_comm,
                             set_ptr->stream_dims[_T_]);
                    ncclGroupEnd();
                }
                checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
                {
                    // u_2dim_comm
                    {
                        // xy edge part comm
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_B_X_B_Y_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BX_BY_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_a_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_F_X_F_Y_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FX_FY_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_a_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_F_X_B_Y_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FX_BY_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_b_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_B_X_F_Y_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BX_FY_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_b_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_B_X_F_Y_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BX_FY_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_c_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_F_X_B_Y_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FX_BY_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_c_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_F_X_F_Y_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FX_FY_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_d_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_B_X_B_Y_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BX_BY_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_d_]);
                        ncclGroupEnd();
                    }
                    {
                        // xz edge part comm
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_B_X_B_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BX_BZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_a_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_F_X_F_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FX_FZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_a_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_F_X_B_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FX_BZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_b_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_B_X_F_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BX_FZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_b_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_B_X_F_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BX_FZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_c_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_F_X_B_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FX_BZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_c_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_F_X_F_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FX_FZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_d_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_B_X_B_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BX_BZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_d_]);
                        ncclGroupEnd();
                    }
                    {
                        // xt edge part comm
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_B_X_B_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BX_BT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_a_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_F_X_F_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FX_FT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_a_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_F_X_B_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FX_BT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_b_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_B_X_F_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BX_FT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_b_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_B_X_F_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BX_FT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_c_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_F_X_B_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FX_BT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_c_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_F_X_F_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FX_FT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_d_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_B_X_B_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BX_BT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_d_]);
                        ncclGroupEnd();
                    }
                    {
                        // yz edge part comm
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_B_Y_B_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BY_BZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_a_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_F_Y_F_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FY_FZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_a_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_F_Y_B_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FY_BZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_b_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_B_Y_F_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BY_FZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_b_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_B_Y_F_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BY_FZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_c_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_F_Y_B_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FY_BZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_c_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_F_Y_F_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FY_FZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_d_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_B_Y_B_Z_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BY_BZ_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_d_]);
                        ncclGroupEnd();
                    }
                    {
                        // yt edge part comm
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_B_Y_B_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BY_BT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_a_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_F_Y_F_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FY_FT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_a_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_F_Y_B_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FY_BT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_b_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_B_Y_F_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BY_FT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_b_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_B_Y_F_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BY_FT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_c_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_F_Y_B_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FY_BT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_c_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_F_Y_F_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FY_FT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_d_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_B_Y_B_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BY_BT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_d_]);
                        ncclGroupEnd();
                    }
                    {
                        // zt edge part comm
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_B_Z_B_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BZ_BT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_a_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_F_Z_F_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FZ_FT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_a_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_F_Z_B_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FZ_BT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_b_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_B_Z_F_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BZ_FT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_b_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_B_Z_F_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BZ_FT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_c_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_F_Z_B_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FZ_BT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_c_]);
                        ncclGroupEnd();
                        ncclGroupStart();
                        ncclSend(set_ptr->device_u_2dim_send_vec[_F_Z_F_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_FZ_FT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_d_]);
                        ncclRecv(set_ptr->device_u_2dim_recv_vec[_B_Z_B_T_],
                                 set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                     _REAL_IMAG_,
                                 ncclDouble, set_ptr->move_wards[_BZ_BT_], set_ptr->nccl_comm,
                                 set_ptr->stream_dims[_d_]);
                        ncclGroupEnd();
                    }
                }
            }
            // edge recv part
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
            make_clover_all<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                              set_ptr->stream>>>(
                gauge, clover, set_ptr->device_params,
                set_ptr->device_u_1dim_recv_vec[_B_X_],
                set_ptr->device_u_1dim_recv_vec[_F_X_],
                set_ptr->device_u_1dim_recv_vec[_B_Y_],
                set_ptr->device_u_1dim_recv_vec[_F_Y_],
                set_ptr->device_u_1dim_recv_vec[_B_Z_],
                set_ptr->device_u_1dim_recv_vec[_F_Z_],
                set_ptr->device_u_1dim_recv_vec[_B_T_],
                set_ptr->device_u_1dim_recv_vec[_F_T_],
                set_ptr->device_u_2dim_recv_vec[_B_X_B_Y_],
                set_ptr->device_u_2dim_recv_vec[_F_X_B_Y_],
                set_ptr->device_u_2dim_recv_vec[_B_X_F_Y_],
                set_ptr->device_u_2dim_recv_vec[_F_X_F_Y_],
                set_ptr->device_u_2dim_recv_vec[_B_X_B_Z_],
                set_ptr->device_u_2dim_recv_vec[_F_X_B_Z_],
                set_ptr->device_u_2dim_recv_vec[_B_X_F_Z_],
                set_ptr->device_u_2dim_recv_vec[_F_X_F_Z_],
                set_ptr->device_u_2dim_recv_vec[_B_X_B_T_],
                set_ptr->device_u_2dim_recv_vec[_F_X_B_T_],
                set_ptr->device_u_2dim_recv_vec[_B_X_F_T_],
                set_ptr->device_u_2dim_recv_vec[_F_X_F_T_],
                set_ptr->device_u_2dim_recv_vec[_B_Y_B_Z_],
                set_ptr->device_u_2dim_recv_vec[_F_Y_B_Z_],
                set_ptr->device_u_2dim_recv_vec[_B_Y_F_Z_],
                set_ptr->device_u_2dim_recv_vec[_F_Y_F_Z_],
                set_ptr->device_u_2dim_recv_vec[_B_Y_B_T_],
                set_ptr->device_u_2dim_recv_vec[_F_Y_B_T_],
                set_ptr->device_u_2dim_recv_vec[_B_Y_F_T_],
                set_ptr->device_u_2dim_recv_vec[_F_Y_F_T_],
                set_ptr->device_u_2dim_recv_vec[_B_Z_B_T_],
                set_ptr->device_u_2dim_recv_vec[_F_Z_B_T_],
                set_ptr->device_u_2dim_recv_vec[_B_Z_F_T_],
                set_ptr->device_u_2dim_recv_vec[_F_Z_F_T_]);
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
        }
        void _make_mpi(void *gauge)
        {
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
            // edge send part
            {
                // u_1dim_send
                pick_up_u_x<<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                              set_ptr->stream_dims[_X_]>>>(
                    gauge, set_ptr->device_params, set_ptr->device_u_1dim_send_vec[_B_X_],
                    set_ptr->device_u_1dim_send_vec[_F_X_]);
                pick_up_u_y<<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                              set_ptr->stream_dims[_Y_]>>>(
                    gauge, set_ptr->device_params, set_ptr->device_u_1dim_send_vec[_B_Y_],
                    set_ptr->device_u_1dim_send_vec[_F_Y_]);
                pick_up_u_z<<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                              set_ptr->stream_dims[_Z_]>>>(
                    gauge, set_ptr->device_params, set_ptr->device_u_1dim_send_vec[_B_Z_],
                    set_ptr->device_u_1dim_send_vec[_F_Z_]);
                pick_up_u_t<<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                              set_ptr->stream_dims[_T_]>>>(
                    gauge, set_ptr->device_params, set_ptr->device_u_1dim_send_vec[_B_T_],
                    set_ptr->device_u_1dim_send_vec[_F_T_]);
            }
            {
                // u_2dim_send
                pick_up_u_xy<<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _XY_],
                               set_ptr->blockDim, 0, set_ptr->stream>>>(
                    gauge, set_ptr->device_params,
                    set_ptr->device_u_2dim_send_vec[_B_X_B_Y_],
                    set_ptr->device_u_2dim_send_vec[_F_X_B_Y_],
                    set_ptr->device_u_2dim_send_vec[_B_X_F_Y_],
                    set_ptr->device_u_2dim_send_vec[_F_X_F_Y_]);
                pick_up_u_xz<<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _XZ_],
                               set_ptr->blockDim, 0, set_ptr->stream>>>(
                    gauge, set_ptr->device_params,
                    set_ptr->device_u_2dim_send_vec[_B_X_B_Z_],
                    set_ptr->device_u_2dim_send_vec[_F_X_B_Z_],
                    set_ptr->device_u_2dim_send_vec[_B_X_F_Z_],
                    set_ptr->device_u_2dim_send_vec[_F_X_F_Z_]);
                pick_up_u_xt<<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _XT_],
                               set_ptr->blockDim, 0, set_ptr->stream>>>(
                    gauge, set_ptr->device_params,
                    set_ptr->device_u_2dim_send_vec[_B_X_B_T_],
                    set_ptr->device_u_2dim_send_vec[_F_X_B_T_],
                    set_ptr->device_u_2dim_send_vec[_B_X_F_T_],
                    set_ptr->device_u_2dim_send_vec[_F_X_F_T_]);
                pick_up_u_yz<<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _YZ_],
                               set_ptr->blockDim, 0, set_ptr->stream>>>(
                    gauge, set_ptr->device_params,
                    set_ptr->device_u_2dim_send_vec[_B_Y_B_Z_],
                    set_ptr->device_u_2dim_send_vec[_F_Y_B_Z_],
                    set_ptr->device_u_2dim_send_vec[_B_Y_F_Z_],
                    set_ptr->device_u_2dim_send_vec[_F_Y_F_Z_]);
                pick_up_u_yt<<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _YT_],
                               set_ptr->blockDim, 0, set_ptr->stream>>>(
                    gauge, set_ptr->device_params,
                    set_ptr->device_u_2dim_send_vec[_B_Y_B_T_],
                    set_ptr->device_u_2dim_send_vec[_F_Y_B_T_],
                    set_ptr->device_u_2dim_send_vec[_B_Y_F_T_],
                    set_ptr->device_u_2dim_send_vec[_F_Y_F_T_]);
                pick_up_u_zt<<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _ZT_],
                               set_ptr->blockDim, 0, set_ptr->stream>>>(
                    gauge, set_ptr->device_params,
                    set_ptr->device_u_2dim_send_vec[_B_Z_B_T_],
                    set_ptr->device_u_2dim_send_vec[_F_Z_B_T_],
                    set_ptr->device_u_2dim_send_vec[_B_Z_F_T_],
                    set_ptr->device_u_2dim_send_vec[_F_Z_F_T_]);
            }
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
            // edge comm part
            {
                { // d2h
                    for (int i = 0; i < _DIM_; i++)
                    {
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->host_u_1dim_send_vec[i * _BF_],
                            set_ptr->device_u_1dim_send_vec[i * _BF_],
                            set_ptr->lat_3dim[i] * _LAT_PDCC_ * sizeof(LatticeComplex), cudaMemcpyDeviceToHost,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->host_u_1dim_send_vec[i * _BF_ + 1],
                            set_ptr->device_u_1dim_send_vec[i * _BF_ + 1],
                            set_ptr->lat_3dim[i] * _LAT_PDCC_ * sizeof(LatticeComplex), cudaMemcpyDeviceToHost,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->host_u_1dim_recv_vec[i * _BF_],
                            set_ptr->device_u_1dim_recv_vec[i * _BF_],
                            set_ptr->lat_3dim[i] * _LAT_PDCC_ * sizeof(LatticeComplex), cudaMemcpyDeviceToHost,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->host_u_1dim_recv_vec[i * _BF_ + 1],
                            set_ptr->device_u_1dim_recv_vec[i * _BF_ + 1],
                            set_ptr->lat_3dim[i] * _LAT_PDCC_ * sizeof(LatticeComplex), cudaMemcpyDeviceToHost,
                            set_ptr->stream));
                    }
                    for (int i = 0; i < _2DIM_; i++)
                    {
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 0],
                            set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 0],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyDeviceToHost,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 0],
                            set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 0],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyDeviceToHost,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 1],
                            set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 1],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyDeviceToHost,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 1],
                            set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 1],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyDeviceToHost,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 2],
                            set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 2],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyDeviceToHost,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 2],
                            set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 2],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyDeviceToHost,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 3],
                            set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 3],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyDeviceToHost,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 3],
                            set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 3],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyDeviceToHost,
                            set_ptr->stream));
                    }
                }
            }
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
            {
                {
                    {
                    }
                    // x edge part comm

                    MPI_Sendrecv(set_ptr->host_u_1dim_send_vec[_B_X_],
                                 set_ptr->lat_3dim[_X_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_B_X_], 0, set_ptr->host_u_1dim_recv_vec[_F_X_],
                                 set_ptr->lat_3dim[_X_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_F_X_], 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                    MPI_Sendrecv(set_ptr->host_u_1dim_send_vec[_F_X_],
                                 set_ptr->lat_3dim[_X_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_F_X_], 0, set_ptr->host_u_1dim_recv_vec[_B_X_],
                                 set_ptr->lat_3dim[_X_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_B_X_], 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                }
                {
                    // y edge part comm

                    MPI_Sendrecv(set_ptr->host_u_1dim_send_vec[_B_Y_],
                                 set_ptr->lat_3dim[_Y_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_B_Y_], 0, set_ptr->host_u_1dim_recv_vec[_F_Y_],
                                 set_ptr->lat_3dim[_Y_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_F_Y_], 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                    MPI_Sendrecv(set_ptr->host_u_1dim_send_vec[_F_Y_],
                                 set_ptr->lat_3dim[_Y_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_F_Y_], 0, set_ptr->host_u_1dim_recv_vec[_B_Y_],
                                 set_ptr->lat_3dim[_Y_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_B_Y_], 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                }
                {
                    // z edge part comm

                    MPI_Sendrecv(set_ptr->host_u_1dim_send_vec[_B_Z_],
                                 set_ptr->lat_3dim[_Z_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_B_Z_], 0, set_ptr->host_u_1dim_recv_vec[_F_Z_],
                                 set_ptr->lat_3dim[_Z_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_F_Z_], 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                    MPI_Sendrecv(set_ptr->host_u_1dim_send_vec[_F_Z_],
                                 set_ptr->lat_3dim[_Z_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_F_Z_], 0, set_ptr->host_u_1dim_recv_vec[_B_Z_],
                                 set_ptr->lat_3dim[_Z_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_B_Z_], 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                }
                {
                    // t edge part comm

                    MPI_Sendrecv(set_ptr->host_u_1dim_send_vec[_B_T_],
                                 set_ptr->lat_3dim[_T_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_B_T_], 0, set_ptr->host_u_1dim_recv_vec[_F_T_],
                                 set_ptr->lat_3dim[_T_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_F_T_], 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                    MPI_Sendrecv(set_ptr->host_u_1dim_send_vec[_F_T_],
                                 set_ptr->lat_3dim[_T_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_F_T_], 0, set_ptr->host_u_1dim_recv_vec[_B_T_],
                                 set_ptr->lat_3dim[_T_] * _LAT_PDCC_ * _REAL_IMAG_, MPI_DOUBLE,
                                 set_ptr->move_wards[_B_T_], 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                }
                {
                    // u_2dim_comm
                    {
                        // xy edge part comm
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_B_X_B_Y_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BX_BY_], 0, set_ptr->host_u_2dim_recv_vec[_F_X_F_Y_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FX_FY_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_F_X_B_Y_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FX_BY_], 0, set_ptr->host_u_2dim_recv_vec[_B_X_F_Y_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BX_FY_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_B_X_F_Y_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BX_FY_], 0, set_ptr->host_u_2dim_recv_vec[_F_X_B_Y_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FX_BY_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_F_X_F_Y_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FX_FY_], 0, set_ptr->host_u_2dim_recv_vec[_B_X_B_Y_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BX_BY_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                    }
                    {
                        // xz edge part comm
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_B_X_B_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BX_BZ_], 0, set_ptr->host_u_2dim_recv_vec[_F_X_F_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FX_FZ_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_F_X_B_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FX_BZ_], 0, set_ptr->host_u_2dim_recv_vec[_B_X_F_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BX_FZ_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_B_X_F_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BX_FZ_], 0, set_ptr->host_u_2dim_recv_vec[_F_X_B_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FX_BZ_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_F_X_F_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FX_FZ_], 0, set_ptr->host_u_2dim_recv_vec[_B_X_B_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BX_BZ_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                    }
                    {
                        // xt edge part comm
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_B_X_B_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BX_BT_], 0, set_ptr->host_u_2dim_recv_vec[_F_X_F_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FX_FT_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_F_X_B_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FX_BT_], 0, set_ptr->host_u_2dim_recv_vec[_B_X_F_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BX_FT_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_B_X_F_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BX_FT_], 0, set_ptr->host_u_2dim_recv_vec[_F_X_B_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FX_BT_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_F_X_F_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FX_FT_], 0, set_ptr->host_u_2dim_recv_vec[_B_X_B_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BX_BT_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                    }
                    {
                        // yz edge part comm
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_B_Y_B_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BY_BZ_], 0, set_ptr->host_u_2dim_recv_vec[_F_Y_F_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FY_FZ_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_F_Y_B_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FY_BZ_], 0, set_ptr->host_u_2dim_recv_vec[_B_Y_F_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BY_FZ_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_B_Y_F_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BY_FZ_], 0, set_ptr->host_u_2dim_recv_vec[_F_Y_B_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FY_BZ_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_F_Y_F_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FY_FZ_], 0, set_ptr->host_u_2dim_recv_vec[_B_Y_B_Z_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BY_BZ_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                    }
                    {
                        // yt edge part comm
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_B_Y_B_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BY_BT_], 0, set_ptr->host_u_2dim_recv_vec[_F_Y_F_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FY_FT_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_F_Y_B_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FY_BT_], 0, set_ptr->host_u_2dim_recv_vec[_B_Y_F_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BY_FT_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_B_Y_F_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BY_FT_], 0, set_ptr->host_u_2dim_recv_vec[_F_Y_B_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FY_BT_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_F_Y_F_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FY_FT_], 0, set_ptr->host_u_2dim_recv_vec[_B_Y_B_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BY_BT_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                    }
                    {
                        // zt edge part comm
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_B_Z_B_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BZ_BT_], 0, set_ptr->host_u_2dim_recv_vec[_F_Z_F_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FZ_FT_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_F_Z_B_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FZ_BT_], 0, set_ptr->host_u_2dim_recv_vec[_B_Z_F_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BZ_FT_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_B_Z_F_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BZ_FT_], 0, set_ptr->host_u_2dim_recv_vec[_F_Z_B_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FZ_BT_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                        MPI_Sendrecv(set_ptr->host_u_2dim_send_vec[_F_Z_F_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_FZ_FT_], 0, set_ptr->host_u_2dim_recv_vec[_B_Z_B_T_],
                                     set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PDCC_ *
                                         _REAL_IMAG_,
                                     MPI_DOUBLE, set_ptr->move_wards[_BZ_BT_], 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                    }
                }
            }
            {
                { // h2d
                    for (int i = 0; i < _DIM_; i++)
                    {
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->device_u_1dim_send_vec[i * _BF_],
                            set_ptr->host_u_1dim_send_vec[i * _BF_],
                            set_ptr->lat_3dim[i] * _LAT_PDCC_ * sizeof(LatticeComplex), cudaMemcpyHostToDevice,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->device_u_1dim_send_vec[i * _BF_ + 1],
                            set_ptr->host_u_1dim_send_vec[i * _BF_ + 1],
                            set_ptr->lat_3dim[i] * _LAT_PDCC_ * sizeof(LatticeComplex), cudaMemcpyHostToDevice,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->device_u_1dim_recv_vec[i * _BF_],
                            set_ptr->host_u_1dim_recv_vec[i * _BF_],
                            set_ptr->lat_3dim[i] * _LAT_PDCC_ * sizeof(LatticeComplex), cudaMemcpyHostToDevice,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->device_u_1dim_recv_vec[i * _BF_ + 1],
                            set_ptr->host_u_1dim_recv_vec[i * _BF_ + 1],
                            set_ptr->lat_3dim[i] * _LAT_PDCC_ * sizeof(LatticeComplex), cudaMemcpyHostToDevice,
                            set_ptr->stream));
                    }
                    for (int i = 0; i < _2DIM_; i++)
                    {
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 0],
                            set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 0],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyHostToDevice,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 0],
                            set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 0],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyHostToDevice,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 1],
                            set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 1],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyHostToDevice,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 1],
                            set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 1],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyHostToDevice,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 2],
                            set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 2],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyHostToDevice,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 2],
                            set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 2],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyHostToDevice,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 3],
                            set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 3],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyHostToDevice,
                            set_ptr->stream));
                        checkCudaErrors(cudaMemcpyAsync(
                            set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 3],
                            set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 3],
                            set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PDCC_ * sizeof(LatticeComplex),
                            cudaMemcpyHostToDevice,
                            set_ptr->stream));
                    }
                }
            }
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
            // edge recv part
            make_clover_all<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                              set_ptr->stream>>>(
                gauge, clover, set_ptr->device_params,
                set_ptr->device_u_1dim_recv_vec[_B_X_],
                set_ptr->device_u_1dim_recv_vec[_F_X_],
                set_ptr->device_u_1dim_recv_vec[_B_Y_],
                set_ptr->device_u_1dim_recv_vec[_F_Y_],
                set_ptr->device_u_1dim_recv_vec[_B_Z_],
                set_ptr->device_u_1dim_recv_vec[_F_Z_],
                set_ptr->device_u_1dim_recv_vec[_B_T_],
                set_ptr->device_u_1dim_recv_vec[_F_T_],
                set_ptr->device_u_2dim_recv_vec[_B_X_B_Y_],
                set_ptr->device_u_2dim_recv_vec[_F_X_B_Y_],
                set_ptr->device_u_2dim_recv_vec[_B_X_F_Y_],
                set_ptr->device_u_2dim_recv_vec[_F_X_F_Y_],
                set_ptr->device_u_2dim_recv_vec[_B_X_B_Z_],
                set_ptr->device_u_2dim_recv_vec[_F_X_B_Z_],
                set_ptr->device_u_2dim_recv_vec[_B_X_F_Z_],
                set_ptr->device_u_2dim_recv_vec[_F_X_F_Z_],
                set_ptr->device_u_2dim_recv_vec[_B_X_B_T_],
                set_ptr->device_u_2dim_recv_vec[_F_X_B_T_],
                set_ptr->device_u_2dim_recv_vec[_B_X_F_T_],
                set_ptr->device_u_2dim_recv_vec[_F_X_F_T_],
                set_ptr->device_u_2dim_recv_vec[_B_Y_B_Z_],
                set_ptr->device_u_2dim_recv_vec[_F_Y_B_Z_],
                set_ptr->device_u_2dim_recv_vec[_B_Y_F_Z_],
                set_ptr->device_u_2dim_recv_vec[_F_Y_F_Z_],
                set_ptr->device_u_2dim_recv_vec[_B_Y_B_T_],
                set_ptr->device_u_2dim_recv_vec[_F_Y_B_T_],
                set_ptr->device_u_2dim_recv_vec[_B_Y_F_T_],
                set_ptr->device_u_2dim_recv_vec[_F_Y_F_T_],
                set_ptr->device_u_2dim_recv_vec[_B_Z_B_T_],
                set_ptr->device_u_2dim_recv_vec[_F_Z_B_T_],
                set_ptr->device_u_2dim_recv_vec[_B_Z_F_T_],
                set_ptr->device_u_2dim_recv_vec[_F_Z_F_T_]);
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
        }
        void make(void *gauge)
        {
            set_ptr->_print();
            // make clover
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
            auto start = std::chrono::high_resolution_clock::now();
            _make_nccl(gauge);
            _make_mpi(gauge);
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
        void inverse()
        {
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
        void give(void *fermion_out)
        {
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
        void end()
        {
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
            checkCudaErrors(cudaFreeAsync(clover, set_ptr->stream));
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        }
    };
}
#endif